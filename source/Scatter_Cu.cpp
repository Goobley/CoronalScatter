#include "Constants.hpp"
#include <cstdio>
#include <cstring>
#include <fmt/core.h>
#include "LoopUtils.hpp"
#include "JasPP.hpp"
#include "State.hpp"
#include "Random.hpp"
#include "DensityModel.hpp"
#include "ScatteringModel.hpp"
#include "SimpleLut.hpp"
#include "cnpy.h"

// #define WRITE_OUT
// TODO(cmo): Look at compacting particle buffer as they start to die (and maybe
// sort by position) to keep everything maximally coalesced

constexpr int32_t LutSize = 1024 * 1024;

#ifdef SCATTER_SINGLE_PRECISION

KOKKOS_INLINE_FUNCTION fp_t draw_u(RandState* s)
{
    uint32_t i = Rand::next(s);
    return RandomTransforms::u32_to_unit_T<fp_t>(i);
}

KOKKOS_INLINE_FUNCTION fp_t box_muller_next(RandState* s)
{
    uint32_t x = next(s);
    return RandomTransforms::u32_to_unit_T<fp_t>(x);
}

KOKKOS_INLINE_FUNCTION BoxMullerResult<fp_t> draw_2_n(RandState* s)
{
    return RandomTransforms::box_muller(box_muller_next, s);
}

#else

KOKKOS_INLINE_FUNCTION fp_t draw_u(RandState* s)
{
    uint64_t i = Rand::next(s);
    return RandomTransforms::u64_to_unit_T<fp_t>(i);
}

KOKKOS_INLINE_FUNCTION fp_t box_muller_next(RandState* s)
{
    uint64_t x = next(s);
    return RandomTransforms::u64_to_unit_T<fp_t>(x);
}

KOKKOS_INLINE_FUNCTION BoxMullerResult<fp_t> draw_2_n(RandState* s)
{
    return RandomTransforms::box_muller(box_muller_next, s);
}

#endif

struct SimParams
{
    fp_t eps;
    fp_t Rinit;
    fp_t Rstop;
    fp_t Rscat;
    fp_t Rtau1;
    fp_t aniso;
    fp_t fRatio;
    fp_t asym;
    fp_t theta0;
    fp_t omega0;
    fp_t nu_s0;
    fp_t dt0;
    fp_t dtSave;
    uint64_t seed;
};

struct SimRadii
{
    fp_t Rscat;
    fp_t Rtau1;
    fp_t Rstop;
};

SimParams default_params()
{
    SimParams result;
    result.eps = 0.1;
    result.Rinit = 1.75;
    result.Rstop =  215.0; // Max val, 1 a.u.
    result.aniso = 0.3;
    result.fRatio = 1.1;
    result.asym = 1.0;
    result.theta0 = 0.0;
    result.nu_s0 = 0.0;
    result.seed = 110081;
    return result;
}

SimRadii compute_sim_radii(SimParams p, SimState* s)
{
    // NOTE(cmo): This is just a direct copy from the existing code, albeit
    // using trapezoidal integration instead. Called by init_particles.
    constexpr int IntegrationSize = 3999;
    yakl::Array<f64, 1, yakl::memHost> rint("rint", IntegrationSize);
    yakl::Array<f64, 1, yakl::memHost> tau_scat("tau_scat", IntegrationSize);

    for (int i = 0; i < IntegrationSize; ++i)
    {
        rint(i) = p.Rinit * (1.0 + fp_t(i) / 49.0 + 1e-3);
    }

    auto opac_i = [&](int i)
    {
        return nu_scat(rint(i), p.omega0, p.eps)
                / Constants::c_r / sqrt(1.0 - square(omega_pe(rint(i))) / square(p.omega0));
    };

    fp_t prev_opac = opac_i(IntegrationSize - 1);
    for (int i = IntegrationSize - 2; i > -1; --i)
    {
        fp_t opac = opac_i(i);
        tau_scat(i) = tau_scat(i+1)
                       + (rint(i+1) - rint(i))
                         * 0.5 * (opac + prev_opac);
        prev_opac = opac;
    }

    fp_t r_scat = -1.0; // NOTE(cmo): These are negative as sentinel
    fp_t r_tau1 = -1.0;
    fp_t r_stop = -1.0;

    fp_t omega_pe0 = omega_pe(p.Rinit);
    fp_t f_start = omega_pe0 / 1e6 / Constants::TwoPi;

    for (int i = 0; i < IntegrationSize; ++i)
    {
        if (tau_scat(i) <= 1.0 && r_tau1 < 0.0)
        {
            r_tau1 = rint(i);
        }
        if (tau_scat(i) <= 0.1 && r_scat < 0.0)
        {
            r_scat = rint(i);
        }
        if (tau_scat(i) <= square(0.1 * 30 * Constants::Pi / 180.0 / f_start)
            && r_stop < 0.0)
        {
            r_stop = rint(i);
        }
    }

    r_stop = max(r_stop, max(r_scat, FP(2.0) * p.Rinit));
    r_stop = min(r_stop, FP(215.0));

    SimRadii result;
    result.Rscat = r_scat;
    result.Rstop = r_stop;
    result.Rtau1 = r_tau1;
    return result;
}

void init_luts(const SimParams& p, SimState* s)
{
    f64 r_min = 0.0;
    f64 r_max = p.Rstop * 1.1;

    fmt::println("\n------\nComputing LUTs\n------");
    s->density_r.init(r_min, r_max, LutSize, density_r);
    fmt::println("density done");
    s->omega_pe.init(r_min, r_max, LutSize, omega_pe);
    fmt::println("omega_pe done");
    s->domega_dr.init(r_min, r_max, LutSize, domega_dr);
    fmt::println("domega_dr done");
    fmt::println("\n------\nLUTs Done\n------");
}

template <typename T>
inline void cmo_alloc(T** ptr, size_t size)
{
#ifdef __CUDACC__
    cudaMallocManaged(ptr, size);
#else
    *ptr = (T*)malloc(size);
#endif
}

template <typename T>
inline void cmo_free(T* ptr)
{
#ifdef __CUDACC__
    cudaFree(ptr);
#else
    free(ptr);
#endif

}

SimState init_particles(i64 Nparticles, SimParams* p)
{
    SimState result;
    result.init(Nparticles);

    // NOTE(cmo): Seed states
    auto host_rand_states = result.rand_states.createHostObject();
    host_rand_states(0) = Rand::seed_state(p->seed);
    for (int i = 1; i < Nparticles; ++i)
        host_rand_states(i) = Rand::copy_jump(&host_rand_states(i-1));
    result.rand_states = host_rand_states.createDeviceCopy();
    fmt::println("~~ {}", Nparticles);

    namespace C = Constants;
    // NOTE(cmo): Distribute initial particles
    const fp_t Rinit = p->Rinit;
    const fp_t theta0 = p->theta0;
    dex_parallel_for(
        "Init particle pos",
        FlatLoop<1>(Nparticles),
        KOKKOS_LAMBDA (i64 i) {
            fp_t r = Rinit;
            fp_t rTheta = theta0 * C::Pi / FP(180.0);
            fp_t rPhi = FP(0.0);

            result.r(i) = r;
            result.rz(i) = r * std::cos(rTheta);
            result.rx(i) = r * std::sin(rTheta) * std::cos(rPhi);
            result.ry(i) = r * std::sin(rTheta) * std::sin(rPhi);
        }
    );
    fmt::println("~~");

    // NOTE(cmo): Initial frequencies and k
    fp_t f_pe0 = omega_pe(p->Rinit) / C::TwoPi;
    fp_t omega_pe0 = omega_pe(p->Rinit);
    fp_t omega = p->fRatio * omega_pe(p->Rinit);
    fp_t kc0 = std::sqrt(square(omega) - square(omega_pe0));
    fmt::println("kc0: {}", kc0);
    f64 mu_sum = 0.0;
    dex_parallel_reduce(
        "Init particle k/omega",
        FlatLoop<1>(Nparticles),
        KOKKOS_LAMBDA (i64 i, f64& mu_s) {
            fp_t mu = draw_u(&result.rand_states(i));
            fp_t phi = draw_u(&result.rand_states(i)) * C::TwoPi;
            mu_s += mu;

            result.kc(i) = kc0;
            result.kz(i) = kc0 * mu;
            result.kx(i) = kc0 * std::sqrt(FP(1.0) - square(mu)) * std::cos(phi);
            result.ky(i) = kc0 * std::sqrt(FP(1.0) - square(mu)) * std::sin(phi);
            result.omega(i) = omega;
        },
        Kokkos::Sum<f64>(mu_sum)
    );
    fmt::println("~~");
    f64 mean_mu = mu_sum / Nparticles;
    fmt::println("mean_mu: {}, mean_kz: {}", mean_mu, kc0 * mean_mu);
    p->omega0 = omega;

    p->nu_s0 = nu_scat(p->Rinit, omega, p->eps);
    result.nu_s = p->nu_s0;
    fmt::println("~~");

    SimRadii radii = compute_sim_radii(*p, &result);
    p->Rstop = radii.Rstop;
    p->Rscat = radii.Rscat;
    p->Rtau1 = radii.Rtau1;
    fmt::println("Stopping radius: {} Rs\n", p->Rstop);
    fmt::println("~~");

    fp_t f_start = omega_pe0 / 1e6 / C::TwoPi;
    fp_t exp_size = 1.25 * 30.0 / f_start;
    p->dt0 = 0.01 * exp_size / C::c_r;
    p->dtSave = 2e7 / omega;
    p->dtSave = (p->Rstop - p->Rinit) / C::c_r / 10.0;

    init_luts(*p, &result);

    return result;
}

KOKKOS_INLINE_FUNCTION void advance_dtsave_kernel(const SimParams& p, const SimState& state, i64 idx)
{
    // NOTE(cmo): This kernel doesn't advance state->time (as it is shared),
    // but does move all particles by p.dtSave (or until their exit).
    // Once out of this kernel, increment state->time in host code.
    JasUnpack(state, Nparticles);
    namespace C = Constants;
    fp_t time0 = state.time;
    fp_t particle_time = state.time;
    const fp_t dt = p.dtSave;

    if (idx >= Nparticles)
    {
        return;
    }

    if (!state.active(idx))
    {
        return;
    }

    // NOTE(cmo): Pull everything out to registers (hopefully)
    fp_t r = state.r(idx);
    fp_t rx = state.rx(idx);
    fp_t ry = state.ry(idx);
    fp_t rz = state.rz(idx);
    fp_t kc = state.kc(idx);
    fp_t kx = state.kx(idx);
    fp_t ky = state.ky(idx);
    fp_t kz = state.kz(idx);
    fp_t omega = state.omega(idx);
    fp_t nu_s = state.nu_s(idx);
    RandState rand_state = state.rand_states(idx);

    int iters = 0;
    while (particle_time - time0 < dt)
    {
        fp_t dt_step = p.dt0;
        if (abs(time0 + dt - particle_time) < FP(1e-6))
            break;
        // NOTE(cmo): Compute state vars and timestep
        r = std::sqrt(square(rx) + square(ry) + square(rz));
        kc = std::sqrt(square(kx) + square(ky) + square(kz));
        // omega[idx] = sqrt(square(omega_pe(r[idx])) + square(kc[idx]));
        fp_t omega_pe_r = state.omega_pe(r);
        omega = sqrt(square(omega_pe_r) + square(kc));

        nu_s = nu_scat(r, omega, p.eps, &state);
        nu_s = min(nu_s, p.nu_s0);

        fp_t dt_ref = std::abs(kc / (state.domega_dr(r) * C::c_r) / FP(20.0));
        fp_t dt_dr = r / (C::c_r / p.omega0 * kc) / FP(20.0);
        dt_step = min(dt_step, fp_t(FP(0.1) / nu_s));
        dt_step = min(dt_step, dt_ref);
        dt_step = min(dt_step, dt_dr);

        if (particle_time + dt_step > time0 + dt)
            dt_step = time0 + dt - particle_time;

        fp_t sqrt_dt = std::sqrt(dt_step);

        fp_t drx_dt = C::c_r / omega * kx;
        fp_t dry_dt = C::c_r / omega * ky;
        fp_t drz_dt = C::c_r / omega * kz;

        auto res0 = draw_2_n(&rand_state);
        auto res1 = draw_2_n(&rand_state);
        fp_t wx = res0.z0 * sqrt_dt;
        fp_t wy = res0.z1 * sqrt_dt;
        fp_t wz = res1.z0 * sqrt_dt;

        // rotate to r-aligned
        fp_t phi = std::atan2(ry, rx);
        fp_t sintheta = std::sqrt(FP(1.0) - square(rz) / square(r));
        fp_t costheta = rz / r;
        fp_t sinphi = std::sin(phi);
        fp_t cosphi = std::cos(phi);

        fp_t kc_old = kc;

        fp_t kc_x = - kx * sinphi + ky * cosphi;
        fp_t kc_y = - kx * costheta * cosphi
                    - ky * costheta * sinphi
                    + kz * sintheta;
        fp_t kc_z =   kx * sintheta * cosphi
                    + ky * sintheta * sinphi
                    + kz * costheta;

        // scatter
        fp_t kw = wx*kc_x + wy*kc_y + wz*kc_z*p.aniso;
        fp_t Akc = std::sqrt(square(kc_x) + square(kc_y) + square(kc_z) * square(p.aniso));
        fp_t z_asym = p.asym;
        if (kc_z <= FP(0.0))
            z_asym = (FP(2.0) - p.asym);
        z_asym *= square(kc / Akc);

        fp_t aniso2 = square(p.aniso);
        fp_t aniso4 = square(aniso2);
        fp_t Akc2 = square(Akc);
        fp_t Akc3 = cube(Akc);
        fp_t Aperp = nu_s * z_asym * kc / Akc3
                        * (- (FP(1.0) + aniso2) * Akc2
                        + FP(3.0) * aniso2 * (aniso2 - FP(1.0)) * square(kc_z))
                        * p.aniso;
        fp_t Apara = nu_s * z_asym * kc / Akc3
                        * ((FP(-3.0) * aniso4 + aniso2) * Akc2
                        + FP(3.0) * aniso4 * (aniso2 - FP(1.0)) * square(kc_z))
                        * p.aniso;

        fp_t g0 = std::sqrt(nu_s * square(kc));
        fp_t Ag0 = g0 * std::sqrt(z_asym * p.aniso);

        kc_x +=  Aperp * kc_x * dt_step
                + Ag0 * (wx - kc_x * kw / Akc2);
        kc_y +=  Aperp * kc_y * dt_step
                + Ag0 * (wy - kc_y * kw / Akc2);
        kc_z +=  Apara * kc_z * dt_step
                + Ag0 * (wz - kc_z * kw * p.aniso / Akc2) * p.aniso;

        // rotate back to cartesian

        kx = -kc_x*sinphi - kc_y*costheta*cosphi + kc_z*sintheta*cosphi;
        ky =  kc_x*cosphi - kc_y*costheta*sinphi + kc_z*sintheta*sinphi;
        kz =  kc_y*sintheta + kc_z*costheta;

        fp_t kc_norm = std::sqrt(square(kx) + square(ky) + square(kz));
        kx *= kc / kc_norm;
        ky *= kc / kc_norm;
        kz *= kc / kc_norm;

        // do time integration
        // fp_t dk_dt = (omega_pe(r[i]) / omega[i]) * domega_dr(r[i]) * C::c_r;
        fp_t dk_dt = (omega_pe_r / omega) * state.domega_dr(r) * C::c_r;
        kx -= dk_dt * (rx / r) * dt_step;
        ky -= dk_dt * (ry / r) * dt_step;
        kz -= dk_dt * (rz / r) * dt_step;

        rx += drx_dt * dt_step;
        ry += dry_dt * dt_step;
        rz += drz_dt * dt_step;

        r = std::sqrt(square(rx) + square(ry) + square(rz));
        kc = std::sqrt(square(kx) + square(ky) + square(kz));
        omega_pe_r = state.omega_pe(r);

        // conserve frequency
        // fp_t kc_new_old = kc_old / kc[i];
        fp_t kc_new_old = sqrt(square(omega) - square(omega_pe_r));
        kc_new_old /= kc;
        // kc_new_old = 1.0;
        kx *= kc_new_old;
        ky *= kc_new_old;
        kz *= kc_new_old;


        kc = sqrt(square(kx) + square(ky) + square(kz));
        particle_time += dt_step;

        iters += 1;
        if (r > p.Rstop)
        {
            state.active(idx) = 0;
            break;
        }
    }

    // NOTE(cmo): Update everything we pulled out of global memory;
    state.r(idx) = r;
    state.rx(idx) = rx;
    state.ry(idx) = ry;
    state.rz(idx) = rz;
    state.kc(idx) = kc;
    state.kx(idx) = kx;
    state.ky(idx) = ky;
    state.kz(idx) = kz;
    state.omega(idx) = omega;
    state.nu_s(idx) = nu_s;
    state.rand_states(idx) = rand_state;
}

void advance_dtsave(const SimParams& p, const SimState& state) {
    dex_parallel_for(
        "Update particle loop",
        FlatLoop<1>(state.Nparticles),
        KOKKOS_LAMBDA (i64 idx) {
            advance_dtsave_kernel(p, state, idx);
        }
    );
    Kokkos::fence();
}


int count_active(const SimState& s)
{
    i64 count = 0;

    dex_parallel_reduce(
        "Count active",
        FlatLoop<1>(s.Nparticles),
        KOKKOS_LAMBDA (i64 i, i64& count) {
            if (s.active(i)) {
                count += 1;
            }
        },
        Kokkos::Sum<i64>(count)
    );
    return count;
}

#ifdef WRITE_OUT
void write_positions(const SimState& s)
{
#if 0
    char filename[1024] = {};
    snprintf(filename, 1024, "Output_%08.4f.txt", s.time);
    FILE* f = fopen(filename, "w");

    auto rx = s.rx.createHostCopy();
    auto ry = s.ry.createHostCopy();
    auto rz = s.rz.createHostCopy();

    for (int i = 0; i < s.Nparticles; ++i)
    {
        fprintf(f, "%f %f %f\n", rx(i), ry(i), rz(i));
    }

    fflush(f);
    fclose(f);
#else
    char filename[1024] = {};
    snprintf(filename, 1024, "Output_%08.4f.npz", s.time);

    auto rx = s.rx.createHostCopy();
    auto ry = s.ry.createHostCopy();
    auto rz = s.rz.createHostCopy();
    auto kx = s.kx.createHostCopy();
    auto ky = s.ky.createHostCopy();
    auto kz = s.kz.createHostCopy();
    fp_t* buf = (fp_t*)calloc(3 * s.Nparticles, sizeof(fp_t));
    for (int i = 0; i < s.Nparticles; ++i) {
        buf[3 * i] = rx(i);
        buf[3 * i + 1] = ry(i);
        buf[3 * i + 2] = rz(i);
    }
    cnpy::npz_save(filename, "r", buf, {(uint64_t)s.Nparticles, 3}, "w");
    for (int i = 0; i < s.Nparticles; ++i) {
        buf[3 * i] = kx(i);
        buf[3 * i + 1] = ky(i);
        buf[3 * i + 2] = kz(i);
    }
    cnpy::npz_save(filename, "k", buf, {(uint64_t)s.Nparticles, 3}, "a");
    cnpy::npz_save(filename, "time", &s.time, {1}, "a");

    free(buf);
#endif
}
#endif


// TODO(cmo): Check if we wrap over random state
// TODO(cmo): Optical depth
// TODO(cmo): Minimise writebacks to the global arrays in the cuda kernel.
int main(int argc, const char* argv[])
{
    Kokkos::initialize();
    yakl::init();
    {
        constexpr int Nparticles = 1024 * 1024;
        SimParams params = default_params();
        SimState state = init_particles(Nparticles, &params);

        fp_t omega1 = omega_pe(1.75);
        fp_t omega2 = omega_pe(2.15);
        fp_t domega = domega_dr(3.0);
        fp_t nu = nu_scat(2.0, 221427600.0, params.eps);
        fmt::println("omega: {}, {}, nu_s: {}", omega1, omega2, nu);
        // omega: 201297810.663404 103774832.114953

        int count = count_active(state);
        fmt::println("Time: {} s, Starting particles: {}\n", state.time, count);

    #ifdef WRITE_OUT
        write_positions(state);
    #endif
        while (count >= Nparticles / 200)
        {
            advance_dtsave(params, state);
            state.time += params.dtSave;

    #ifdef WRITE_OUT
            write_positions(state);
    #endif

            count = count_active(state);
            fp_t mean_r = yakl::intrinsics::sum(state.r);
            fp_t F = 0.0;
            fp_t mean_ky = yakl::intrinsics::sum(state.ky);
            fp_t mean_kx = yakl::intrinsics::sum(state.kx);
            fp_t mean_kz = yakl::intrinsics::sum(state.kz);
            fp_t mean_nus = yakl::intrinsics::sum(state.nu_s);
            dex_parallel_reduce(
                FlatLoop<1>(state.Nparticles),
                KOKKOS_LAMBDA(i64 i, fp_t& Facc) {
                    Facc += std::sqrt(square(state.omega_pe(state.r(i))) + square(state.kc(i))) / state.omega(i);
                },
                Kokkos::Sum<fp_t>(F)
            );
            mean_r /= Nparticles;
            F /= Nparticles;
            mean_kx /= Nparticles;
            mean_ky /= Nparticles;
            mean_kz /= Nparticles;
            mean_nus /= Nparticles;

            fmt::println("Time: {} s, living particles: {}, <r>: {}", state.time, count, mean_r);
            fmt::println("F: {}", F);
            fmt::println("{}, {}, {:e}", mean_kx, mean_ky, mean_kz);
            fmt::println("{}", mean_nus);
        }
    }

    yakl::finalize();
    Kokkos::finalize();

    return 0;
}

#include "cnpy.cpp"