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
#include <argparse/argparse.hpp>
#include "GitVersion.hpp"
#include "YAKL_netcdf.h"

#define WRITE_OUT
// TODO(cmo): Look at compacting particle buffer as they start to die (and maybe
// sort by position) to keep everything maximally coalesced

constexpr int32_t LutSize = 1024 * 1024;

#ifdef SCATTER_SINGLE_PRECISION

KOKKOS_INLINE_FUNCTION fp_t draw_u(RandState* s) {
    uint32_t i = Rand::next(s);
    return RandomTransforms::u32_to_unit_T<fp_t>(i);
}

KOKKOS_INLINE_FUNCTION fp_t box_muller_next(RandState* s) {
    uint32_t x = next(s);
    return RandomTransforms::u32_to_unit_T<fp_t>(x);
}

KOKKOS_INLINE_FUNCTION BoxMullerResult<fp_t> draw_2_n(RandState* s) {
    return RandomTransforms::box_muller(box_muller_next, s);
}

#else

KOKKOS_INLINE_FUNCTION fp_t draw_u(RandState* s) {
    uint64_t i = Rand::next(s);
    return RandomTransforms::u64_to_unit_T<fp_t>(i);
}

KOKKOS_INLINE_FUNCTION fp_t box_muller_next(RandState* s) {
    uint64_t x = next(s);
    return RandomTransforms::u64_to_unit_T<fp_t>(x);
}

KOKKOS_INLINE_FUNCTION BoxMullerResult<fp_t> draw_2_n(RandState* s) {
    return RandomTransforms::box_muller(box_muller_next, s);
}

#endif

struct SimParams {
    fp_t eps;
    fp_t Rinit;
    fp_t theta0;
    fp_t f_ratio;
    fp_t aniso;
    fp_t asym;

    fp_t omega0;
    fp_t omega_pe0;
    fp_t nu_s0;
    fp_t dt0;
    fp_t dt_save;

    fp_t Rstop;
    fp_t Rscat;
    fp_t Rtau1;

    uint64_t seed;

    void set_default_params() {
        eps = FP(0.1);
        Rinit = FP(1.75);
        Rstop = FP(215.0);
        theta0 = FP(0.0);
        f_ratio = FP(1.1);
        aniso = FP(0.3);
        asym = FP(1.0);
        seed = 110081;

        // NOTE(cmo): Sentinel to calculate this value
        dt_save = FP(-1.0);
    }
};

struct SimRadii {
    fp_t Rscat;
    fp_t Rtau1;
    fp_t Rstop;
};

struct ExtraArgs {
    std::string filename;
    i64 Nparticles;
};

SimRadii compute_sim_radii(SimParams p, SimState* s) {
    // NOTE(cmo): This is just a direct copy from the existing code, albeit
    // using trapezoidal integration instead. Called by init_particles.
    constexpr int IntegrationSize = 3999;
    yakl::Array<f64, 1, yakl::memHost> rint("rint", IntegrationSize);
    yakl::Array<f64, 1, yakl::memHost> tau_scat("tau_scat", IntegrationSize);

    for (int i = 0; i < IntegrationSize; ++i) {
        rint(i) = p.Rinit * (1.0 + fp_t(i) / 49.0 + 1e-3);
    }

    auto opac_i = [&](int i) {
        return nu_scat(rint(i), p.omega0, p.eps)
                / Constants::c_r / sqrt(1.0 - square(omega_pe(rint(i))) / square(p.omega0));
    };

    fp_t prev_opac = opac_i(IntegrationSize - 1);
    for (int i = IntegrationSize - 2; i > -1; --i) {
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

    for (int i = 0; i < IntegrationSize; ++i) {
        if (tau_scat(i) <= 1.0 && r_tau1 < 0.0) {
            r_tau1 = rint(i);
        }
        if (tau_scat(i) <= 0.1 && r_scat < 0.0) {
            r_scat = rint(i);
        }
        if (tau_scat(i) <= square(0.1 * 30 * Constants::Pi / 180.0 / f_start)
            && r_stop < 0.0) {
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

void init_luts(const SimParams& p, SimState* s) {
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

SimState init_particles(i64 Nparticles, SimParams* p) {
    SimState result;
    result.init(Nparticles);

    // NOTE(cmo): Seed states
    auto host_rand_states = result.rand_states.createHostObject();
    host_rand_states(0) = Rand::seed_state(p->seed);
    for (int i = 1; i < Nparticles; ++i) {
        host_rand_states(i) = Rand::copy_jump(&host_rand_states(i-1));
    }
    result.rand_states = host_rand_states.createDeviceCopy();

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

    // NOTE(cmo): Initial frequencies and k
    fp_t f_pe0 = omega_pe(p->Rinit) / C::TwoPi;
    fp_t omega_pe0 = omega_pe(p->Rinit);
    fp_t omega = p->f_ratio * omega_pe(p->Rinit);
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
    f64 mean_mu = mu_sum / Nparticles;
    p->omega0 = omega;
    p->omega_pe0 = omega_pe0;

    p->nu_s0 = nu_scat(p->Rinit, omega, p->eps);
    result.nu_s = p->nu_s0;

    SimRadii radii = compute_sim_radii(*p, &result);
    p->Rstop = radii.Rstop;
    p->Rscat = radii.Rscat;
    p->Rtau1 = radii.Rtau1;

    fp_t f_start = omega_pe0 / 1e6 / C::TwoPi;
    fp_t exp_size = 1.25 * 30.0 / f_start;
    p->dt0 = 0.01 * exp_size / C::c_r;
    if (p->dt_save < FP(0.0)) {
        p->dt_save = (p->Rstop - p->Rinit) / C::c_r / 10.0;
    }

    init_luts(*p, &result);

    return result;
}

KOKKOS_INLINE_FUNCTION void advance_dtsave_kernel(const SimParams& p, const SimState& state, i64 idx) {
    // NOTE(cmo): This kernel doesn't advance state->time (as it is shared),
    // but does move all particles by p.dt_save (or until their exit).
    // Once out of this kernel, increment state->time in host code.
    JasUnpack(state, Nparticles);
    namespace C = Constants;
    fp_t time0 = state.time;
    fp_t particle_time = state.time;
    const fp_t dt = p.dt_save;

    if (idx >= Nparticles) {
        return;
    }

    if (!state.active(idx)) {
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
    while (particle_time - time0 < dt) {
        fp_t dt_step = p.dt0;
        if (abs(time0 + dt - particle_time) < FP(1e-6)) {
            break;
        }
        // NOTE(cmo): Compute state vars and timestep
        r = std::sqrt(square(rx) + square(ry) + square(rz));
        kc = std::sqrt(square(kx) + square(ky) + square(kz));
        fp_t omega_pe_r = state.omega_pe(r);
        omega = sqrt(square(omega_pe_r) + square(kc));

        nu_s = nu_scat(r, omega, p.eps, &state);
        nu_s = min(nu_s, p.nu_s0);

        fp_t dt_ref = std::abs(kc / (state.domega_dr(r) * C::c_r) / FP(20.0));
        fp_t dt_dr = r / (C::c_r / p.omega0 * kc) / FP(20.0);
        dt_step = min(dt_step, fp_t(FP(0.1) / nu_s));
        dt_step = min(dt_step, dt_ref);
        dt_step = min(dt_step, dt_dr);

        if (particle_time + dt_step > time0 + dt) {
            dt_step = time0 + dt - particle_time;
        }

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

        // fp_t kc_old = kc;

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
        if (kc_z <= FP(0.0)) {
            z_asym = (FP(2.0) - p.asym);
        }
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
        if (r > p.Rstop) {
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

ExtraArgs parse_arguments(int argc, char** argv, SimParams* p) {
    argparse::ArgumentParser program("Scatter", GIT_HASH);
    program
        .add_argument("--eps")
        .scan<'g', f64>()
        .help("Scattering epsilon.")
        .default_value(f64(p->eps))
        .metavar("EPS");
    program
        .add_argument("--r-init")
        .scan<'g', f64>()
        .help("initial position (R_sun). Sets plasma frequency.")
        .default_value(f64(p->Rinit))
        .metavar("R_INIT");
    program
        .add_argument("--theta0")
        .scan<'g', f64>()
        .help("Initial angle (degrees).")
        .default_value(f64(p->theta0))
        .metavar("THETA0");
    program
        .add_argument("--f-ratio")
        .scan<'g', f64>()
        .help("Ratio of emission to plasma frequency. 2.0 for harmonic.")
        .default_value(f64(p->f_ratio))
        .metavar("F_RATIO");
    program
        .add_argument("--aniso")
        .scan<'g', f64>()
        .help("Density fluctuation anisotropy, aniso = q_parallel / q_perp_x. 1 => isotropic, 0 => 2D density fluctuations, -> infty => quasi-parallel fluctuations.")
        .default_value(f64(p->aniso))
        .metavar("ANISO");
    program
        .add_argument("--asym")
        .scan<'g', f64>()
        .help("Density fluctuation asymmetry along r. 1 => symmetric, 0 < asym < 1 => more outward density fluctuation (stronger scattering inwards), 1 < asym < 2 => more inward density fluctuation (stronger scattering outwards).")
        .default_value(f64(p->asym))
        .metavar("ASYM");
    program
        .add_argument("--seed")
        .scan<'u', u64>()
        .help("Random seed for simulation.")
        .default_value(u64(p->seed))
        .metavar("SEED");
    program
        .add_argument("--n-particles")
        .scan<'i', i64>()
        .help("Number of particles to simulate.")
        .default_value(i64(1024 * 1024))
        .metavar("N_PARTICLES");
    program
        .add_argument("--dt-save")
        .scan<'g', f64>()
        .help("Output timestep (set to -1 to calculate internally).")
        .default_value(f64(p->dt_save))
        .metavar("dt_save");
    program
        .add_argument("--filename")
        .help("Name for output file")
        .default_value(std::string("scatter.nc"))
        .metavar("FILE");
    program.add_epilog("Scattering for radio emission in the solar corona, using Kokkos.");

    program.parse_known_args(argc, argv);

    p->eps = program.get<f64>("--eps");
    p->Rinit = program.get<f64>("--r-init");
    p->theta0 = program.get<f64>("--theta0");
    p->f_ratio = program.get<f64>("--f-ratio");
    p->aniso = program.get<f64>("--aniso");
    p->asym = program.get<f64>("--asym");
    p->seed = program.get<u64>("--seed");
    p->dt_save = program.get<f64>("--dt-save");

    ExtraArgs extras;
    extras.filename = program.get<std::string>("--filename");
    extras.Nparticles = program.get<i64>("--n-particles");
    return extras;
}

void write_output_time(yakl::SimpleNetCDF& nc, const SimState& state) {
    std::string time_name("time");
    int time_idx = nc.getDimSize(time_name);
    nc.write1(f64(state.time), "time", time_idx, time_name);
    nc.write1(state.r, "r", {"n_particles"}, time_idx, time_name);
    nc.write1(state.rx, "rx", {"n_particles"}, time_idx, time_name);
    nc.write1(state.ry, "ry", {"n_particles"}, time_idx, time_name);
    nc.write1(state.rz, "rz", {"n_particles"}, time_idx, time_name);

    nc.write1(state.kc, "kc", {"n_particles"}, time_idx, time_name);
    nc.write1(state.kx, "kx", {"n_particles"}, time_idx, time_name);
    nc.write1(state.ky, "ky", {"n_particles"}, time_idx, time_name);
    nc.write1(state.kz, "kz", {"n_particles"}, time_idx, time_name);

    nc.write1(state.active, "active", {"n_particles"}, time_idx, time_name);
    nc.write1(state.omega, "omega", {"n_particles"}, time_idx, time_name);
    nc.write1(state.nu_s, "nu_s", {"n_particles"}, time_idx, time_name);
}

yakl::SimpleNetCDF setup_output(const std::string& path, const SimParams& params, const SimState& state) {
    yakl::SimpleNetCDF nc;
    nc.create(path, yakl::NETCDF_MODE_REPLACE);

    nc.createDim("n_particles", state.Nparticles);
    // NOTE(cmo): This is an unlimited dim, so we can write into it over time.
    nc.createDim("time");

    // NOTE(cmo): Write attributes
    const auto ncwrap = [] (int ierr, int line) {
        if (ierr != NC_NOERR) {
            printf("NetCDF Error writing attributes at scatter.cpp:%d\n", line);
            printf("%s\n",nc_strerror(ierr));
            Kokkos::abort(nc_strerror(ierr));
        }
    };

    int ncid = nc.file.ncid;
    std::string program = "scatter";
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "program", program.size(), program.c_str()),
        __LINE__
    );
    std::string git_hash(GIT_HASH);
    ncwrap(
        nc_put_att_text(ncid, NC_GLOBAL, "git_hash", git_hash.size(), git_hash.c_str()),
        __LINE__
    );

    f64 r_init = params.Rinit;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "r_init", NC_DOUBLE, 1, &r_init),
        __LINE__
    );
    f64 theta0 = params.theta0;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "theta0", NC_DOUBLE, 1, &theta0),
        __LINE__
    );
    f64 f_pe = params.omega_pe0 / 1e6 / Constants::TwoPi;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "f_pe", NC_DOUBLE, 1, &f_pe),
        __LINE__
    );
    f64 f_emit = f_pe * params.f_ratio;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "f_emit", NC_DOUBLE, 1, &f_emit),
        __LINE__
    );
    f64 f_ratio = params.f_ratio;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "f_ratio", NC_DOUBLE, 1, &f_ratio),
        __LINE__
    );
    f64 r_scat = params.Rscat;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "r_scat", NC_DOUBLE, 1, &r_scat),
        __LINE__
    );
    f64 r_stop = params.Rstop;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "r_stop", NC_DOUBLE, 1, &r_stop),
        __LINE__
    );
    f64 scattering_time = 1.0 / params.nu_s0;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "scattering_time", NC_DOUBLE, 1, &scattering_time),
        __LINE__
    );
    f64 eps = params.eps;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "eps", NC_DOUBLE, 1, &eps),
        __LINE__
    );
    f64 aniso = params.aniso;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "aniso", NC_DOUBLE, 1, &aniso),
        __LINE__
    );
    f64 asym = params.asym;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "asym", NC_DOUBLE, 1, &asym),
        __LINE__
    );
    f64 dt_save = params.dt_save;
    ncwrap(
        nc_put_att_double(ncid, NC_GLOBAL, "dt_save", NC_DOUBLE, 1, &dt_save),
        __LINE__
    );
    unsigned long long seed = params.seed;
    ncwrap(
        nc_put_att_ulonglong(ncid, NC_GLOBAL, "seed", NC_UINT64, 1, &seed),
        __LINE__
    );

    // NOTE(cmo): Write initial output
    write_output_time(nc, state);

    return nc;
}

int main(int argc, char* argv[]) {
    SimParams params;
    params.set_default_params();
    ExtraArgs args = parse_arguments(argc, argv, &params);

    Kokkos::initialize(argc, argv);
    yakl::init();
    {
        SimState state = init_particles(args.Nparticles, &params);

        f64 f_start = params.omega_pe0 / 1e6 / Constants::TwoPi;
        fmt::println("=== Initialisation ===");
        fmt::println("    Source located at {:.2f} R_sun, Angle = {:.2f} degrees", params.Rinit, params.theta0);
        fmt::println("    f_pe = {:.3e} MHz for emission frequency = {:.3e} MHz", f_start, f_start * params.f_ratio);
        fmt::println("    scattering radius = {:.2f} R_sun", params.Rscat);
        fmt::println("    scattering time = {:.2f} s", 1.0 / params.nu_s0);
        fmt::println("    calculation stopping distance = {:.2f} R_sun", params.Rstop);
        fmt::println("=== Key Parameters ===");
        fmt::println("    eps = {:.2f}", params.eps);
        fmt::println("    anisotropy = {:.2f}", params.aniso);
        fmt::println("    asymmetry = {:.2f}", params.asym);
        fmt::println("    dt_save = {:.2f} s", params.dt_save);

        int count = count_active(state);
        fmt::println("Time: {} s, Starting particles: {}\n", state.time, count);

        yakl::SimpleNetCDF output = setup_output(args.filename, params, state);
        while (count >= state.Nparticles / 200) {
            advance_dtsave(params, state);
            state.time += params.dt_save;

            write_output_time(output, state);

            count = count_active(state);
            fp_t mean_r = yakl::intrinsics::sum(state.r);
            fp_t F = 0.0;
            fp_t mean_kx = yakl::intrinsics::sum(state.kx);
            fp_t mean_ky = yakl::intrinsics::sum(state.ky);
            fp_t mean_kz = yakl::intrinsics::sum(state.kz);
            fp_t mean_nus = yakl::intrinsics::sum(state.nu_s);
            dex_parallel_reduce(
                FlatLoop<1>(state.Nparticles),
                KOKKOS_LAMBDA(i64 i, fp_t& Facc) {
                    Facc += std::sqrt(square(state.omega_pe(state.r(i))) + square(state.kc(i))) / state.omega(i);
                },
                Kokkos::Sum<fp_t>(F)
            );
            mean_r /= state.Nparticles;
            F /= state.Nparticles;
            mean_kx /= state.Nparticles;
            mean_ky /= state.Nparticles;
            mean_kz /= state.Nparticles;
            mean_nus /= state.Nparticles;

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
