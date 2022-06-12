#define _USE_MATH_DEFINES
#include <cstdio>
#include <cstring>
#include "JasPP.hpp"
#include "Random.hpp"
#include "density_model.h"

typedef double fp_t;
// typedef float fp_t;
#define WRITE_OUT 0

namespace ConstantsF64
{
    constexpr f64 Rs = 6.96e10;
    constexpr f64 c = 2.998e10;
    constexpr f64 c_r = c / Rs;
    constexpr f64 c_s = 2e7 / Rs;
    constexpr f64 au = 215.0;
    constexpr f64 Pi = M_PI;
    constexpr f64 TwoPi = 2.0 * M_PI;
}
namespace ConstantsF32
{
    constexpr float Rs = 6.96e10f;
    constexpr float c = 2.998e10f;
    constexpr float c_r = c / Rs;
    constexpr float c_s = 2e7f / Rs;
    constexpr float au = 215.0f;
    constexpr float Pi = M_PI;
    constexpr float TwoPi = 2.0f * M_PI;
}

template <typename T>
constexpr T square(T x)
{
    return x * x;
}

template <typename T>
constexpr T cube(T x)
{
    return x * x * x;
}

template <typename T>
constexpr T min(T a, T b)
{
    return a <= b ? a : b;
}

template <typename T>
constexpr T max(T a, T b)
{
    return a >= b ? a : b;
}

namespace Constants = ConstantsF64;
// namespace Constants = ConstantsF32;

template <typename RandState>
struct BaseSimState
{
    int64_t Nparticles;
    fp_t time;
    int32_t* active;

    fp_t* r;
    fp_t* rx;
    fp_t* ry;
    fp_t* rz;

    fp_t* kc;
    fp_t* kx;
    fp_t* ky;
    fp_t* kz;

    fp_t* omega;
    fp_t* nu_s;

    RandState* randStates;
};

using SimState = BaseSimState<Xoroshiro256StarStar::Xoro256State>;
namespace Rand = Xoroshiro256StarStar;
using RandomTransforms::BoxMullerResult;
typedef Rand::Xoro256State RandState;

fp_t draw_u(RandState* s)
{
    uint64_t i = Rand::next(s);
    return RandomTransforms::u64_to_unit_T<fp_t>(i);
}

BoxMullerResult<fp_t> draw_2_n(RandState* s)
{
    uint64_t i0 = next(s);
    uint64_t i1 = next(s);

    fp_t u0 = RandomTransforms::u64_to_unit_T<fp_t>(i0);
    fp_t u1 = RandomTransforms::u64_to_unit_T<fp_t>(i1);

    return RandomTransforms::box_muller(u0, u1);
}

struct SimParams
{
    fp_t eps;
    fp_t Rinit;
    fp_t Rstop;
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

SimParams default_params()
{
    SimParams result;
    result.eps = 0.1;
    result.Rinit = 1.75;
    // TODO(cmo): This needs to be calculated
    result.Rstop =  6.9660356;
    result.aniso = 0.3;
    result.fRatio = 1.1;
    result.asym = 1.0;
    result.theta0 = 0.0;
    result.nu_s0 = 0.0;
    result.seed = 110081;
    return result;
}

inline void omega_pe_dr(fp_t r, fp_t* result)
{
    result[0] = omega_pe(r);
    result[1] = domega_dr(r);
}

fp_t nu_scat_krupar(fp_t r, fp_t omega, fp_t eps)
{
    constexpr fp_t Third = 1.0 / 3.0;
    constexpr fp_t TwoThird = 2.0 / 3.0;
    namespace C = Constants;
    // inner turbulence scale
    fp_t l_i = r * 1e5;
    // outer turbulence scale
    fp_t l_0 = 0.23 * 6.9e10 * std::pow(r, 0.82);

    fp_t w_pe = omega_pe(r);
    fp_t w_pe2 = square(w_pe);
    fp_t w_pe4 = square(w_pe2);
    fp_t nu_s = C::Pi * square(eps) / (std::pow(l_i, Third) * std::pow(l_0, TwoThird));
    nu_s *= w_pe4 * C::c / omega / std::pow(square(omega) - w_pe2, 1.5);
    return nu_s;
}

fp_t nu_scat(fp_t r, fp_t omega, fp_t eps)
{
    return nu_scat_krupar(r, omega, eps);
}

SimState init_particles(int Nparticles, SimParams* p)
{
    // TODO(cmo): Switch to aligned allocation
    SimState result;
    result.Nparticles = Nparticles;
    result.time = 0.0;
    result.active = (int32_t*)calloc(Nparticles, sizeof(int32_t));
    for (int i = 0; i < Nparticles; ++i)
        result.active[i] = 1;

    result.r  = (fp_t*)calloc(Nparticles, sizeof(fp_t));
    result.rx = (fp_t*)calloc(Nparticles, sizeof(fp_t));
    result.ry = (fp_t*)calloc(Nparticles, sizeof(fp_t));
    result.rz = (fp_t*)calloc(Nparticles, sizeof(fp_t));
    result.kc = (fp_t*)calloc(Nparticles, sizeof(fp_t));
    result.kx = (fp_t*)calloc(Nparticles, sizeof(fp_t));
    result.ky = (fp_t*)calloc(Nparticles, sizeof(fp_t));
    result.kz = (fp_t*)calloc(Nparticles, sizeof(fp_t));
    result.omega = (fp_t*)calloc(Nparticles, sizeof(fp_t));
    result.nu_s = (fp_t*)calloc(Nparticles, sizeof(fp_t));

    result.randStates = (RandState*)calloc(Nparticles, sizeof(RandState));

    // NOTE(cmo): Seed states
    result.randStates[0] = Rand::seed_state(p->seed);
    for (int i = 1; i < Nparticles; ++i)
        result.randStates[i] = Rand::copy_jump(&result.randStates[i-1]);

    // NOTE(cmo): Distribute initial particles
    namespace C = Constants;
    for (int i = 0; i < Nparticles; ++i)
    {
        fp_t r = p->Rinit;
        fp_t rTheta = p->theta0 * C::Pi / 180.0;
        fp_t rPhi = 0.0;

        result.r[i] = r;
        result.rz[i] = r * std::cos(rTheta);
        result.rx[i] = r * std::sin(rTheta) * std::cos(rPhi);
        result.ry[i] = r * std::sin(rTheta) * std::sin(rPhi);
    }

    // NOTE(cmo): Initial frequencies and k
    fp_t f_pe0 = omega_pe(p->Rinit) / C::TwoPi;
    fp_t omega_pe0 = omega_pe(p->Rinit);
    fp_t omega = p->fRatio * omega_pe(p->Rinit);
    fp_t kc0 = std::sqrt(square(omega) - square(omega_pe0));
    printf("kc0: %f\n", kc0);
    fp_t mean_mu = 0.0;
    fp_t mean_kz = 0.0;
    for (int i = 0; i < Nparticles; ++i)
    {
        fp_t mu = draw_u(&result.randStates[i]);
        fp_t phi = draw_u(&result.randStates[i]) * C::TwoPi;
        mean_mu += mu;

        result.kc[i] = kc0;
        result.kz[i] = kc0 * mu;
        mean_kz += result.kz[i];
        result.kx[i] = kc0 * std::sqrt(1.0 - square(mu)) * std::cos(phi);
        result.ky[i] = kc0 * std::sqrt(1.0 - square(mu)) * std::sin(phi);
        result.omega[i] = omega;
    }
    mean_mu /= Nparticles;
    mean_kz /= Nparticles;
    printf("mean_mu: %f, mean_kz: %f\n", mean_mu, mean_kz);
    p->omega0 = omega;

    p->nu_s0 = nu_scat(p->Rinit, omega, p->eps);
    for (int i = 0; i < Nparticles; ++i)
    {
        result.nu_s[i] = p->nu_s0;
    }

    fp_t f_start = omega_pe0 / 1e6 / C::TwoPi;
    fp_t exp_size = 1.25 * 30.0 / f_start;
    p->dt0 = 0.01 * exp_size / C::c_r;
    p->dtSave = 2e7 / omega;
    p->dtSave = (p->Rstop - p->Rinit) / C::c_r / 10.0;

    return result;
}

void free_particles(SimState* state)
{
    free(state->active);
    free(state->r);
    free(state->rx);
    free(state->ry);
    free(state->rz);
    free(state->kc);
    free(state->kx);
    free(state->ky);
    free(state->kz);
    free(state->omega);
    free(state->nu_s);
    free(state->randStates);
}

void advance_dtsave(SimParams p, SimState* state)
{
    JasUnpack((*state), Nparticles, r, rx, ry, rz);
    JasUnpack((*state), kc, kx, ky, kz);
    JasUnpack((*state), omega, nu_s);
    namespace C = Constants;
    fp_t time0 = state->time;
    fp_t dt = p.dtSave;

    int iters = 0;
    while (state->time - time0 < dt)
    {
        fp_t dt_step = p.dt0;
        if (std::abs(time0 + dt - state->time) < 1e-6)
            break;
        // NOTE(cmo): Compute state vars and timestep
        for (int i = 0; i < Nparticles; ++i)
        {
            if (!state->active[i])
                continue;

            r[i] = std::sqrt(square(rx[i]) + square(ry[i]) + square(rz[i]));
            kc[i] = std::sqrt(square(kx[i]) + square(ky[i]) + square(kz[i]));
            omega[i] = std::sqrt(square(omega_pe(r[i])) + square(kc[i]));

            nu_s[i] = nu_scat(r[i], omega[i], p.eps);
            nu_s[i] = min(nu_s[i], p.nu_s0);

            fp_t dt_ref = std::abs(kc[i] / (domega_dr(r[i]) * C::c_r) / 20.0);
            fp_t dt_dr = r[i] / (C::c_r / p.omega0 * kc[i]) / 20.0;
            dt_step = min(dt_step, fp_t(0.1 / nu_s[i]));
            dt_step = min(dt_step, dt_ref);
            dt_step = min(dt_step, dt_dr);
        }
        if (state->time + dt_step > time0 + dt)
            dt_step = time0 + dt - state->time;

        fp_t sqrt_dt = std::sqrt(dt_step);
        for (int i = 0; i < Nparticles; ++i)
        {
            if (!state->active[i])
                continue;

            fp_t drx_dt = C::c_r / omega[i] * kx[i];
            fp_t dry_dt = C::c_r / omega[i] * ky[i];
            fp_t drz_dt = C::c_r / omega[i] * kz[i];

            auto res0 = draw_2_n(&state->randStates[i]);
            auto res1 = draw_2_n(&state->randStates[i]);
            fp_t wx = res0.z0 * sqrt_dt;
            fp_t wy = res0.z1 * sqrt_dt;
            fp_t wz = res1.z0 * sqrt_dt;

            // rotate to r-aligned
            fp_t phi = std::atan2(ry[i], rx[i]);
            fp_t sintheta = std::sqrt(1.0 - square(rz[i]) / square(r[i]));
            fp_t costheta = rz[i] / r[i];
            fp_t sinphi = std::sin(phi);
            fp_t cosphi = std::cos(phi);

            fp_t kc_old = kc[i];

            fp_t kc_x = - kx[i] * sinphi + ky[i] * cosphi;
            fp_t kc_y = - kx[i] * costheta * cosphi
                        - ky[i] * costheta * sinphi
                        + kz[i] * sintheta;
            fp_t kc_z =   kx[i] * sintheta * cosphi
                        + ky[i] * sintheta * sinphi
                        + kz[i] * costheta;

            // scatter
            fp_t kw = wx*kc_x + wy*kc_y + wz*kc_z*p.aniso;
            fp_t Akc = std::sqrt(square(kc_x) + square(kc_y) + square(kc_z) * square(p.aniso));
            fp_t z_asym = p.asym;
            if (kc_z <= 0.0)
                z_asym = (2.0 - p.asym);
            z_asym *= square(kc[i] / Akc);

            fp_t aniso2 = square(p.aniso);
            fp_t aniso4 = square(aniso2);
            fp_t Akc2 = square(Akc);
            fp_t Akc3 = cube(Akc);
            fp_t Aperp = nu_s[i] * z_asym * kc[i] / Akc3
                         * (- (1.0 + aniso2) * Akc2
                            + 3.0 * aniso2 * (aniso2 - 1.0) * square(kc_z))
                         * p.aniso;
            fp_t Apara = nu_s[i] * z_asym * kc[i] / Akc3
                         * ((-3.0 * aniso4 + aniso2) * Akc2
                            + 3.0 * aniso4 * (aniso2 - 1.0) * square(kc_z))
                         * p.aniso;

            fp_t g0 = std::sqrt(nu_s[i] * square(kc[i]));
            fp_t Ag0 = g0 * std::sqrt(z_asym * p.aniso);

            kc_x +=  Aperp * kc_x * dt_step
                   + Ag0 * (wx - kc_x * kw / Akc2);
            kc_y +=  Aperp * kc_y * dt_step
                   + Ag0 * (wy - kc_y * kw / Akc2);
            kc_z +=  Apara * kc_z * dt_step
                   + Ag0 * (wz - kc_z * kw * p.aniso / Akc2) * p.aniso;

            // rotate back to cartesian

            kx[i] = -kc_x*sinphi - kc_y*costheta*cosphi + kc_z*sintheta*cosphi;
            ky[i] =  kc_x*cosphi - kc_y*costheta*sinphi + kc_z*sintheta*sinphi;
            kz[i] =  kc_y*sintheta + kc_z*costheta;

            fp_t kc_norm = std::sqrt(square(kx[i]) + square(ky[i]) + square(kz[i]));
            kx[i] *= kc[i] / kc_norm;
            ky[i] *= kc[i] / kc_norm;
            kz[i] *= kc[i] / kc_norm;

            // do time integration
            // fp_t dk_dt = (omega_pe(r[i]) / omega[i]) * domega_dr(r[i]) * C::c_r;
            fp_t om[2];
            omega_pe_dr(r[i], om);
            fp_t dk_dt = (om[0] / omega[i]) * om[1] * C::c_r;
            kx[i] -= dk_dt * (rx[i] / r[i]) * dt_step;
            ky[i] -= dk_dt * (ry[i] / r[i]) * dt_step;
            kz[i] -= dk_dt * (rz[i] / r[i]) * dt_step;

            rx[i] += drx_dt * dt_step;
            ry[i] += dry_dt * dt_step;
            rz[i] += drz_dt * dt_step;

            r[i] = std::sqrt(square(rx[i]) + square(ry[i]) + square(rz[i]));
            kc[i] = std::sqrt(square(kx[i]) + square(ky[i]) + square(kz[i]));

            // conserve frequency
            // fp_t kc_new_old = kc_old / kc[i];
            fp_t kc_new_old = std::sqrt(square(omega[i]) - square(omega_pe(r[i])));
            kc_new_old /= kc[i];
            // kc_new_old = 1.0;
            kx[i] *= kc_new_old;
            ky[i] *= kc_new_old;
            kz[i] *= kc_new_old;


            kc[i] = std::sqrt(square(kx[i]) + square(ky[i]) + square(kz[i]));
        }
        state->time += dt_step;

        for (int i = 0; i < Nparticles; ++i)
        {
            if (state->active[i] && (r[i] > p.Rstop))
                state->active[i] = 0;
        }
        iters += 1;
    }
}


int count_active(SimState* s)
{
    int count = 0;
    for (int i = 0; i < s->Nparticles; ++i)
        count += s->active[i];
    return count;
}

void write_positions(SimState* s)
{
    char filename[1024] = {};
    snprintf(filename, 1024, "Output_%08.4f.txt", s->time);
    FILE* f = fopen(filename, "w");

    for (int i = 0; i < s->Nparticles; ++i)
    {
        fprintf(f, "%f %f %f\n", s->rx[i], s->ry[i], s->rz[i]);
    }

    fflush(f);
    fclose(f);
}


// TODO(cmo): Check if we wrap over random state
// TODO(cmo): Optical depth
int main(void)
{
    constexpr int Nparticles = 1024;
    SimParams params = default_params();
    SimState state = init_particles(Nparticles, &params);

    fp_t omega1 = omega_pe(1.75);
    fp_t omega2 = omega_pe(2.15);
    fp_t domega = domega_dr(3.0);
    fp_t nu = nu_scat(2.0, 221427600.0, params.eps);
    printf("omega: %f %f\n", omega1, omega2);
    // omega: 201297810.663404 103774832.114953

    int count = count_active(&state);
    printf("Time: %f s, Starting particles: %d\n", state.time, count);

    write_positions(&state);
    while (count >= Nparticles / 200)
    {
        advance_dtsave(params, &state);
#if WRITE_OUT
        write_positions(&state);
#endif
        count = count_active(&state);
        fp_t mean_r = 0.0;
        fp_t F = 0.0;
        fp_t mean_kz = 0.0;
        fp_t mean_ky = 0.0;
        fp_t mean_kx = 0.0;
        fp_t mean_nus = 0.0;
        for (int i = 0; i < Nparticles; ++i)
        {
            mean_r += state.r[i];
            F += std::sqrt(square(omega_pe(state.r[i])) + square(state.kc[i])) / state.omega[i];
            mean_kx += state.kx[i];
            mean_ky += state.ky[i];
            mean_kz += state.kz[i];
            mean_nus += state.nu_s[i];
        }
        mean_r /= Nparticles;
        F /= Nparticles;
        mean_kx /= Nparticles;
        mean_ky /= Nparticles;
        mean_kz /= Nparticles;
        mean_nus /= Nparticles;

        printf("Time: %f s, living particles: %d, <r>: %f\n", state.time, count, mean_r);
        printf("F: %f\n", F);
        printf("%f, %f, %e\n", mean_kx, mean_ky, mean_kz);
        printf("%f\n", mean_nus);
        // count = 0;
    }

    free_particles(&state);
    return 0;
}

// cl /Ox /D "NDEGUG" /std:c++17 -nologo /Z7 -WL /MD /GL /arch:AVX2 /FC /EHsc Scatter.cpp /link /LTCG /OUT:scatter.exe /DEBUG:FULL