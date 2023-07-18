#include "Constants.h"
#include <cstdio>
#include <cstring>
#include "JasPP.hpp"
#include "State.hpp"
#include "Random.hpp"
#include "DensityModelCuda.h"
#include "ScatteringModel.h"
#include "SimpleLut.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#endif

#ifdef WRITE_OUT
#include "cnpy.h"
#endif

constexpr int32_t LutSize = 1024 * 1024;

fp_t draw_u(RandState* s)
{
    uint64_t i = Rand::next(s);
    return RandomTransforms::u64_to_unit_T<fp_t>(i);
}

CudaFn BoxMullerResult<fp_t> draw_2_n(RandState* s)
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
    fp_t* rint = (fp_t*)calloc(IntegrationSize, sizeof(fp_t));
    fp_t* tau_scat = (fp_t*)calloc(IntegrationSize, sizeof(fp_t));

    for (int i = 0; i < IntegrationSize; ++i)
    {
        rint[i] = p.Rinit * (1.0 + fp_t(i) / 49.0 + 1e-3);
    }

    auto opac_i = [&](int i)
    {
        return nu_scat(rint[i], p.omega0, p.eps)
                / Constants::c_r / sqrt(1.0 - square(omega_pe(rint[i])) / square(p.omega0));
    };

    fp_t prev_opac = opac_i(IntegrationSize - 1);
    for (int i = IntegrationSize - 2; i > -1; --i)
    {
        fp_t opac = opac_i(i);
        tau_scat[i] = tau_scat[i+1]
                       + (rint[i+1] - rint[i])
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
        if (tau_scat[i] <= 1.0 && r_tau1 < 0.0)
        {
            r_tau1 = rint[i];
        }
        if (tau_scat[i] <= 0.1 && r_scat < 0.0)
        {
            r_scat = rint[i];
            // break;
            // NOTE(cmo): This is safe as tau must monotonically decrease
            // with distance (under standard assumptions).
            // Not sure of the magnitude of the block below, so removing this
            // for now.
        }
        if (tau_scat[i] <= square(0.1 * 30 * Constants::Pi / 180.0 / f_start)
            && r_stop < 0.0)
        {
            r_stop = rint[i];
        }
    }

    r_stop = max(r_stop, max(r_scat, 2.0 * p.Rinit));
    r_stop = min(r_stop, 215.0);

    free(rint);
    free(tau_scat);

    SimRadii result;
    result.Rscat = r_scat;
    result.Rstop = r_stop;
    result.Rtau1 = r_tau1;
    return result;
}

void init_luts(SimParams p, SimState* s)
{
    f64 r_min = 0.0;
    f64 r_max = p.Rstop * 1.1;

    printf("\n------\nComputing LUTs\n------\n");
    s->density_r.init(r_min, r_max, LutSize, density_r);
    printf("density done\n");
    s->omega_pe.init(r_min, r_max, LutSize, omega_pe);
    printf("omega_pe done\n");
    s->domega_dr.init(r_min, r_max, LutSize, domega_dr);
    printf("domega_dr done\n");
    printf("\n------\nLUTs Done\n------\n");
}

// TODO(cmo): update malloc for non __CUDACC__ case
SimState* init_particles(int Nparticles, SimParams* p)
{
    SimState* true_result;
    checkCudaErrors(cudaMallocManaged(&true_result, sizeof(SimState)));
    SimState& result(*true_result);
    result.Nparticles = Nparticles;
    result.time = 0.0;
    checkCudaErrors(cudaMallocManaged(&result.active, Nparticles * sizeof(int32_t)));
    for (int i = 0; i < Nparticles; ++i)
        result.active[i] = 1;

    const int allocSize = Nparticles * sizeof(fp_t);
    checkCudaErrors(cudaMallocManaged(&result.r, allocSize));
    checkCudaErrors(cudaMallocManaged(&result.rx, allocSize));
    checkCudaErrors(cudaMallocManaged(&result.ry, allocSize));
    checkCudaErrors(cudaMallocManaged(&result.rz, allocSize));
    checkCudaErrors(cudaMallocManaged(&result.kc, allocSize));
    checkCudaErrors(cudaMallocManaged(&result.kx, allocSize));
    checkCudaErrors(cudaMallocManaged(&result.ky, allocSize));
    checkCudaErrors(cudaMallocManaged(&result.kz, allocSize));
    checkCudaErrors(cudaMallocManaged(&result.omega, allocSize));
    checkCudaErrors(cudaMallocManaged(&result.nu_s, allocSize));
    memset(result.r, 0, allocSize);
    memset(result.rx, 0, allocSize);
    memset(result.ry, 0, allocSize);
    memset(result.rz, 0, allocSize);
    memset(result.kc, 0, allocSize);
    memset(result.kx, 0, allocSize);
    memset(result.ky, 0, allocSize);
    memset(result.kz, 0, allocSize);
    memset(result.omega, 0, allocSize);
    memset(result.nu_s, 0, allocSize);

    checkCudaErrors(cudaMallocManaged(&result.randStates, Nparticles * sizeof(RandState)));

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

    SimRadii radii = compute_sim_radii(*p, &result);
    p->Rstop = radii.Rstop;
    p->Rscat = radii.Rscat;
    p->Rtau1 = radii.Rtau1;
    printf("Stopping radius: %f Rs\n", p->Rstop);

    fp_t f_start = omega_pe0 / 1e6 / C::TwoPi;
    fp_t exp_size = 1.25 * 30.0 / f_start;
    p->dt0 = 0.01 * exp_size / C::c_r;
    p->dtSave = 2e7 / omega;
    p->dtSave = (p->Rstop - p->Rinit) / C::c_r / 10.0;

    init_luts(*p, &result);

    return true_result;
}


void free_particles(SimState* state)
{
    cudaFree(state->active);
    cudaFree(state->r);
    cudaFree(state->rx);
    cudaFree(state->ry);
    cudaFree(state->rz);
    cudaFree(state->kc);
    cudaFree(state->kx);
    cudaFree(state->ky);
    cudaFree(state->kz);
    cudaFree(state->omega);
    cudaFree(state->nu_s);
    cudaFree(state->randStates);
    cudaFree(state);
}

CudaFn void advance_dtsave_kernel(SimParams p, SimState* state, int idx)
{
    // NOTE(cmo): This kernel doesn't advance state->time (as it is shared),
    // but does move all particles by p.dtSave (or until their exit).
    // Once out of this kernel, increment state->time in host code.
    JasUnpack((*state), Nparticles, r, rx, ry, rz);
    JasUnpack((*state), kc, kx, ky, kz);
    JasUnpack((*state), omega, nu_s);
    namespace C = Constants;
    fp_t time0 = state->time;
    fp_t particle_time = state->time;
    fp_t dt = p.dtSave;

    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= Nparticles)
    {
        return;
    }

    if (!state->active[idx])
    {
        return;
    }

    int iters = 0;
    while (particle_time - time0 < dt)
    {
        fp_t dt_step = p.dt0;
        if (abs(time0 + dt - particle_time) < fpl(1e-6))
            break;
        // NOTE(cmo): Compute state vars and timestep
        // TODO(cmo): Use LUTs for omega_pe, density, and domega_dr
        r[idx] = sqrt(square(rx[idx]) + square(ry[idx]) + square(rz[idx]));
        kc[idx] = sqrt(square(kx[idx]) + square(ky[idx]) + square(kz[idx]));
        // omega[idx] = sqrt(square(omega_pe(r[idx])) + square(kc[idx]));
        fp_t omega_pe_r = state->omega_pe(r[idx]);
        omega[idx] = sqrt(square(omega_pe_r) + square(kc[idx]));

        nu_s[idx] = nu_scat(r[idx], omega[idx], p.eps, state);
        nu_s[idx] = min(nu_s[idx], p.nu_s0);

        fp_t dt_ref = abs(kc[idx] / (state->domega_dr(r[idx]) * C::c_r) / fpl(20.0));
        fp_t dt_dr = r[idx] / (C::c_r / p.omega0 * kc[idx]) / fpl(20.0);
        dt_step = min(dt_step, fp_t(fpl(0.1) / nu_s[idx]));
        dt_step = min(dt_step, dt_ref);
        dt_step = min(dt_step, dt_dr);

        if (particle_time + dt_step > time0 + dt)
            dt_step = time0 + dt - particle_time;

        fp_t sqrt_dt = sqrt(dt_step);

        fp_t drx_dt = C::c_r / omega[idx] * kx[idx];
        fp_t dry_dt = C::c_r / omega[idx] * ky[idx];
        fp_t drz_dt = C::c_r / omega[idx] * kz[idx];

        auto res0 = draw_2_n(&state->randStates[idx]);
        auto res1 = draw_2_n(&state->randStates[idx]);
        fp_t wx = res0.z0 * sqrt_dt;
        fp_t wy = res0.z1 * sqrt_dt;
        fp_t wz = res1.z0 * sqrt_dt;

        // rotate to r-aligned
        fp_t phi = atan2(ry[idx], rx[idx]);
        fp_t sintheta = sqrt(fpl(1.0) - square(rz[idx]) / square(r[idx]));
        fp_t costheta = rz[idx] / r[idx];
        fp_t sinphi = sin(phi);
        fp_t cosphi = cos(phi);

        fp_t kc_old = kc[idx];

        fp_t kc_x = - kx[idx] * sinphi + ky[idx] * cosphi;
        fp_t kc_y = - kx[idx] * costheta * cosphi
                    - ky[idx] * costheta * sinphi
                    + kz[idx] * sintheta;
        fp_t kc_z =   kx[idx] * sintheta * cosphi
                    + ky[idx] * sintheta * sinphi
                    + kz[idx] * costheta;

        // scatter
        fp_t kw = wx*kc_x + wy*kc_y + wz*kc_z*p.aniso;
        fp_t Akc = sqrt(square(kc_x) + square(kc_y) + square(kc_z) * square(p.aniso));
        fp_t z_asym = p.asym;
        if (kc_z <= fpl(0.0))
            z_asym = (fpl(2.0) - p.asym);
        z_asym *= square(kc[idx] / Akc);

        fp_t aniso2 = square(p.aniso);
        fp_t aniso4 = square(aniso2);
        fp_t Akc2 = square(Akc);
        fp_t Akc3 = cube(Akc);
        fp_t Aperp = nu_s[idx] * z_asym * kc[idx] / Akc3
                        * (- (fpl(1.0) + aniso2) * Akc2
                        + fpl(3.0) * aniso2 * (aniso2 - fpl(1.0)) * square(kc_z))
                        * p.aniso;
        fp_t Apara = nu_s[idx] * z_asym * kc[idx] / Akc3
                        * ((fpl(-3.0) * aniso4 + aniso2) * Akc2
                        + fpl(3.0) * aniso4 * (aniso2 - fpl(1.0)) * square(kc_z))
                        * p.aniso;

        fp_t g0 = sqrt(nu_s[idx] * square(kc[idx]));
        fp_t Ag0 = g0 * sqrt(z_asym * p.aniso);

        kc_x +=  Aperp * kc_x * dt_step
                + Ag0 * (wx - kc_x * kw / Akc2);
        kc_y +=  Aperp * kc_y * dt_step
                + Ag0 * (wy - kc_y * kw / Akc2);
        kc_z +=  Apara * kc_z * dt_step
                + Ag0 * (wz - kc_z * kw * p.aniso / Akc2) * p.aniso;

        // rotate back to cartesian

        kx[idx] = -kc_x*sinphi - kc_y*costheta*cosphi + kc_z*sintheta*cosphi;
        ky[idx] =  kc_x*cosphi - kc_y*costheta*sinphi + kc_z*sintheta*sinphi;
        kz[idx] =  kc_y*sintheta + kc_z*costheta;

        fp_t kc_norm = sqrt(square(kx[idx]) + square(ky[idx]) + square(kz[idx]));
        kx[idx] *= kc[idx] / kc_norm;
        ky[idx] *= kc[idx] / kc_norm;
        kz[idx] *= kc[idx] / kc_norm;

        // do time integration
        // fp_t dk_dt = (omega_pe(r[i]) / omega[i]) * domega_dr(r[i]) * C::c_r;
        fp_t dk_dt = (omega_pe_r / omega[idx]) * state->domega_dr(r[idx]) * C::c_r;
        kx[idx] -= dk_dt * (rx[idx] / r[idx]) * dt_step;
        ky[idx] -= dk_dt * (ry[idx] / r[idx]) * dt_step;
        kz[idx] -= dk_dt * (rz[idx] / r[idx]) * dt_step;

        rx[idx] += drx_dt * dt_step;
        ry[idx] += dry_dt * dt_step;
        rz[idx] += drz_dt * dt_step;

        r[idx] = sqrt(square(rx[idx]) + square(ry[idx]) + square(rz[idx]));
        kc[idx] = sqrt(square(kx[idx]) + square(ky[idx]) + square(kz[idx]));
        omega_pe_r = state->omega_pe(r[idx]);

        // conserve frequency
        // fp_t kc_new_old = kc_old / kc[i];
        fp_t kc_new_old = sqrt(square(omega[idx]) - square(omega_pe_r));
        kc_new_old /= kc[idx];
        // kc_new_old = 1.0;
        kx[idx] *= kc_new_old;
        ky[idx] *= kc_new_old;
        kz[idx] *= kc_new_old;


        kc[idx] = sqrt(square(kx[idx]) + square(ky[idx]) + square(kz[idx]));
        particle_time += dt_step;

        iters += 1;
        if (r[idx] > p.Rstop)
        {
            state->active[idx] = 0;
            break;
        }
    }
}

void advance_dtsave_cpu(SimParams p, SimState* state)
{
    for (int idx = 0; idx < state->Nparticles; ++idx)
    {
        advance_dtsave_kernel(p, state, idx);
    }
}

__global__ void advance_dtsave_cuda(SimParams p, SimState* state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    advance_dtsave_kernel(p, state, idx);
}

int count_active(SimState* s)
{
    int count = 0;
    for (int i = 0; i < s->Nparticles; ++i)
        count += s->active[i];
    return count;
}

#ifdef WRITE_OUT
void write_positions(SimState* s)
{
#if 0
    char filename[1024] = {};
    snprintf(filename, 1024, "Output_%08.4f.txt", s->time);
    FILE* f = fopen(filename, "w");

    for (int i = 0; i < s->Nparticles; ++i)
    {
        fprintf(f, "%f %f %f\n", s->rx[i], s->ry[i], s->rz[i]);
    }

    fflush(f);
    fclose(f);
#else
    char filename[1024] = {};
    snprintf(filename, 1024, "Output_%08.4f.npz", s->time);
    fp_t* buf = (fp_t*)calloc(3 * s->Nparticles, sizeof(fp_t));
    for (int i = 0; i < s->Nparticles; ++i) {
        buf[3 * i] = s->rx[i];
        buf[3 * i + 1] = s->ry[i];
        buf[3 * i + 2] = s->rz[i];
    }
    cnpy::npz_save(filename, "r", buf, {(uint64_t)s->Nparticles, 3}, "w");
    for (int i = 0; i < s->Nparticles; ++i) {
        buf[3 * i] = s->kx[i];
        buf[3 * i + 1] = s->ky[i];
        buf[3 * i + 2] = s->kz[i];
    }
    cnpy::npz_save(filename, "k", buf, {(uint64_t)s->Nparticles, 3}, "a");
    cnpy::npz_save(filename, "time", &s->time, {1}, "a");

    free(buf);
#endif
}
#endif


// TODO(cmo): Check if we wrap over random state
// TODO(cmo): Optical depth
// TODO(cmo): Minimise writebacks to the global arrays in the cuda kernel.
int main(int argc, const char* argv[])
{
    findCudaDevice(argc, argv);
    constexpr int Nparticles = 1024;
    SimParams params = default_params();
    SimState* state = init_particles(Nparticles, &params);

    fp_t omega1 = omega_pe(1.75);
    fp_t omega2 = omega_pe(2.15);
    fp_t domega = domega_dr(3.0);
    fp_t nu = nu_scat(2.0, 221427600.0, params.eps);
    printf("omega: %f %f\n", omega1, omega2);
    // omega: 201297810.663404 103774832.114953

    int count = count_active(state);
    printf("Time: %f s, Starting particles: %d\n", state->time, count);

#ifdef __CUDACC__
    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    int blockSize;
    int minGridSize;
    int gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, advance_dtsave_cuda);
    gridSize = (Nparticles + blockSize - 1) / blockSize;
    printf("CUDA Parameters: <<< %d, %d >>>\n", gridSize, blockSize);
#endif

#ifdef WRITE_OUT
    write_positions(state);
#endif
    while (count >= Nparticles / 200)
    {
#ifdef __CUDACC__
        advance_dtsave_cuda<<< gridSize, blockSize >>>(params, state);
        cudaDeviceSynchronize();
#else
        advance_dtsave_cpu(params, state);
#endif

        state->time += params.dtSave;
#ifdef WRITE_OUT
        write_positions(state);
#endif
        count = count_active(state);
        fp_t mean_r = 0.0;
        fp_t F = 0.0;
        fp_t mean_kz = 0.0;
        fp_t mean_ky = 0.0;
        fp_t mean_kx = 0.0;
        fp_t mean_nus = 0.0;
        for (int i = 0; i < Nparticles; ++i)
        {
            mean_r += state->r[i];
            F += std::sqrt(square(omega_pe(state->r[i])) + square(state->kc[i])) / state->omega[i];
            mean_kx += state->kx[i];
            mean_ky += state->ky[i];
            mean_kz += state->kz[i];
            mean_nus += state->nu_s[i];
        }
        mean_r /= Nparticles;
        F /= Nparticles;
        mean_kx /= Nparticles;
        mean_ky /= Nparticles;
        mean_kz /= Nparticles;
        mean_nus /= Nparticles;

        printf("Time: %f s, living particles: %d, <r>: %f\n", state->time, count, mean_r);
        printf("F: %f\n", F);
        printf("%f, %f, %e\n", mean_kx, mean_ky, mean_kz);
        printf("%f\n", mean_nus);
        // count = 0;
    }

    free_particles(state);
    return 0;
}

// cl /Ox /D "NDEGUG" /std:c++17 -nologo /Z7 -WL /MD /GL /arch:AVX2 /FC /EHsc Scatter.cpp /link /LTCG /OUT:scatter.exe /DEBUG:FULL

#ifdef WRITE_OUT
#include "cnpy.cpp" // NOTE(cmo): Need to link with `-lz` for zlib.
#endif