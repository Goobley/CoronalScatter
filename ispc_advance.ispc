#include "Constants.h"
#include "SimState.h"
#include "DensityScatteringModel.h"

export void advance_dtsave_ispc(uniform SimParams* uniform p, uniform SimState* uniform state)
{
    uniform int64 Nparticles = state->Nparticles;
    uniform fp_t* r = state->r;
    uniform fp_t* rx = state->rx;
    uniform fp_t* ry = state->ry;
    uniform fp_t* rz = state->rz;
    uniform fp_t* kc = state->kc;
    uniform fp_t* kx = state->kx;
    uniform fp_t* ky = state->ky;
    uniform fp_t* kz = state->kz;
    uniform fp_t* omega = state->omega;
    uniform fp_t* nu_s = state->nu_s;
    uniform fp_t time0 = state->time;
    uniform fp_t dt = p->dtSave;

    uniform int iters = 0;
    while (state->time - time0 < dt)
    {
        fp_t dt_step = p->dt0;
        if (abs(time0 + dt - state->time) < 1e-6d)
            break;
        // NOTE(cmo): Compute state vars and timestep
        foreach (i = 0 ... Nparticles)
        {
            cif (!state->active[i])
                continue;

            r[i] = sqrt(square(rx[i]) + square(ry[i]) + square(rz[i]));
            kc[i] = sqrt(square(kx[i]) + square(ky[i]) + square(kz[i]));
            omega[i] = sqrt(square(omega_pe(r[i])) + square(kc[i]));

            nu_s[i] = nu_scat(r[i], omega[i], p->eps);
            nu_s[i] = min(nu_s[i], p->nu_s0);

            fp_t dt_ref = abs(kc[i] / (domega_dr(r[i]) * c_r) / 20.0d);
            fp_t dt_dr = r[i] / (c_r / p->omega0 * kc[i]) / 20.0d;
            dt_step = min(dt_step, 0.1d / nu_s[i]);
            dt_step = min(dt_step, dt_ref);
            dt_step = min(dt_step, dt_dr);
        }
        if (state->time + dt_step > time0 + dt)
            dt_step = time0 + dt - state->time;
        uniform fp_t udt_step = reduce_min(dt_step);
        uniform fp_t sqrt_dt = sqrt(udt_step);

        foreach (i = 0 ... Nparticles)
        {
            cif (!state->active[i])
                continue;

            fp_t drx_dt = c_r / omega[i] * kx[i];
            fp_t dry_dt = c_r / omega[i] * ky[i];
            fp_t drz_dt = c_r / omega[i] * kz[i];

            BoxMullerResult res0 = draw_2_n(state->randState, i);
            BoxMullerResult res1 = draw_2_n(state->randState, i);
            fp_t wx = res0.z0 * sqrt_dt;
            fp_t wy = res0.z1 * sqrt_dt;
            fp_t wz = res1.z0 * sqrt_dt;

            // rotate to r-aligned
            fp_t phi = atan2(ry[i], rx[i]);
            fp_t sintheta = sqrt(1.0d - square(rz[i]) / square(r[i]));
            fp_t costheta = rz[i] / r[i];
            fp_t sinphi = sin(phi);
            fp_t cosphi = cos(phi);

            fp_t kc_old = kc[i];

            fp_t kc_x = - kx[i] * sinphi + ky[i] * cosphi;
            fp_t kc_y = - kx[i] * costheta * cosphi
                        - ky[i] * costheta * sinphi
                        + kz[i] * sintheta;
            fp_t kc_z =   kx[i] * sintheta * cosphi
                        + ky[i] * sintheta * sinphi
                        + kz[i] * costheta;

            // scatter
            fp_t kw = wx*kc_x + wy*kc_y + wz*kc_z*p->aniso;
            fp_t Akc = sqrt(square(kc_x) + square(kc_y) + square(kc_z) * square(p->aniso));
            fp_t z_asym = p->asym;
            if (kc_z <= 0.0d)
                z_asym = (2.0d - p->asym);
            z_asym *= square(kc[i] / Akc);

            fp_t aniso2 = square(p->aniso);
            fp_t aniso4 = square(aniso2);
            fp_t Akc2 = square(Akc);
            fp_t Akc3 = cube(Akc);
            fp_t Aperp = nu_s[i] * z_asym * kc[i] / Akc3
                         * (- (1.0d + aniso2) * Akc2
                            + 3.0d * aniso2 * (aniso2 - 1.0d) * square(kc_z))
                         * p->aniso;
            fp_t Apara = nu_s[i] * z_asym * kc[i] / Akc3
                         * ((-3.0d * aniso4 + aniso2) * Akc2
                            + 3.0d * aniso4 * (aniso2 - 1.0d) * square(kc_z))
                         * p->aniso;

            fp_t g0 = sqrt(nu_s[i] * square(kc[i]));
            fp_t Ag0 = g0 * sqrt(z_asym * p->aniso);

            kc_x +=  Aperp * kc_x * udt_step
                   + Ag0 * (wx - kc_x * kw / Akc2);
            kc_y +=  Aperp * kc_y * udt_step
                   + Ag0 * (wy - kc_y * kw / Akc2);
            kc_z +=  Apara * kc_z * udt_step
                   + Ag0 * (wz - kc_z * kw * p->aniso / Akc2) * p->aniso;

            // rotate back to cartesian

            kx[i] = -kc_x*sinphi - kc_y*costheta*cosphi + kc_z*sintheta*cosphi;
            ky[i] =  kc_x*cosphi - kc_y*costheta*sinphi + kc_z*sintheta*sinphi;
            kz[i] =  kc_y*sintheta + kc_z*costheta;

            fp_t kc_norm = sqrt(square(kx[i]) + square(ky[i]) + square(kz[i]));
            kx[i] *= kc[i] / kc_norm;
            ky[i] *= kc[i] / kc_norm;
            kz[i] *= kc[i] / kc_norm;

            // do time integration
            fp_t dk_dt = (omega_pe(r[i]) / omega[i]) * domega_dr(r[i]) * c_r;
            // fp_t om[2];
            // omega_pe_dr(r[i], om);
            // fp_t dk_dt = (om[0] / omega[i]) * om[1] * c_r;
            kx[i] -= dk_dt * (rx[i] / r[i]) * udt_step;
            ky[i] -= dk_dt * (ry[i] / r[i]) * udt_step;
            kz[i] -= dk_dt * (rz[i] / r[i]) * udt_step;

            rx[i] += drx_dt * udt_step;
            ry[i] += dry_dt * udt_step;
            rz[i] += drz_dt * udt_step;

            r[i] = sqrt(square(rx[i]) + square(ry[i]) + square(rz[i]));
            kc[i] = sqrt(square(kx[i]) + square(ky[i]) + square(kz[i]));

            // conserve frequency
            // fp_t kc_new_old = kc_old / kc[i];
            fp_t kc_new_old = sqrt(square(omega[i]) - square(omega_pe(r[i])));
            kc_new_old /= kc[i];
            // kc_new_old = 1.0;
            kx[i] *= kc_new_old;
            ky[i] *= kc_new_old;
            kz[i] *= kc_new_old;


            kc[i] = sqrt(square(kx[i]) + square(ky[i]) + square(kz[i]));
        }
        state->time += udt_step;

        foreach (i = 0 ... Nparticles)
        {
            if (state->active[i] && (r[i] > p->Rstop))
                state->active[i] = 0;
                // TODO(cmo): Add stop time for particle
        }
        iters += 1;
        break;
    }
}