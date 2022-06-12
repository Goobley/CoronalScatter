#if !defined(SCATTERING_MODEL_H)
#define SCATTERING_MODEL_H
#include "Constants.h"
#include "DensityModel.h"
#ifdef __cplusplus
extern "C" {
#endif

inline fp_t nu_scat_krupar(fp_t r, fp_t omega, fp_t eps)
{
    const fp_t Third = 1.0 / 3.0;
    const fp_t TwoThird = 2.0 / 3.0;
    // inner turbulence scale
    fp_t l_i = r * 1e5;
    // outer turbulence scale
    fp_t l_0 = 0.23 * 6.9e10 * pow(r, 0.82);

    fp_t w_pe = omega_pe(r);
    fp_t w_pe2 = square(w_pe);
    fp_t w_pe4 = square(w_pe2);
    fp_t nu_s = Pi * square(eps) / (pow(l_i, Third) * pow(l_0, TwoThird));
    nu_s *= w_pe4 * c_light / omega / pow(square(omega) - w_pe2, 1.5);
    return nu_s;
}

inline fp_t nu_scat(fp_t r, fp_t omega, fp_t eps)
{
    return nu_scat_krupar(r, omega, eps);
}

#ifdef __cplusplus
}
#endif
#else
#endif