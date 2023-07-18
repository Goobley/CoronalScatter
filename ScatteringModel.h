#if !defined(SCATTERING_MODEL_H)
#define SCATTERING_MODEL_H
#include "Constants.h"
#include "DensityModel.h"
#include "State.hpp"
#ifdef __cplusplus
extern "C" {
#endif


CudaFn inline fp_t nu_scat_krupar(fp_t r, fp_t omega, fp_t eps, SimState* s=nullptr)
{
#ifndef OriginalNuScatForm
#define OriginalNuScatForm 0
#endif
#if OriginalNuScatForm
    // NOTE(cmo): This function breaks in single precision.
    using fp_t = f64;
#endif
    namespace C = Constants;
    constexpr fp_t Third = fpl(1.0) / fpl(3.0);
    constexpr fp_t TwoThird = fpl(2.0) / fpl(3.0);
    // inner turbulence scale
    fp_t l_i = r * fpl(1e5);
    // outer turbulence scale
    fp_t l_0 = fpl(0.23) * fpl(6.9e10) * pow(r, fpl(0.82));

    fp_t w_pe;
    if (s)
    {
        w_pe = s->omega_pe(r);
    }
    else
    {
        w_pe = omega_pe(r);
    }

    fp_t w_pe2 = square(w_pe);
#if OriginalNuScatForm
    fp_t w_pe4 = square(w_pe2);
    fp_t nu_s = C::Pi * square(eps) / (pow(l_i, Third) * pow(l_0, TwoThird));
    nu_s *= w_pe4 * C::c / omega / pow(square(omega) - w_pe2, fpl(1.5));
#else
    fp_t diff_term = pow(square(omega) - w_pe2, fpl(1.5));
    fp_t t_1 = square(cbrt(l_0));
    fp_t t_2 = cbrt(l_i);
    fp_t nu_s = square(eps) * C::Pi * w_pe2 / (t_1 * t_2) / diff_term;
    nu_s *= w_pe2 / omega * C::c;
#endif
    return nu_s;
}

CudaFn inline fp_t nu_scat(fp_t r, fp_t omega, fp_t eps, SimState* s=nullptr)
{
    return nu_scat_krupar(r, omega, eps, s);
}

#ifdef __cplusplus
}
#endif
#else
#endif