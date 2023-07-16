#include "cuda_runtime.h"
#if !defined(DENSITY_MODEL_H)
#define DENSITY_MODEL_H
#ifdef __cplusplus
extern "C" {
#endif
#include <math.h>
__host__ __device__ inline double density_r(double r) {
   double density_r_result;
   density_r_result = 1390000.0*pow(r, -2.2999999999999998) + 300000000.0*1.0/(r*r*r*r*r*r) + 4800000000.0*1.0/(r*r*r*r*r*r*r*r*r*r*r*r*r*r) + 2.1050207736292895e+32*exp(-48.0*r);
   return density_r_result;
}
__host__ __device__ inline double omega_pe(double r) {
   double omega_pe_result;
   omega_pe_result = (2.605760673536811e+20*M_PI)*sqrt(6.6032602500329988e-27*pow(r, -2.2999999999999998) + 1.425164082740935e-24*1.0/(r*r*r*r*r*r) + 2.2802625323854961e-23*1.0/(r*r*r*r*r*r*r*r*r*r*r*r*r*r) + exp(-48.0*r));
   return omega_pe_result;
}
__host__ __device__ inline double domega_dr(double r) {
   const double x0 = exp(-48.0*r);
   double domega_dr_result;
   domega_dr_result = -2.605760673536811e+20*M_PI*pow(6.6032602500329988e-27*pow(r, -2.2999999999999998) + x0 + 1.425164082740935e-24*1.0/(r*r*r*r*r*r) + 2.2802625323854961e-23*1.0/(r*r*r*r*r*r*r*r*r*r*r*r*r*r), -1.0/2.0)*(7.5937492875379479e-27*pow(r, -3.2999999999999998) + 24.0*x0 + 4.2754922482228053e-24*1.0/(r*r*r*r*r*r*r) + 1.5961837726698473e-22*1.0/(r*r*r*r*r*r*r*r*r*r*r*r*r*r*r));
   return domega_dr_result;
}
#ifdef __cplusplus
}
#endif
#else
#endif
