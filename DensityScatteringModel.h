#if !defined(DENSITY_MODEL_H)
#define DENSITY_MODEL_H
#ifdef __cplusplus
extern "C" {
#endif
#ifndef ISPC
#ifndef ISPC
#include <math.h>
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
inline double density_r(double r) {
   double density_r_result;
   density_r_result = 1390000.0*pow(r, -2.2999999999999998) + 300000000.0*1.0/(r*r*r*r*r*r) + 4800000000.0*1.0/(r*r*r*r*r*r*r*r*r*r*r*r*r*r) + 2.1050207736292895e+32*exp(-48.0*r);
   return density_r_result;
}
inline double omega_pe(double r) {
   double omega_pe_result;
   omega_pe_result = 8.1862385889964366e+20*sqrt(6.6032602500329988e-27*pow(r, -2.2999999999999998) + 1.425164082740935e-24*1.0/(r*r*r*r*r*r) + 2.2802625323854961e-23*1.0/(r*r*r*r*r*r*r*r*r*r*r*r*r*r) + exp(-48.0*r));
   return omega_pe_result;
}
inline double domega_dr(double r) {
   double domega_dr_result;
   domega_dr_result = 8.1862385889964366e+20*pow(6.6032602500329988e-27*pow(r, -2.2999999999999998) + 1.425164082740935e-24*1.0/(r*r*r*r*r*r) + 2.2802625323854961e-23*1.0/(r*r*r*r*r*r*r*r*r*r*r*r*r*r) + exp(-48.0*r), -1.0/2.0)*(-7.5937492875379479e-27*pow(r, -3.2999999999999998) - 4.2754922482228053e-24*1.0/(r*r*r*r*r*r*r) - 1.5961837726698473e-22*1.0/(r*r*r*r*r*r*r*r*r*r*r*r*r*r*r) - 24.0*exp(-48.0*r));
   return domega_dr_result;
}
#ifndef ISPC
#include <math.h>
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
inline double nu_scat(double r, double omega, double eps) {
   double nu_scat_result;
   nu_scat_result = 2.6303565064881985e+23*pow(r, -0.87999999999999989)*(eps*eps)*1.0/omega*pow(6.6032602500329988e-27*pow(r, -2.2999999999999998) + 1.425164082740935e-24*1.0/(r*r*r*r*r*r) + 2.2802625323854961e-23*1.0/(r*r*r*r*r*r*r*r*r*r*r*r*r*r) + exp(-48.0*r), 2.0)*pow(-6.6032602500329988e-27*pow(r, -2.2999999999999998) + 1.492214321728089e-42*(omega*omega) - 1.425164082740935e-24*1.0/(r*r*r*r*r*r) - 2.2802625323854961e-23*1.0/(r*r*r*r*r*r*r*r*r*r*r*r*r*r) - exp(-48.0*r), -1.5);
   return nu_scat_result;
}
#else
#ifndef ISPC
#include <math.h>
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846d
#endif
inline double density_r(double r) {
   double density_r_result;
   density_r_result = 1390000.0d*pow(r, -2.2999999999999998d) + 300000000.0d*1.0d/(r*r*r*r*r*r) + 4800000000.0d*1.0d/(r*r*r*r*r*r*r*r*r*r*r*r*r*r) + 2.1050207736292895e+32d*exp(-48.0d*r);
   return density_r_result;
}
inline double omega_pe(double r) {
   double omega_pe_result;
   omega_pe_result = 8.1862385889964366e+20d*sqrt(6.6032602500329988e-27d*pow(r, -2.2999999999999998d) + 1.425164082740935e-24d*1.0d/(r*r*r*r*r*r) + 2.2802625323854961e-23d*1.0d/(r*r*r*r*r*r*r*r*r*r*r*r*r*r) + exp(-48.0d*r));
   return omega_pe_result;
}
inline double domega_dr(double r) {
   double domega_dr_result;
   domega_dr_result = 8.1862385889964366e+20d*pow(6.6032602500329988e-27d*pow(r, -2.2999999999999998d) + 1.425164082740935e-24d*1.0d/(r*r*r*r*r*r) + 2.2802625323854961e-23d*1.0d/(r*r*r*r*r*r*r*r*r*r*r*r*r*r) + exp(-48.0d*r), -1.0d/2.0d)*(-7.5937492875379479e-27d*pow(r, -3.2999999999999998d) - 4.2754922482228053e-24d*1.0d/(r*r*r*r*r*r*r) - 1.5961837726698473e-22d*1.0d/(r*r*r*r*r*r*r*r*r*r*r*r*r*r*r) - 24.0d*exp(-48.0d*r));
   return domega_dr_result;
}
#ifndef ISPC
#include <math.h>
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846d
#endif
inline double nu_scat(double r, double omega, double eps) {
   double nu_scat_result;
   nu_scat_result = 2.6303565064881985e+23d*pow(r, -0.87999999999999989d)*(eps*eps)*1.0d/omega*pow(6.6032602500329988e-27d*pow(r, -2.2999999999999998d) + 1.425164082740935e-24d*1.0d/(r*r*r*r*r*r) + 2.2802625323854961e-23d*1.0d/(r*r*r*r*r*r*r*r*r*r*r*r*r*r) + exp(-48.0d*r), 2.0d)*pow(-6.6032602500329988e-27d*pow(r, -2.2999999999999998d) + 1.492214321728089e-42d*(omega*omega) - 1.425164082740935e-24d*1.0d/(r*r*r*r*r*r) - 2.2802625323854961e-23d*1.0d/(r*r*r*r*r*r*r*r*r*r*r*r*r*r) - exp(-48.0d*r), -1.5d);
   return nu_scat_result;
}
#endif
#ifdef __cplusplus
}
#endif
#else
#endif
