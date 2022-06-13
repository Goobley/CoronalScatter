#ifndef CMO_SIM_STATE_H
#define CMO_SIM_STATE_H

#include "Constants.h"
#include "Random.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ISPC
#define uniform
#endif

struct SimState
{
    uniform int64_t Nparticles;
    uniform fp_t time;
    uniform int32_t* active;

    uniform fp_t* r;
    uniform fp_t* rx;
    uniform fp_t* ry;
    uniform fp_t* rz;

    uniform fp_t* kc;
    uniform fp_t* kx;
    uniform fp_t* ky;
    uniform fp_t* kz;

    uniform fp_t* omega;
    uniform fp_t* nu_s;

    uniform RandState* randState;
};

struct SimParams
{
    uniform fp_t eps;
    uniform fp_t Rinit;
    uniform fp_t Rstop;
    uniform fp_t aniso;
    uniform fp_t fRatio;
    uniform fp_t asym;
    uniform fp_t theta0;
    uniform fp_t omega0;
    uniform fp_t nu_s0;
    uniform fp_t dt0;
    uniform fp_t dtSave;
    uniform uint64_t seed;
};

#ifdef __cplusplus
}
#endif
#else
#endif