#if !defined(CMO_STATE_HPP)
#define CMO_STATE_HPP
#include "Constants.h"
#include "Random.hpp"
#include "SimpleLut.hpp"

typedef UniformLut<f64, fp_t> RadiusLut;

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

    RadiusLut density_r;
    RadiusLut omega_pe;
    RadiusLut domega_dr;
};

using SimState = BaseSimState<Xoroshiro256StarStar::Xoro256State>;
namespace Rand = Xoroshiro256StarStar;
using RandomTransforms::BoxMullerResult;
typedef Rand::Xoro256State RandState;
#else
#endif