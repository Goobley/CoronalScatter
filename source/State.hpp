#if !defined(CMO_STATE_HPP)
#define CMO_STATE_HPP
#include "Constants.hpp"
#include "Random.hpp"
#include "SimpleLut.hpp"

typedef UniformLut<f64, fp_t> RadiusLut;

template <typename RandState>
struct BaseSimState
{
    int64_t Nparticles;
    fp_t time;
    yakl::Array<int32_t, 1, yakl::memDevice> active;

    yakl::Array<fp_t, 1, yakl::memDevice> r;
    yakl::Array<fp_t, 1, yakl::memDevice> rx;
    yakl::Array<fp_t, 1, yakl::memDevice> ry;
    yakl::Array<fp_t, 1, yakl::memDevice> rz;

    yakl::Array<fp_t, 1, yakl::memDevice> kc;
    yakl::Array<fp_t, 1, yakl::memDevice> kx;
    yakl::Array<fp_t, 1, yakl::memDevice> ky;
    yakl::Array<fp_t, 1, yakl::memDevice> kz;

    yakl::Array<fp_t, 1, yakl::memDevice> omega;
    yakl::Array<fp_t, 1, yakl::memDevice> nu_s;

    yakl::Array<RandState, 1, yakl::memDevice> rand_states;

    RadiusLut density_r;
    RadiusLut omega_pe;
    RadiusLut domega_dr;

    bool init(i64 Nparticles_) {
        Nparticles = Nparticles_;
        time = FP(0.0);
        active = decltype(active)("active", Nparticles);
        active = 1;

        r = decltype(r)("r", Nparticles);
        rx = decltype(rx)("rx", Nparticles);
        ry = decltype(ry)("ry", Nparticles);
        rz = decltype(rz)("rz", Nparticles);
        r =  FP(0.0);
        rx = FP(0.0);
        ry = FP(0.0);
        rz = FP(0.0);

        kc = decltype(kc)("kc", Nparticles);
        kx = decltype(kx)("kx", Nparticles);
        ky = decltype(ky)("ky", Nparticles);
        kz = decltype(kz)("kz", Nparticles);
        kc = FP(0.0);
        kx = FP(0.0);
        ky = FP(0.0);
        kz = FP(0.0);

        omega = decltype(omega)("omega", Nparticles);
        nu_s = decltype(nu_s)("nu_s", Nparticles);
        omega = FP(0.0);
        nu_s = FP(0.0);

        rand_states = decltype(rand_states)("rand_states", Nparticles);
        return true;
    }
};

#ifdef SCATTER_SINGLE_PRECISION
    using SimState = BaseSimState<Xoshiro128StarStar::Xo128State>;
    namespace Rand = Xoshiro128StarStar;
    typedef Rand::Xo128State RandState;
#else
    using SimState = BaseSimState<Xoroshiro256StarStar::Xoro256State>;
    namespace Rand = Xoroshiro256StarStar;
    typedef Rand::Xoro256State RandState;
#endif

using RandomTransforms::BoxMullerResult;
#else
#endif