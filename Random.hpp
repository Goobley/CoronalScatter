#ifndef CMO_RANDOM_HPP
#define CMO_RANDOM_HPP

#include <cmath>
#include <limits>
#include "Constants.h"

/*  Written in 2015 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

// NOTE(cmo): Modified to take state as an argument.

#include <stdint.h>

/* This is a fixed-increment version of Java 8's SplittableRandom generator
   See http://dx.doi.org/10.1145/2714064.2660195 and
   http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

   It is a very fast generator passing BigCrush, and it can be useful if
   for some reason you absolutely want 64 bits of state. */

uint64_t splitmix64(uint64_t* state) {
    uint64_t z = ((*state) += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

/* This is xoshiro256** 1.0, one of our all-purpose, rock-solid
   generators. It has excellent (sub-ns) speed, a state (256 bits) that is
   large enough for any parallel application, and it passes all tests we
   are aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

namespace Xoroshiro256StarStar
{

static CudaFn inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

struct Xoro256State
{
    uint64_t s[4];
};

Xoro256State seed_state(uint64_t x)
{
    Xoro256State result;
    for (int i = 0; i < 4; ++i)
        result.s[i] = splitmix64(&x);
    return result;
}

CudaFn uint64_t next(Xoro256State* state) {
    uint64_t* s = state->s;
    const uint64_t result = rotl(s[1] * 5, 7) * 9;

    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 45);

    return result;
}


/* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */

void jump(Xoro256State* state) {
    static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
    uint64_t* s = state->s;

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            next(state);
        }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}

Xoro256State copy_jump(Xoro256State* state)
{
    Xoro256State result(*state);
    jump(&result);
    return result;
}

/* This is the long-jump function for the generator. It is equivalent to
   2^192 calls to next(); it can be used to generate 2^64 starting points,
   from each of which jump() will generate 2^64 non-overlapping
   subsequences for parallel distributed computations. */

void long_jump(Xoro256State* state) {
    static const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };
    uint64_t* s = state->s;

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (LONG_JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            next(state);
        }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}
}

namespace Xoshiro128StarStar
{
/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */


/* This is xoshiro128** 1.1, one of our 32-bit all-purpose, rock-solid
   generators. It has excellent speed, a state size (128 bits) that is
   large enough for mild parallelism, and it passes all tests we are aware
   of.

   Note that version 1.0 had mistakenly s[0] instead of s[1] as state
   word passed to the scrambler.

   For generating just single-precision (i.e., 32-bit) floating-point
   numbers, xoshiro128+ is even faster.

   The state must be seeded so that it is not everywhere zero. */


static CudaFn inline uint32_t rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

struct Xo128State
{
    uint32_t s[4];
};

Xo128State seed_state(uint64_t x)
{
    Xo128State result;
    for (int i = 0; i < 4; ++i)
    {
        // NOTE(cmo): Take the lower 32-bits.
        result.s[i] = splitmix64(&x) & UINT32_MAX;
    }
    return result;
}

CudaFn uint32_t next(Xo128State* state) {
    uint32_t* s = state->s;
    const uint32_t result = rotl(s[1] * 5, 7) * 9;

    const uint32_t t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 11);

    return result;
}

/* This is the jump function for the generator. It is equivalent
   to 2^64 calls to next(); it can be used to generate 2^64
   non-overlapping subsequences for parallel computations. */

void jump(Xo128State* state) {
    static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };
    uint32_t* s = state->s;

    uint32_t s0 = 0;
    uint32_t s1 = 0;
    uint32_t s2 = 0;
    uint32_t s3 = 0;
    for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for(int b = 0; b < 32; b++) {
            if (JUMP[i] & UINT32_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            next(state);
        }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}

Xo128State copy_jump(Xo128State* state)
{
    Xo128State result(*state);
    jump(&result);
    return result;
}


/* This is the long-jump function for the generator. It is equivalent to
   2^96 calls to next(); it can be used to generate 2^32 starting points,
   from each of which jump() will generate 2^32 non-overlapping
   subsequences for parallel distributed computations. */

void long_jump(Xo128State* state) {
    static const uint32_t LONG_JUMP[] = { 0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662 };
    uint32_t* s = state->s;

    uint32_t s0 = 0;
    uint32_t s1 = 0;
    uint32_t s2 = 0;
    uint32_t s3 = 0;
    for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
        for(int b = 0; b < 32; b++) {
            if (LONG_JUMP[i] & UINT32_C(1) << b) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            next(state);
        }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}
}

namespace RandomTransforms
{
// NOTE(cmo): Based on the numba implementation
// https://github.com/numba/numba/blob/main/numba/cuda/random.py
// http://marc-b-reynolds.github.io/math/2020/06/16/UniformFloat.html
template <typename T>
CudaFn T u64_to_unit_T(uint64_t x)
{
    // NOTE(cmo): I believe the 1.0 should be left as double, not fpl.
    // return T((x >> 11) * (1.0 / (UINT64_C(1) << 53)));
    return T((x >> 11) * 0x1p-53);
}

template <typename T>
CudaFn T u32_to_unit_T(uint32_t x)
{
    // NOTE(cmo): Case for when state is 32-bit (assumed to be generating f32s).
    // return T((x >> 8) * (1.0f / (UINT32_C(1) << 24)));
    return T((x >> 8) * 0x1p-24f);
}

template <typename T>
struct BoxMullerResult
{
    T z0;
    T z1;
};

template <typename T, typename RandState>
CudaFn BoxMullerResult<T> box_muller(T (*next)(RandState*), RandState* state)
{
    // NOTE(cmo): This function may need to draw more than once to ensure u0 >=
    // epsilon (or infs appear from log. Box-Muller is technically defined for
    // u0 and u1 in (0, 1), and we generate [0, 1)). In practice the bias is not
    // meaningful if we occasionally tale a u1=0 sample.
    // Given the _much_ smaller state space, we just happened to run into this
    // case faster in pure f32 generation.
    constexpr T epsilon = std::numeric_limits<T>::epsilon();
    constexpr T TwoPi = fpl(2.0) * fpl(M_PI);
    T u0 = fpl(0.0);
    while (u0 <= epsilon)
    {
        u0 = next(state);
    }
    T u1 = next(state);

    T prefactor = sqrt(T(fpl(-2.0)) * log(u0));
    T c = cos(TwoPi * u1);
    T s = sin(TwoPi * u1);

    BoxMullerResult<T> result;
    result.z0 = prefactor * c;
    result.z1 = prefactor * s;
    return result;
}

}

#else
#endif