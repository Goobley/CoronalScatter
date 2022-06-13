#ifndef CMO_RANDOM_HPP
#define CMO_RANDOM_HPP

#ifdef __cplusplus
extern "C" {
#endif

#include "Constants.h"
#ifndef ISPC
#include <math.h>
#include <stdint.h>
#else
typedef int32 int32_t;
typedef int64 int64_t;
typedef uint64 uint64_t;
#endif

/*  Written in 2015 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

// NOTE(cmo): Modified to take state as an argument.


/* This is a fixed-increment version of Java 8's SplittableRandom generator
   See http://dx.doi.org/10.1145/2714064.2660195 and
   http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

   It is a very fast generator passing BigCrush, and it can be useful if
   for some reason you absolutely want 64 bits of state. */

inline uint64_t splitmix64(uint64_t* state) {
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

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// namespace Xoroshiro256StarStar
// {

// struct Xoro256State
// {
//     uint64_t s[4];
// };

// Xoro256State seed_state(uint64_t x)
// {
//     Xoro256State result;
//     for (int i = 0; i < 4; ++i)
//         result.s[i] = splitmix64(&x);
//     return result;
// }

// uint64_t next(Xoro256State* state) {
//     uint64_t* s = state->s;
//     const uint64_t result = rotl(s[1] * 5, 7) * 9;

//     const uint64_t t = s[1] << 17;

//     s[2] ^= s[0];
//     s[3] ^= s[1];
//     s[1] ^= s[2];
//     s[0] ^= s[3];

//     s[2] ^= t;

//     s[3] = rotl(s[3], 45);

//     return result;
// }


// /* This is the jump function for the generator. It is equivalent
//    to 2^128 calls to next(); it can be used to generate 2^128
//    non-overlapping subsequences for parallel computations. */

// void jump(Xoro256State* state) {
//     static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
//     uint64_t* s = state->s;

//     uint64_t s0 = 0;
//     uint64_t s1 = 0;
//     uint64_t s2 = 0;
//     uint64_t s3 = 0;
//     for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
//         for(int b = 0; b < 64; b++) {
//             if (JUMP[i] & UINT64_C(1) << b) {
//                 s0 ^= s[0];
//                 s1 ^= s[1];
//                 s2 ^= s[2];
//                 s3 ^= s[3];
//             }
//             next(state);
//         }

//     s[0] = s0;
//     s[1] = s1;
//     s[2] = s2;
//     s[3] = s3;
// }

// Xoro256State copy_jump(Xoro256State* state)
// {
//     Xoro256State result(*state);
//     jump(&result);
//     return result;
// }

// /* This is the long-jump function for the generator. It is equivalent to
//    2^192 calls to next(); it can be used to generate 2^64 starting points,
//    from each of which jump() will generate 2^64 non-overlapping
//    subsequences for parallel distributed computations. */

// void long_jump(Xoro256State* state) {
//     static const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };
//     uint64_t* s = state->s;

//     uint64_t s0 = 0;
//     uint64_t s1 = 0;
//     uint64_t s2 = 0;
//     uint64_t s3 = 0;
//     for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
//         for(int b = 0; b < 64; b++) {
//             if (LONG_JUMP[i] & UINT64_C(1) << b) {
//                 s0 ^= s[0];
//                 s1 ^= s[1];
//                 s2 ^= s[2];
//                 s3 ^= s[3];
//             }
//             next(state);
//         }

//     s[0] = s0;
//     s[1] = s1;
//     s[2] = s2;
//     s[3] = s3;
// }
// }


struct Xoro256StateSoa
{
    uint64_t* s0;
    uint64_t* s1;
    uint64_t* s2;
    uint64_t* s3;
};

#ifndef ISPC
inline void seed_state(Xoro256StateSoa* s, int32_t idx, uint64_t x)
{
    s->s0[idx] = splitmix64(&x);
    s->s1[idx] = splitmix64(&x);
    s->s2[idx] = splitmix64(&x);
    s->s3[idx] = splitmix64(&x);
}
#endif

inline uint64_t next(Xoro256StateSoa* state, int32_t idx)
{
    uint64_t s0 = state->s0[idx];
    uint64_t s1 = state->s1[idx];
    uint64_t s2 = state->s2[idx];
    uint64_t s3 = state->s3[idx];

    const uint64_t result = rotl(s1 * 5, 7) * 9;

    const uint64_t t = s1 << 17;

    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;

    s2 ^= t;

    s3 = rotl(s3, 45);

    state->s0[idx] = s0;
    state->s1[idx] = s1;
    state->s2[idx] = s2;
    state->s3[idx] = s3;

    return result;
}


#ifndef ISPC
/* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */

inline void jump(Xoro256StateSoa* state, int32_t idx) {
    static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (JUMP[i] & UINT64_C(1) << b) {
                s0 ^= state->s0[idx];
                s1 ^= state->s1[idx];
                s2 ^= state->s2[idx];
                s3 ^= state->s3[idx];
            }
            next(state, idx);
        }

    state->s0[idx] = s0;
    state->s1[idx] = s1;
    state->s2[idx] = s2;
    state->s3[idx] = s3;
}

inline void copy_jump(Xoro256StateSoa* state, int32_t copyto, int32_t copyfrom)
{
    state->s0[copyto] = state->s0[copyfrom];
    state->s1[copyto] = state->s1[copyfrom];
    state->s2[copyto] = state->s2[copyfrom];
    state->s3[copyto] = state->s3[copyfrom];
    jump(state, copyto);
}
#endif

inline fp_t u64_to_unit_T(uint64_t x)
{
#ifdef ISPC
    return (fp_t)((x >> 11) * (1.0d / (1ULL << 53)));
#else
    return (fp_t)((x >> 11) * (1.0 / (1ULL << 53)));
#endif
}

struct BoxMullerResult
{
    fp_t z0;
    fp_t z1;
};

inline BoxMullerResult box_muller(fp_t u0, fp_t u1)
{
#ifdef ISPC
    fp_t prefactor = sqrt(-2.0d * log(u0));
#else
    fp_t prefactor = sqrt(-2.0 * log(u0));
#endif
    fp_t c = cos(TwoPi * u1);
    fp_t s = sin(TwoPi * u1);

    BoxMullerResult result;
    result.z0 = prefactor * c;
    result.z1 = prefactor * s;
    return result;
}

typedef Xoro256StateSoa RandState;
inline fp_t draw_u(RandState* s, int32_t idx)
{
    uint64_t i = next(s, idx);
    return u64_to_unit_T(i);
}

inline BoxMullerResult draw_2_n(RandState* s, int32_t idx)
{
    uint64_t i0 = next(s, idx);
    uint64_t i1 = next(s, idx);

    fp_t u0 = u64_to_unit_T(i0);
    fp_t u1 = u64_to_unit_T(i1);

    return box_muller(u0, u1);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "Memory.hpp"

Xoro256StateSoa* alloc_n_states(int32_t numStates)
{
    Xoro256StateSoa* state = (Xoro256StateSoa*)calloc(1, sizeof(Xoro256StateSoa));
    if (!state)
        return nullptr;

    state->s0 = CmoMemory::aligned_alloc<uint64_t>(numStates, 32);
    state->s1 = CmoMemory::aligned_alloc<uint64_t>(numStates, 32);
    state->s2 = CmoMemory::aligned_alloc<uint64_t>(numStates, 32);
    state->s3 = CmoMemory::aligned_alloc<uint64_t>(numStates, 32);

    return state;
}

void free_states(Xoro256StateSoa* states)
{
    if (!states)
        return;

    CmoMemory::deallocate(states->s0);
    CmoMemory::deallocate(states->s1);
    CmoMemory::deallocate(states->s2);
    CmoMemory::deallocate(states->s3);
    free(states);
}

#endif

#else
#endif
