#if !defined(CONSTANTS_HPP)
#define CONSTANTS_HPP

#include "YAKL.h"

#define _USE_MATH_DEFINES
#include <math.h>

typedef int32_t i32;
typedef int64_t i64;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;
typedef double f64;
#ifdef SCATTER_SINGLE_PRECISION
    typedef f32 fp_t;
    #define CMO_EXPAND(x) x
    #ifdef _MSC_VER
        #define FP(x) (CMO_EXPAND(x)##f)
    #else
        #define FP(x) (x##f)
    #endif
#else
    typedef double fp_t;
    #define FP(x) (x)
#endif

namespace Constants
{
    constexpr fp_t Rs = FP(6.96e10);
    constexpr fp_t c = FP(2.998e10);
    constexpr fp_t c_r = c / Rs;
    constexpr fp_t c_s = FP(2e7) / Rs;
    constexpr fp_t au = FP(215.0);
    constexpr fp_t Pi = M_PI;
    constexpr fp_t TwoPi = FP(2.0) * M_PI;
}

template <typename T>
constexpr KOKKOS_INLINE_FUNCTION T square(T x)
{
    return x * x;
}

template <typename T>
constexpr KOKKOS_INLINE_FUNCTION T cube(T x)
{
    return x * x * x;
}

template <typename T>
constexpr KOKKOS_INLINE_FUNCTION T min(T a, T b)
{
    return a <= b ? a : b;
}

template <typename T>
constexpr KOKKOS_INLINE_FUNCTION T max(T a, T b)
{
    return a >= b ? a : b;
}

#else
#endif