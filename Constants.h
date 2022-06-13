#if !defined(CONSTANTS_H)
#define CONSTANTS_H

#ifndef ISPC
#define _USE_MATH_DEFINES
#include <math.h>
#endif

typedef double fp_t;
typedef double f64;
// typedef float fp_t;

inline const fp_t Rs = 6.96e10;
inline const fp_t c_light = 2.998e10;
inline const fp_t c_r = c_light / Rs;
inline const fp_t c_s = 2e7 / Rs;
inline const fp_t au = 215.0;
inline const fp_t Pi = 3.14159265358979323846;
inline const fp_t TwoPi = 2.0 * Pi;

#ifdef __cplusplus
namespace ConstantsF64
{
    constexpr f64 Rs = 6.96e10;
    constexpr f64 c = 2.998e10;
    constexpr f64 c_r = c / Rs;
    constexpr f64 c_s = 2e7 / Rs;
    constexpr f64 au = 215.0;
    constexpr f64 Pi = 3.14159265358979323846;
    constexpr f64 TwoPi = 2.0 * Pi;
}
namespace ConstantsF32
{
    constexpr float Rs = 6.96e10f;
    constexpr float c = 2.998e10f;
    constexpr float c_r = c / Rs;
    constexpr float c_s = 2e7f / Rs;
    constexpr float au = 215.0f;
    constexpr float Pi = 3.14159265358979323846f;
    constexpr float TwoPi = 2.0 * Pi;
}

template <typename T>
constexpr T square(T x)
{
    return x * x;
}

template <typename T>
constexpr T cube(T x)
{
    return x * x * x;
}

template <typename T>
constexpr T min(T a, T b)
{
    return a <= b ? a : b;
}

template <typename T>
constexpr T max(T a, T b)
{
    return a >= b ? a : b;
}

namespace Constants = ConstantsF64;
// namespace Constants = ConstantsF32;
#else

inline fp_t square(fp_t x)
{
    return x * x;
}

inline fp_t cube(fp_t x)
{
    return x * x * x;
}

#endif
#else
#endif