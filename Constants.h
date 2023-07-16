#if !defined(CONSTANTS_H)
#define CONSTANTS_H

#define _USE_MATH_DEFINES
#include <math.h>

typedef double f64;
#ifdef CMO_SINGLE_PRECISION
    typedef float fp_t;
    #define fpl(x) (x##f)
#else
    typedef double fp_t;
    #define fpl(x) (x)
#endif

#ifdef __CUDACC__
    #define CudaFn __host__ __device__
#else
    #define CudaFn
#endif

#ifdef __cplusplus
namespace Constants
{
    constexpr fp_t Rs = fpl(6.96e10);
    constexpr fp_t c = fpl(2.998e10);
    constexpr fp_t c_r = c / Rs;
    constexpr fp_t c_s = fpl(2e7) / Rs;
    constexpr fp_t au = fpl(215.0);
    constexpr fp_t Pi = M_PI;
    constexpr fp_t TwoPi = fpl(2.0) * M_PI;
}

template <typename T>
constexpr CudaFn T square(T x)
{
    return x * x;
}

template <typename T>
constexpr CudaFn T cube(T x)
{
    return x * x * x;
}

template <typename T>
constexpr CudaFn T min(T a, T b)
{
    return a <= b ? a : b;
}

template <typename T>
constexpr CudaFn T max(T a, T b)
{
    return a >= b ? a : b;
}

#else

inline fp_t square(fp_t x)
{
    return x * x;
}

inline fp_t cube(fp_t x)
{
    return x * x * x;
}

inline min(fp_t a, fp_t b)
{
    return a <= b ? a : b;
}

inline max(fp_t a, fp_t b)
{
    return a >= b ? a : b;
}

#endif
#else
#endif