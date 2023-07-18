#if !defined(CMO_SIMPLE_LUT_HPP)
#define CMO_SIMPLE_LUT_HPP
#include "Constants.h"
#include <functional>
#include <assert.h>

#ifdef __CUDACC__
#include "helper_cuda.h"
#endif

template <typename Tcalc, typename Tstore>
struct UniformLut
{
    Tstore x_min;
    Tstore x_max;
    int32_t Npoints;
    Tstore step;
    Tstore* fp;

    UniformLut() : x_min(Tstore(0.0)),
                   x_max(Tstore(0.0)),
                   step(Tstore(0.0)),
                   Npoints(0),
                   fp(nullptr)
    {}

    UniformLut(
        Tcalc x_min_,
        Tcalc x_max_,
        int32_t Nx,
        std::function<Tcalc(Tcalc)> fn
    ) : x_min(x_min_),
        x_max(x_max_),
        step(),
        Npoints(Nx),
        fp(nullptr)
    {
        init(x_min, x_max, Nx, fn);
    }

    ~UniformLut()
    {
        free_fp();
    }

    void init(Tcalc x_min_, Tcalc x_max_, int32_t Nx, std::function<Tcalc(Tcalc)> fn)
    {
        x_min = x_min_;
        x_max = x_max_;
        Npoints = Nx;
        step = (x_max - x_min) / Tstore(Npoints - 1);
        Tcalc step_precise = (x_max - x_min) / Tcalc(Npoints - 1);

        alloc_fp();

        for (int idx = 0; idx < Npoints; ++idx)
        {
            Tcalc x = x_min + step_precise * idx;
            fp[idx] = fn(x);
        }
    }

    void free_fp()
    {
        if (fp)
        {
#ifdef __CUDACC__
            cudaFree(fp);
#else
            free(fp);
#endif
        }
    }

    void alloc_fp()
    {
        free_fp();
#if __CUDACC__
        checkCudaErrors(cudaMallocManaged(&fp, Npoints * sizeof(Tstore)));
#else
        fp = calloc(Npoints, sizeof(Tstore)));
#endif
    }

    CudaFn Tstore operator()(Tstore x)
    {
        if (x > x_max || x < x_min)
        {
            printf("Accessed outside of range: [%f, %f]: <%f>\n", x_min, x_max, x);
            assert(false && "Out of bounds");
        }

        int32_t idx_m = (x - x_min) / step;
        int32_t idx_p = (x - x_min) / step;
        Tstore xm = x_min + idx_m * step;
        Tstore xp = xm + step;
        Tstore alpha = (x - xm) / (xp - xm);

        return (fpl(1.0) - alpha) * fp[idx_m] + alpha * fp[idx_p];
    }
};

#else
#endif