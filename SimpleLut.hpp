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
#ifdef __CUDACC__
    cudaTextureObject_t tex;
#endif

    UniformLut() : x_min(Tstore(0.0)),
                   x_max(Tstore(0.0)),
                   step(Tstore(0.0)),
                   Npoints(0),
                   fp(nullptr),
                   tex(0)
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
        fp(nullptr),
        tex(0)
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

#ifdef __CUDACC__
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT_1g16ac75814780c3a16e4c63869feb9ad3
        // https://developer.nvidia.com/blog/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        // cudaArray* cuArray;
        // cudaCheckErrors(cudaMallocArray(&cuArray, &channelDesc, Npoints, 0));
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = fp;
        resDesc.res.linear.desc = channelDesc;
        resDesc.res.linear.sizeInBytes = Npoints * sizeof(fp_t);

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;

        checkCudaErrors(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
#endif
    }

    void free_fp()
    {
        if (fp)
        {
#ifdef __CUDACC__
            cudaDestroyTextureObject(tex);
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
        fp = (Tstore*)calloc(Npoints, sizeof(Tstore));
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
        int32_t idx_p = idx_m + 1;
        Tstore xm = x_min + idx_m * step;
        Tstore xp = xm + step;
        Tstore alpha = (x - xm) / (xp - xm);

#ifdef __CUDACC__
        Tstore fpm = tex1Dfetch<Tstore>(tex, idx_m);
        Tstore fpp = tex1Dfetch<Tstore>(tex, idx_p);
        return (fpl(1.0) - alpha) * fpm + alpha * fpp;
#else
        return (fpl(1.0) - alpha) * fp[idx_m] + alpha * fp[idx_p];
#endif
    }
};

#else
#endif