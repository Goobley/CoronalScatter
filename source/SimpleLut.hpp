#if !defined(CMO_SIMPLE_LUT_HPP)
#define CMO_SIMPLE_LUT_HPP
#include "Constants.hpp"
#include <functional>
#include <assert.h>

template <typename Tcalc, typename Tstore>
struct UniformLut
{
    Tstore x_min;
    Tstore x_max;
    int32_t Npoints;
    Tstore step;
    yakl::Array<Tstore, 1, yakl::memDevice> data;

    void init(Tcalc x_min_, Tcalc x_max_, int32_t Nx, std::function<Tcalc(Tcalc)> fn)
    {
        x_min = x_min_;
        x_max = x_max_;
        Npoints = Nx;
        step = (x_max - x_min) / Tstore(Npoints - 1);
        Tcalc step_precise = (x_max - x_min) / Tcalc(Npoints - 1);

        yakl::Array<Tstore, 1, yakl::memHost> host_data("host LUT", Npoints);

        for (int idx = 0; idx < Npoints; ++idx)
        {
            Tcalc x = x_min + step_precise * idx;
            host_data(idx) = fn(x);
        }

        data = host_data.createDeviceCopy();
    }

    KOKKOS_INLINE_FUNCTION Tstore operator()(Tstore x) const
    {
#ifdef YAKL_DEBUG
        if (x > x_max || x < x_min)
        {
            printf("Accessed outside of range: [%f, %f]: <%f>\n", x_min, x_max, x);
            assert(false && "Out of bounds");
        }
#endif

        Tstore frac_a = (x - x_min) / step;
        int32_t idx_m = int32_t(std::floor(frac_a));
        int32_t idx_p = idx_m + 1;
        Tstore tp = frac_a - idx_m;
        Tstore tm = FP(1.0) - tp;
        if (frac_a < FP(0.0) || frac_a >= (Npoints - 1)) {
            idx_m = std::min(std::max(idx_m, 0), Npoints - 1);
            idx_p = idx_m;
            tm = FP(1.0);
            tp = FP(0.0);
        }

        return tm * data(idx_m) + tp * data(idx_p);
    }
};

#else
#endif