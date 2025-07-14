#include <cstdio>
#include <chrono>
#include "Random.hpp"

using namespace Xoroshiro256StarStar;
using namespace RandomTransforms;
using std::chrono::high_resolution_clock;

constexpr int Nstates = 1024;
constexpr int Ndraws = 1024;
Xoro256State states[Nstates] = {};

BoxMullerResult<f64> draw_2_n(Xoro256State* s)
{
    uint64_t i0 = next(s);
    uint64_t i1 = next(s);

    f64 u0 = u64_to_unit_T<f64>(i0);
    f64 u1 = u64_to_unit_T<f64>(i1);

    return box_muller(u0, u1);
}

void write_results(const char* filename, f64* results,
                   int resultLength, int rowLength)
{
    FILE* f = fopen(filename, "w");
    int currentRowLength = 0;

    for (int i = 0; i < resultLength; ++i)
    {
        fprintf(f, "%.17f", results[i]);
        if (++currentRowLength == rowLength)
        {
            fprintf(f, "\n");
            currentRowLength = 0;
        }
        else
        {
            fprintf(f, " ");
        }
    }
    fflush(f);
    fclose(f);
}

int main(void)
{
    uint64_t seed = 13;
    states[0] = seed_state(seed);

    for (int i = 1; i < Nstates; ++i)
        states[i] = copy_jump(&states[i-1]);

    // NOTE(cmo): Box-Muller produces 2 values
    int numGen = 2 * Ndraws * Nstates;
    f64* results = (f64*)calloc(numGen, sizeof(f64));

    int resultIdx = 0;
    for (int d = 0; d < Ndraws; ++d)
    {
        for (int s = 0; s < Nstates; ++s)
        {
            auto draw = draw_2_n(&states[s]);
            results[resultIdx++] = draw.z0;
            results[resultIdx++] = draw.z1;
        }
    }

    auto start = high_resolution_clock::now();
    resultIdx = 0;
    for (int d = 0; d < Ndraws; ++d)
    {
        for (int s = 0; s < Nstates; ++s)
        {
            auto draw = draw_2_n(&states[s]);
            results[resultIdx++] = draw.z0;
            results[resultIdx++] = draw.z1;
        }
    }
    auto end = high_resolution_clock::now();
    std::chrono::duration<f64> diff = end - start;
    printf("Time taken: %.10e s\n", diff.count());

    write_results("NormalResults.txt", results, numGen, 2 * Nstates);

    free(results);

    return 0;
}