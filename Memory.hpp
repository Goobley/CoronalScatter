#if !defined(MEMORY_HPP)
#define MEMORY_HPP

#include <cstddef>

namespace CmoMemory {

template <typename T>
T* aligned_alloc(size_t n, size_t alignment);

template <typename T>
void deallocate(T* ptr);

#ifdef _WIN32
template <typename T>
T* aligned_alloc(size_t n, size_t alignment)
{
    return (T*)_aligned_malloc(n * sizeof(T), alignment);
}

template <typename T>
void deallocate(T* ptr)
{
    _aligned_free(ptr);
}
#else
template <typename T>
T* aligned_alloc(size_t n, size_t alignment)
{
    void* result;
    int error = posix_memalign(&result, alignment, n * sizeof(T));

    return error == 0 ? (T*)result : nullptr;
}

template <typename T>
void deallocate(T* ptr)
{
    free(ptr);
}

}

#endif
#endif