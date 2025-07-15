cmake_minimum_required(VERSION 3.18)
if ("${SCATTER_ARCH}" STREQUAL "CUDA")
    project(CoronalScatter LANGUAGES CXX CUDA)
else ()
    project(CoronalScatter LANGUAGES CXX)
endif()

add_subdirectory(fmt)
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

include("CheckGit.cmake")
CheckGitSetup()

SET(CMAKE_CXX_FLAGS "${CXX_FLAGS}")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Kokkos REQUIRED GLOBAL)

add_library(yakl INTERFACE)
target_include_directories(yakl INTERFACE YAKL/src YAKL/external YAKL/src/extensions)
set(YAKL_DEFS "")
if (${YAKL_AUTO_PROFILE})
    set(YAKL_DEFS "${YAKL_DEFS} -DYAKL_AUTO_PROFILE")
endif()
if (${YAKL_PROFILE})
    set(YAKL_DEFS "${YAKL_DEFS} -DYAKL_PROFILE")
endif()
if (${YAKL_AUTO_FENCE})
    set(YAKL_DEFS "${YAKL_DEFS} -DYAKL_AUTO_FENCE")
endif()

list(APPEND SCATTER_SOURCE_FILES
    "source/scatter.cpp"
)

add_executable(scatter
    "${SCATTER_SOURCE_FILES}"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if ("${SCATTER_ARCH}" STREQUAL "CUDA")
    set(CMAKE_CXX_STANDARD 20)
endif()

if ("${SCATTER_SINGLE_PRECISION}")
    target_compile_definitions(scatter PUBLIC SCATTER_SINGLE_PRECISION)
    set(SCATTER_CXX_FLAGS "${SCATTER_CXX_FLAGS} -DSCATTER_SINGLE_PRECISION")
endif()

# NOTE(cmo): Override language type for cuda, may need to do this for HIP too.
set_source_files_properties(${SCATTER_SOURCE_FILES} PROPERTIES CXX_STANDARD 20)
set_source_files_properties(${SCATTER_SOURCE_FILES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
if ("${SCATTER_ARCH}" STREQUAL "CUDA")
    set(SCATTER_EXTRA_CUDA_STD_FLAG "")
    if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
        set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
        set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
        set(SCATTER_EXTRA_CUDA_STD_FLAG "-std=c++20")
    endif()
    set_target_properties(scatter PROPERTIES CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})

    set_source_files_properties(${SCATTER_SOURCE_FILES} PROPERTIES CUDA_STANDARD 20)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} ${SCATTER_EXTRA_CUDA_STD_FLAG}")
endif()

############## scatter ##############
target_include_directories(scatter SYSTEM PUBLIC
    "${GCC_INCLUDE_PATH}"
    "${MPI_INCLUDE_PATH}"
    "${NETCDF_INCLUDE_PATH}"
    "${CUDA_MATH_INCLUDE_PATH}"
)
target_link_libraries(scatter "${LDLIBS}")
target_link_libraries(scatter fmt::fmt)
target_link_libraries(scatter argparse)
target_link_libraries(scatter git_version)
target_link_libraries(scatter Kokkos::kokkos)
target_link_libraries(scatter yakl)

set_target_properties(scatter PROPERTIES
    COMPILE_FLAGS "${YAKL_DEFS} ${SCATTER_CXX_FLAGS}"
    LINK_FLAGS "${LDFLAGS}"
)

message("Compiling with: ${CMAKE_CXX_FLAGS} ${YAKL_DEFS} ${SCATTER_CXX_FLAGS}")