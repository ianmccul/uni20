cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0135 NEW)
project(uni20 VERSION 0.1.0 LANGUAGES CXX)

# Include extra CMake modules.
include(cmake/ClangFormat.cmake)

# Set a centralized directory for fetched content.
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cmake/third_party")

# Build Options (including testing/benchmarking)
option(UNI20_BUILD_PYTHON  "Build the Python bindings" ON)
option(UNI20_ENABLE_CUDA   "Enable CUDA backend support" OFF)
option(UNI20_ENABLE_MPI    "Enable MPI support" OFF)
option(UNI20_BUILD_TESTS   "Build unit tests" ON)
option(UNI20_BUILD_BENCH   "Build benchmarks" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# fmt library for formatting
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 9.1.0
)
FetchContent_MakeAvailable(fmt)

# mdspan reference implementation
FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG stable
)
FetchContent_MakeAvailable(mdspan)

# Optional: Enable CUDA if requested
if(UNI20_ENABLE_CUDA)
  message(STATUS "CUDA support enabled. Looking for nvcc...")
  enable_language(CUDA)
endif()

# Optionally fetch GoogleTest for unit testing
if(UNI20_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.12.1.zip
  )
  # For Windows: Force shared CRT to avoid issues
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# Optionally fetch Google Benchmark for performance testing
if(UNI20_BUILD_BENCH)
  FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/v1.6.1.tar.gz
  )
  FetchContent_MakeAvailable(benchmark)
endif()

# Add subdirectories
add_subdirectory(src)

if(UNI20_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()

if(UNI20_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(UNI20_BUILD_BENCH)
  add_subdirectory(benchmarks)
endif()

# add_subdirectory(examples)
