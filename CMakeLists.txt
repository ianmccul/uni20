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

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Sub-modules, for detailed tracing/logging
set(SUBMODULES BLAS2 BLAS3 LAPACK CUSOLVER)

# Create an option for each module.
foreach(module IN LISTS SUBMODULES)
  # Convert the module name to upper case.
  string(TOUPPER "${module}" MODULE_UPPER)
  # This creates options like ENABLE_TRACE_BLAS2, ENABLE_TRACE_BLAS3, etc.
  option(ENABLE_TRACE_${MODULE_UPPER} "Enable tracing for ${module}" OFF)
endforeach()

include(FetchContent)

# fmt library for formatting
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.1.3
)
FetchContent_MakeAvailable(fmt)

# mdspan reference implementation
FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG stable
)
FetchContent_MakeAvailable(mdspan)
# Use the square bracket operator for multidimensional arrays, not needed for c++23
# add_compile_definitions(MDSPAN_USE_BRACKET_OPERATOR=1)

# Enable CUDA if requested
if(UNI20_ENABLE_CUDA)
  message(STATUS "CUDA support enabled. Looking for nvcc...")
  enable_language(CUDA)
  set(UNI20_BACKEND_CUDA_DEFAULT ON)
else()
  set(UNI20_BACKEND_CUDA_DEFAULT OFF)
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

# Create an interface library for common, header-only dependencies.
add_library(uni20_deps INTERFACE)
target_link_libraries(uni20_deps INTERFACE fmt::fmt)

# Backends
option(UNI20_BACKEND_BLAS     "Enable the BLAS backend" ON)
option(UNI20_BACKEND_CUDA     "Enable the CUDA backend" ${UNI20_BACKEND_CUDA_DEFAULT})
option(UNI20_BACKEND_CUSOLVER "Enable the cuSOLVER backend" OFF)
option(UNI20_BACKEND_MKL      "Enable the MKL backend" OFF)

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

add_subdirectory(examples)
