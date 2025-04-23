cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0135 NEW)
project(uni20 VERSION 0.1.0 LANGUAGES CXX)

# Build-type support
# For single-config generators (Makefile, Ninja), default to Release if none given.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release"
      CACHE STRING
      "Choose the type of build (Debug, DebugOpt, Release, RelWithDebInfo, MinSizeRel)"
      FORCE)
endif()

# Define valid configs for multi-config generators (VS, Xcode).
set(CMAKE_CONFIGURATION_TYPES
    Debug
    DebugOpt
    Release
    RelWithDebInfo
    MinSizeRel
    CACHE STRING "Configs" FORCE)

# Tweak per-config flags
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUGOPT       "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE        "-DNDEBUG -O2")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-DNDEBUG -O2 -g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-DNDEBUG -Os")

# silence CMP0115 “source file extensions must be explicit”
if(POLICY CMP0115)
  cmake_policy(SET CMP0115 NEW)
endif()

# Include extra CMake modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ClangFormat)
include(Uni20TestHelpers)
#include(cmake/DetectBlasIntSize.cmake)

# Set a centralized directory for fetched content.
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cmake/third_party")

# Build Options (including testing/benchmarking)
option(UNI20_ILP64                "Enable ILP64 integer type for BLAS (64-bit integers)" OFF)
option(UNI20_BUILD_PYTHON         "Build the Python bindings" ON)
option(UNI20_ENABLE_CUDA          "Enable CUDA backend support" OFF)
option(UNI20_ENABLE_MPI           "Enable MPI support" OFF)
option(UNI20_BUILD_TESTS          "Build unit tests" ON)
option(UNI20_BUILD_COMBINED_TESTS "Build combined test executable (in addition to per-module tests)" ON)
option(UNI20_BUILD_BENCH          "Build benchmarks" ON)
option(UNI20_ENABLE_WARNINGS      "Enable compiler warnings" ON)
option(UNI20_BUILD_ASM            "Build stand-alone assembly dumps for snippets in asm/" ON)

if(UNI20_ENABLE_WARNINGS)
  set(UNI20_WARN_FLAGS "-Wall;-Wextra;-Wattributes;-pedantic;-pedantic-errors;-fstrict-aliasing;-Wstrict-aliasing;-Wno-unused-parameter")
else()
  set(UNI20_WARN_FLAGS "")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(${UNI20_WARN_FLAGS})

# Sub-modules, for detailed tracing/logging
set(SUBMODULES BLAS2 BLAS3 LAPACK CUSOLVER TESTMODULE)

# Create an option for each module.
foreach(module IN LISTS SUBMODULES)
  # Convert the module name to upper case.
  string(TOUPPER "${module}" MODULE_UPPER)

   # Set the default value based on the module name.
   # the TEST module is for debugging the TRACE_MODULE macro, and defaults to ON
  if(module STREQUAL "TESTMODULE")
    set(default_value ON)
  else()
    set(default_value OFF)
  endif()

  # This creates options like ENABLE_TRACE_BLAS2, ENABLE_TRACE_BLAS3, etc.
  option(ENABLE_TRACE_${MODULE_UPPER} "Enable tracing for ${module}" ${default_value})
endforeach()

include(FetchContent)

# fmt library for formatting
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 11.1.3
)
FetchContent_MakeAvailable(fmt)

# mdspan reference implementation, put it into stdex namespace to prevent possible conflict with standard library version
# and set up an interface library uni20_mdspan to set up the include directories
FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG stable
)
FetchContent_MakeAvailable(mdspan)
add_library(uni20_mdspan INTERFACE)
target_include_directories(uni20_mdspan INTERFACE ${mdspan_SOURCE_DIR}/include)
target_compile_definitions(uni20_mdspan INTERFACE MDSPAN_IMPL_STANDARD_NAMESPACE=stdex)

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

# Configure the config.hpp header.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/config.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/config.hpp"
)

# Add subdirectories
add_subdirectory(src)

if(UNI20_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()

if(UNI20_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(UNI20_BUILD_BENCH)
  add_subdirectory(benchmarks)
endif()

add_subdirectory(examples)

if(UNI20_BUILD_ASM)
  add_subdirectory(asm)
endif()

# Doxygen documentation target
find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)
  # Where to look for our sources and markdown files
  set(DOXYGEN_INPUT
      "${CMAKE_SOURCE_DIR}/src"
      "${CMAKE_SOURCE_DIR}/examples"
      "${CMAKE_SOURCE_DIR}/docs"
  )
  list(JOIN DOXYGEN_INPUT " " DOXYGEN_INPUT)

  # Configure the Doxyfile from a template
  set(DOXYFILE_IN   "${CMAKE_SOURCE_DIR}/docs/Doxyfile.in")
  set(DOXYFILE_OUT  "${CMAKE_BINARY_DIR}/docs/Doxyfile")

  configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

  # Add a "doc" target
  add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/docs
    COMMENT "Generating API and Markdown documentation with Doxygen"
    VERBATIM
  )
else()
  message(WARNING "Doxygen not found; 'doc' target will be unavailable.")
endif()
