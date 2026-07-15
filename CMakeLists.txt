cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0135 NEW)
project(uni20 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
  message(FATAL_ERROR "Uni20 requires GCC 13 or newer for its C++23 implementation.")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
  message(FATAL_ERROR "Uni20 requires Clang 19 or newer, including complete C++20 alias-template CTAD support.")
endif()

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
include(DetectStacktrace)
include(Uni20TestHelpers)
include(Uni20Dependencies)
#include(cmake/DetectBlasIntSize.cmake)

# Configure where FetchContent stores third-party build trees.
# Default to a build-local directory to avoid cross-build cache contamination
# (for example Debug and Release builds sharing the same third-party build tree).
set(UNI20_FETCHCONTENT_BASE_DIR ""
    CACHE PATH
    "Override FetchContent base directory for build/stamp directories. Empty uses <binary_dir>/.cmake/third_party")
set(UNI20_FETCHCONTENT_SOURCE ON
    CACHE BOOL
    "Store FetchContent source trees in a shared source cache directory. OFF keeps sources under FETCHCONTENT_BASE_DIR")
set(UNI20_FETCHCONTENT_SOURCE_BASE_DIR ""
    CACHE PATH
    "When UNI20_FETCHCONTENT_SOURCE=ON, override shared FetchContent source directory. Empty uses <source_dir>/.cmake/third_party")
if(NOT UNI20_FETCHCONTENT_BASE_DIR)
  set(UNI20_FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/.cmake/third_party")
endif()

if(UNI20_FETCHCONTENT_SOURCE)
  if(NOT UNI20_FETCHCONTENT_SOURCE_BASE_DIR)
    set(UNI20_FETCHCONTENT_SOURCE_BASE_DIR "${CMAKE_SOURCE_DIR}/.cmake/third_party")
  endif()
else()
  set(UNI20_FETCHCONTENT_SOURCE_BASE_DIR "${UNI20_FETCHCONTENT_BASE_DIR}")
endif()

set(UNI20_FETCHCONTENT_BASE_DIR "${UNI20_FETCHCONTENT_BASE_DIR}"
    CACHE PATH
    "Resolved FetchContent base directory for build/stamp directories"
    FORCE)
set(UNI20_FETCHCONTENT_SOURCE "${UNI20_FETCHCONTENT_SOURCE}"
    CACHE BOOL
    "Store FetchContent source trees in a shared source cache directory. OFF keeps sources under UNI20_FETCHCONTENT_BASE_DIR"
    FORCE)
set(UNI20_FETCHCONTENT_SOURCE_BASE_DIR "${UNI20_FETCHCONTENT_SOURCE_BASE_DIR}"
    CACHE PATH
    "Resolved FetchContent source directory. With UNI20_FETCHCONTENT_SOURCE=OFF this matches UNI20_FETCHCONTENT_BASE_DIR"
    FORCE)

set(FETCHCONTENT_BASE_DIR "${UNI20_FETCHCONTENT_BASE_DIR}")

# Build Options (including testing/benchmarking)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(_uni20_debug_default ON)
else()
  set(_uni20_debug_default OFF)
endif()

uni20_dependency_option(UNI20_USE_SYSTEM_TBB "oneTBB library" AUTO)
uni20_dependency_option(UNI20_USE_SYSTEM_FMT "fmt library" AUTO)
uni20_dependency_option(UNI20_USE_SYSTEM_BENCHMARK "Google Benchmark library" OFF)
uni20_dependency_option(UNI20_USE_SYSTEM_GTEST "Google Test library" AUTO)
uni20_dependency_option(UNI20_USE_SYSTEM_NANOBIND "nanobind library" AUTO)
option(UNI20_BUILD_EXTERNAL_TESTS "Build tests for external dependencies" OFF)
option(UNI20_ILP64                "Enable ILP64 integer type for BLAS (64-bit integers)" OFF)
option(UNI20_BUILD_PYTHON         "Build the Python bindings" ON)
option(UNI20_BUILD_EXAMPLES       "Build the example programs" ON)
option(UNI20_ENABLE_CUDA          "Enable CUDA backend support" OFF)
option(UNI20_ENABLE_MPI           "Enable MPI support" OFF)
option(UNI20_ENABLE_MPLAPACK      "Enable optional MPLAPACK binary128 integration probes" OFF)
set(UNI20_HAS_FLOAT128 OFF)
set(UNI20_FLOAT128_PROVIDER_MPLAPACK OFF)
foreach(_uni20_removed_mplapack_cache_var IN ITEMS
    UNI20_USE_SYSTEM_MPLAPACK
    UNI20_MPLAPACK_GIT_REPOSITORY
    UNI20_MPLAPACK_GIT_TAG
    UNI20_MPLAPACK_SOURCE_DIR
    UNI20_MPLAPACK_VERSION)
  unset(${_uni20_removed_mplapack_cache_var} CACHE)
endforeach()
if(NOT UNI20_ENABLE_MPLAPACK)
  foreach(_uni20_disabled_mplapack_cache_var IN ITEMS
      UNI20_MPLAPACK_DIR
      UNI20_MPLAPACK_SOURCE
      UNI20_MPLAPACK_TARGET
      UNI20_DETECTED_MPLAPACK
      UNI20_MPLAPACK_BINARY128_HEADERS_COMPATIBLE)
    unset(${_uni20_disabled_mplapack_cache_var} CACHE)
  endforeach()
endif()
option(UNI20_BUILD_TESTS          "Build unit tests" ON)
option(UNI20_BUILD_COMBINED_TESTS "Build combined test executable (in addition to per-module tests)" ON)
option(UNI20_BUILD_BENCH          "Build benchmarks" ON)
option(UNI20_ENABLE_WARNINGS      "Enable compiler warnings" ON)
option(UNI20_ENABLE_LTO           "Enable LTO/IPO for optimized builds" ON)
option(UNI20_BUILD_ASM            "Build stand-alone assembly dumps for snippets in asm/" ON)
option(UNI20_DEBUG_DAG            "Enable DAG debug info (for Async buffer tracking and visualization)" ${_uni20_debug_default})
uni20_detect_stacktrace()
option(UNI20_ENABLE_STACKTRACE    "Enable C++23 <stacktrace> support when available" ${UNI20_STACKTRACE_AVAILABLE})
if(UNI20_ENABLE_STACKTRACE AND NOT UNI20_STACKTRACE_AVAILABLE)
  message(FATAL_ERROR
    "UNI20_ENABLE_STACKTRACE=ON, but the active compiler and C++ standard library cannot compile and link "
    "std::stacktrace. Configure with -DUNI20_ENABLE_STACKTRACE=OFF for degraded diagnostics or use a "
    "toolchain with stacktrace support.")
endif()
option(UNI20_SANITIZE             "Enable sanitizers (comma-separated list: address,undefined,thread,memory,leak)" "")
option(UNI20_EXTERNAL_NO_WERROR   "Disable -Werror for external libraries" ON)
option(UNI20_EXTERNAL_NO_WARN     "Disable all warnings for external libraries" OFF)
option(UNI20_ENABLE_COVERAGE      "Enable code coverage instrumentation" OFF)
option(UNI20_DEBUG_ASYNC_TASKS    "Enable AsyncTask debug instrumentation" ${_uni20_debug_default})
option(UNI20_DOCS_WEB             "Enable web-oriented Doxygen configuration for deployment" OFF)
set(UNI20_FALLBACK_TERMINAL_WIDTH 132 CACHE STRING
    "Terminal width used when COLUMNS and operating-system terminal detection are unavailable")
if(NOT UNI20_FALLBACK_TERMINAL_WIDTH MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "UNI20_FALLBACK_TERMINAL_WIDTH must be a positive integer")
endif()

if(UNI20_DEBUG_DAG AND NOT UNI20_DEBUG_ASYNC_TASKS)
  set(UNI20_DEBUG_ASYNC_TASKS ON CACHE BOOL "Enable AsyncTask debug instrumentation" FORCE)
endif()

if(UNI20_DOCS_WEB)
  message(STATUS "UNI20_DOCS_WEB=ON: configuring docs-only mode")
  set(UNI20_BUILD_TESTS OFF CACHE BOOL "Build unit tests" FORCE)
  set(UNI20_BUILD_COMBINED_TESTS OFF CACHE BOOL "Build combined test executable (in addition to per-module tests)" FORCE)
  set(UNI20_BUILD_EXAMPLES OFF CACHE BOOL "Build the example programs" FORCE)
  set(UNI20_BUILD_BENCH OFF CACHE BOOL "Build benchmarks" FORCE)
  set(UNI20_BUILD_PYTHON OFF CACHE BOOL "Build the Python bindings" FORCE)
  set(UNI20_BUILD_ASM OFF CACHE BOOL "Build stand-alone assembly dumps for snippets in asm/" FORCE)
  set(UNI20_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA backend support" FORCE)
  set(UNI20_ENABLE_MPLAPACK OFF CACHE BOOL "Enable optional MPLAPACK binary128 integration probes" FORCE)
endif()

# The nanobind extension links Uni20's static libraries into a shared module.
# Initialize every subsequently-created target with position-independent code so
# any project component can be exposed through Python without target-by-target
# linkage fixes.
if(UNI20_BUILD_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(UNI20_ENABLE_COVERAGE)
  message(STATUS "Enabling coverage instrumentation")

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
    add_custom_target(coverage
      COMMAND ${CMAKE_COMMAND} -E env
              GCOVR_SRCDIR=${CMAKE_SOURCE_DIR}
              GCOVR_BUILDDIR=${CMAKE_BINARY_DIR}
              gcovr
                --root ${CMAKE_SOURCE_DIR}
                --object-directory ${CMAKE_BINARY_DIR}
                --exclude '.*third_party.*'
                --exclude '.*tests/.*'
                --exclude '.*examples/.*'
                --exclude '.*benchmarks/.*'
                --gcov-ignore-parse-errors=negative_hits.warn_once_per_file
                --html --html-details -o coverage.html
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating coverage report"
    )
  else()
    message(WARNING "Coverage not supported for this compiler")
  endif()
endif()

if(NOT DEFINED UNI20_ASYNC_DEBUG)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(UNI20_ASYNC_DEBUG ON CACHE BOOL "Enable Async debug counters and tracing" FORCE)
    else()
        set(UNI20_ASYNC_DEBUG OFF CACHE BOOL "Enable Async debug counters and tracing" FORCE)
    endif()
endif()

if(NOT UNI20_BUILD_EXTERNAL_TESTS)
  set(FMT_TEST OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(TBB_TEST OFF CACHE BOOL "" FORCE)
  set(TBB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(BUILD_GTEST OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
endif()

if(UNI20_ENABLE_WARNINGS)
  set(UNI20_WARN_FLAGS "-Wall;-Wextra;-Wattributes;-pedantic;-pedantic-errors;-fstrict-aliasing;-Wstrict-aliasing;-Wno-unused-parameter;-Wno-array-bounds")

  set(UNI20_WARN_FLAGS_INTERNAL "${UNI20_WARN_FLAGS};-Werror")

  if(UNI20_EXTERNAL_NO_WARN)
    set(UNI20_WARN_FLAGS_EXTERNAL "")
  else()
    set(UNI20_WARN_FLAGS_EXTERNAL "${UNI20_WARN_FLAGS}")
  endif()
else()
  set(UNI20_WARN_FLAGS_INTERNAL "")
  set(UNI20_WARN_FLAGS_EXTERNAL "")
endif()

function(relax_warnings target)
  if(NOT TARGET ${target})
    return() # target not created yet
  endif()

  # If this is an ALIAS target, resolve to the real one
  get_target_property(alias ${target} ALIASED_TARGET)
  if(alias)
    set(real_target ${alias})
  else()
    set(real_target ${target})
  endif()

  get_target_property(_target_type ${real_target} TYPE)
  get_target_property(_target_imported ${real_target} IMPORTED)

  set(_external_warning_flags)
  if(NOT _target_imported AND NOT _target_type STREQUAL "INTERFACE_LIBRARY")
    if(UNI20_EXTERNAL_NO_WARN)
      if(MSVC)
        list(APPEND _external_warning_flags /w)
      elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND _external_warning_flags -w)
      endif()
    elseif(UNI20_WARN_FLAGS_EXTERNAL)
      list(APPEND _external_warning_flags ${UNI20_WARN_FLAGS_EXTERNAL})
    endif()

    if(UNI20_EXTERNAL_NO_WERROR)
      if(MSVC)
        list(APPEND _external_warning_flags /WX-)
      elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND _external_warning_flags -Wno-error)
      endif()
    endif()

    if(_external_warning_flags)
      message(STATUS "Relaxing warnings for external target: ${real_target}")
      target_compile_options(${real_target} PRIVATE ${_external_warning_flags})
    endif()
  endif()

  # mark includes as system
  get_target_property(_includes ${real_target} INTERFACE_INCLUDE_DIRECTORIES)
  if(_includes)
    target_include_directories(${real_target} SYSTEM INTERFACE ${_includes})
  endif()
endfunction()

add_compile_options(${UNI20_WARN_FLAGS_INTERNAL})

if(UNI20_SANITIZE)
  message(STATUS "Enabling sanitizers: ${UNI20_SANITIZE}")
  # Split comma-separated UNI20_SANITIZE into a proper CMake list
  string(REPLACE "," ";" UNI20_SANITIZE_LIST "${UNI20_SANITIZE}")

  foreach(san ${UNI20_SANITIZE_LIST})
    list(APPEND SAN_FLAGS "-fsanitize=${san}")
  endforeach()

  list(APPEND SAN_FLAGS -fno-omit-frame-pointer -g)

  add_compile_options(${SAN_FLAGS})
  add_link_options(${SAN_FLAGS})
endif()

# Sub-modules, for detailed tracing/logging
set(SUBMODULES BLAS LAPACK CUSOLVER ASYNC TESTMODULE)

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

  # This creates options like UNI20_ENABLE_TRACE_BLAS2, UNI20_ENABLE_TRACE_BLAS3, etc.
  option(UNI20_ENABLE_TRACE_${MODULE_UPPER} "Enable tracing for ${module}" ${default_value})
endforeach()

set(UNI20_BACKEND_CUDA_DEFAULT OFF)
if(NOT UNI20_DOCS_WEB AND UNI20_ENABLE_CUDA)
  message(STATUS "CUDA support enabled. Looking for nvcc...")
  enable_language(CUDA)
  set(UNI20_BACKEND_CUDA_DEFAULT ON)
endif()

function(enable_ipo_if_supported target)
  if(_uni20_enable_ipo_for_targets)
    if(_uni20_is_multi_config)
      set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
      set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
      set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
      set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_DEBUGOPT TRUE)
    else()
      set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
  endif()
endfunction()

# Backends
option(UNI20_BACKEND_BLAS     "Enable the BLAS backend" ON)
option(UNI20_BACKEND_CUDA     "Enable the CUDA backend" ${UNI20_BACKEND_CUDA_DEFAULT})
option(UNI20_BACKEND_CUSOLVER "Enable the cuSOLVER backend" OFF)
option(UNI20_BACKEND_MKL      "Enable the compatibility MKL backend extension API selector" OFF)
option(UNI20_BACKEND_MKL_SEQUENTIAL "Enable the sequential MKL backend extension API" OFF)
option(UNI20_BACKEND_MKL_THREADED "Enable the threaded MKL backend extension API" OFF)
option(UNI20_BACKEND_OPENBLAS "Enable the OpenBLAS backend extension API" OFF)
set(UNI20_BLAS_VENDOR "All" CACHE STRING "BLAS vendor hint passed to FindBLAS as BLA_VENDOR (for example: All, OpenBLAS, Generic, Intel10_64lp_seq, Intel10_64_dyn)")

if(UNI20_DOCS_WEB)
  set(UNI20_BACKEND_BLAS OFF CACHE BOOL "Enable the BLAS backend" FORCE)
  set(UNI20_BACKEND_CUDA OFF CACHE BOOL "Enable the CUDA backend" FORCE)
  set(UNI20_BACKEND_CUSOLVER OFF CACHE BOOL "Enable the cuSOLVER backend" FORCE)
  set(UNI20_BACKEND_MKL OFF CACHE BOOL "Enable the compatibility MKL backend extension API selector" FORCE)
  set(UNI20_BACKEND_MKL_SEQUENTIAL OFF CACHE BOOL "Enable the sequential MKL backend extension API" FORCE)
  set(UNI20_BACKEND_MKL_THREADED OFF CACHE BOOL "Enable the threaded MKL backend extension API" FORCE)
  set(UNI20_BACKEND_OPENBLAS OFF CACHE BOOL "Enable the OpenBLAS backend extension API" FORCE)
endif()

if(UNI20_BACKEND_MKL AND NOT UNI20_BACKEND_MKL_SEQUENTIAL AND NOT UNI20_BACKEND_MKL_THREADED)
  set(UNI20_BACKEND_MKL_SEQUENTIAL ON CACHE BOOL "Enable the sequential MKL backend extension API" FORCE)
endif()

if(UNI20_BACKEND_MKL_SEQUENTIAL AND UNI20_BACKEND_MKL_THREADED)
  message(FATAL_ERROR "Enable only one of UNI20_BACKEND_MKL_SEQUENTIAL or UNI20_BACKEND_MKL_THREADED.")
endif()

if(UNI20_BACKEND_MKL_SEQUENTIAL OR UNI20_BACKEND_MKL_THREADED)
  set(UNI20_BACKEND_MKL ON CACHE BOOL "Enable the compatibility MKL backend extension API selector" FORCE)
endif()

# Dependency: MKL requires BLAS
if(UNI20_BACKEND_MKL)
  if(DEFINED UNI20_BACKEND_BLAS AND NOT UNI20_BACKEND_BLAS)
    message(FATAL_ERROR "UNI20_BACKEND_MKL requires UNI20_BACKEND_BLAS to be enabled.")
  endif()
  set(UNI20_BACKEND_BLAS ON CACHE BOOL "Enable the BLAS backend (required by MKL)" FORCE)
endif()

# Dependency: OpenBLAS extensions require BLAS
if(UNI20_BACKEND_OPENBLAS)
  if(DEFINED UNI20_BACKEND_BLAS AND NOT UNI20_BACKEND_BLAS)
    message(FATAL_ERROR "UNI20_BACKEND_OPENBLAS requires UNI20_BACKEND_BLAS to be enabled.")
  endif()
  set(UNI20_BACKEND_BLAS ON CACHE BOOL "Enable the BLAS backend (required by OpenBLAS extensions)" FORCE)
endif()

# Dependency: cuSOLVER requires CUDA
if(UNI20_BACKEND_CUSOLVER)
  if(DEFINED UNI20_BACKEND_CUDA AND NOT UNI20_BACKEND_CUDA)
    message(FATAL_ERROR "UNI20_BACKEND_CUSOLVER requires UNI20_BACKEND_CUDA to be enabled.")
  endif()
  set(UNI20_BACKEND_CUDA ON CACHE BOOL "Enable the CUDA backend (required by cuSOLVER)" FORCE)
endif()

#if(ipo_supported)
#  message(STATUS "Enabling full Link-Time Optimization (LTO)")
#  # Globally turn on IPO for *all* targets and all configs
#  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
#else()
#  message(WARNING "IPO/LTO not supported: ${ipo_error}")
#endif()

if(NOT UNI20_DOCS_WEB)
  include(FetchContent)

  if(UNI20_ENABLE_MPLAPACK)
    find_package(mplapack QUIET CONFIG)
    if(NOT mplapack_FOUND OR NOT TARGET mplapack::mpblas_binary128 OR NOT TARGET mplapack::mplapack_binary128)
      message(FATAL_ERROR
        "UNI20_ENABLE_MPLAPACK=ON requires an externally configured MPLAPACK binary128 package.\n"
        "The package must provide targets mplapack::mpblas_binary128 and mplapack::mplapack_binary128.\n"
        "See docs/mplapack_binary128_setup.md for the full setup and validation workflow.\n"
        "\n"
        "Build MPLAPACK separately, for example:\n"
        "  cmake -S /path/to/mplapack -B /path/to/mplapack-build-binary128 \\\n"
        "    -DCMAKE_BUILD_TYPE=Release \\\n"
        "    -DMPLAPACK_CXX_STANDARD=23 \\\n"
        "    -DMPLAPACK_CXX_EXTENSIONS=ON \\\n"
        "    -DMPLAPACK_ENABLE_DOUBLE=OFF \\\n"
        "    -DMPLAPACK_ENABLE_MPFR=OFF \\\n"
        "    -DMPLAPACK_ENABLE_GMP=OFF \\\n"
        "    -DMPLAPACK_ENABLE_QD=OFF \\\n"
        "    -DMPLAPACK_ENABLE_DD=OFF \\\n"
        "    -DMPLAPACK_ENABLE_BINARY80=OFF \\\n"
        "    -DMPLAPACK_ENABLE_BINARY128=ON \\\n"
        "    -DMPLAPACK_BUILD_EXAMPLES=OFF \\\n"
        "    -DMPLAPACK_BUILD_TESTS=OFF \\\n"
        "    -DMPLAPACK_BUILD_BENCHMARKS=OFF\n"
        "  cmake --build /path/to/mplapack-build-binary128 --parallel\n"
        "  cmake --install /path/to/mplapack-build-binary128 --prefix /path/to/mplapack-install\n"
        "\n"
        "Then configure Uni20 with either:\n"
        "  cmake -S . -B /path/to/uni20-build -DUNI20_ENABLE_MPLAPACK=ON -Dmplapack_DIR=/path/to/mplapack-build-binary128\n"
        "or, after installing MPLAPACK:\n"
        "  cmake -S . -B /path/to/uni20-build -DUNI20_ENABLE_MPLAPACK=ON -DCMAKE_PREFIX_PATH=/path/to/mplapack-install")
    endif()

    include(CheckCXXSourceCompiles)
    set(_uni20_saved_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
    set(_uni20_saved_required_libraries "${CMAKE_REQUIRED_LIBRARIES}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_REQUIRED_LIBRARIES mplapack::mpblas_binary128 mplapack::mplapack_binary128)
    unset(UNI20_MPLAPACK_BINARY128_HEADERS_COMPATIBLE CACHE)
    check_cxx_source_compiles(
      [[
        #include <complex>
        #include <cmath>
        #include <cstdlib>
        #include <mpblas_binary128.h>
        #include <mplapack_binary128.h>

        #ifndef MPLAPACK_BINARY128_MODE
        #error "MPLAPACK_BINARY128_MODE is not defined"
        #endif

        #if MPLAPACK_BINARY128_MODE == MPLAPACK_BINARY128_MODE_LDBL
        #error "MPLAPACK binary128 aliases long double"
        #endif

        int main()
        {
          mplapack_binary128_t x = static_cast<mplapack_binary128_t>(1);
          auto y = abs(x);
          std::complex<mplapack_binary128_t> z{x, x};
          auto a = abs(z);
          auto b = sqrt(z);
          (void)y;
          (void)a;
          (void)b;
          return 0;
        }
      ]]
      UNI20_MPLAPACK_BINARY128_HEADERS_COMPATIBLE)
    set(CMAKE_REQUIRED_LIBRARIES "${_uni20_saved_required_libraries}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_uni20_saved_try_compile_target_type}")
    unset(_uni20_saved_required_libraries)
    unset(_uni20_saved_try_compile_target_type)
    if(NOT UNI20_MPLAPACK_BINARY128_HEADERS_COMPATIBLE)
      message(FATAL_ERROR
        "The found MPLAPACK package is not compatible with Uni20's C++23 binary128 build.\n"
        "Build MPLAPACK separately with GNU++23, apply the local CMake detector patch if needed,\n"
        "and point Uni20 at that build or install prefix. See docs/mplapack_binary128_setup.md.")
    endif()

    set(UNI20_MPLAPACK_SOURCE "package" CACHE STRING "Source type for mplapack" FORCE)
    set(UNI20_MPLAPACK_TARGET "mplapack::mplapack_binary128" CACHE STRING "CMake target name for mplapack" FORCE)
    if(DEFINED mplapack_DIR)
      set(UNI20_MPLAPACK_DIR "${mplapack_DIR}" CACHE PATH "Install or build directory for mplapack" FORCE)
      set(UNI20_DETECTED_MPLAPACK "Found via ${mplapack_DIR}" CACHE STRING "Found MPLAPACK package" FORCE)
    else()
      set(UNI20_DETECTED_MPLAPACK "Found package" CACHE STRING "Found MPLAPACK package" FORCE)
    endif()
    set(UNI20_HAS_FLOAT128 ON)
    set(UNI20_FLOAT128_PROVIDER_MPLAPACK ON)
    message(STATUS "MPLAPACK binary128 backend enabled: ${UNI20_DETECTED_MPLAPACK}")
  endif()

  uni20_add_dependency(
    NAME fmt
    VERSION 8.1.1
    TARGET fmt::fmt
    REPO https://github.com/fmtlib/fmt.git
    TAG 10.2.1
  )
  if(TARGET fmt)
    relax_warnings(fmt)
  endif()

  # mdspan reference implementation, put it into stdex namespace to prevent possible conflict with standard library version
  # and set up an interface library uni20_mdspan to set up the include directories
  set(_uni20_mdspan_fetchcontent_paths
    SOURCE_DIR "${UNI20_FETCHCONTENT_SOURCE_BASE_DIR}/mdspan-src"
    BINARY_DIR "${UNI20_FETCHCONTENT_BASE_DIR}/mdspan-build")

  FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG stable
    ${_uni20_mdspan_fetchcontent_paths}
  )
  FetchContent_MakeAvailable(mdspan)
  target_compile_definitions(mdspan INTERFACE MDSPAN_IMPL_STANDARD_NAMESPACE=stdex)
  relax_warnings(mdspan)

  # Use the square bracket operator for multidimensional arrays, not needed for c++23
  # add_compile_definitions(MDSPAN_USE_BRACKET_OPERATOR=1)

  # oneTBB
  uni20_add_dependency(
    NAME TBB
    VERSION 2021
    TARGET TBB::tbb
    REPO https://github.com/uxlfoundation/oneTBB
    TAG v2022.2.0
    COMPONENTS tbb
    SETTINGS
      "TBB_TEST=OFF"
      "TBB_BUILD_TESTS=OFF"
  )

  # If sanitizers are enabled, exclude them from TBB (known incompatibility).
  if(UNI20_SANITIZE)
    string(FIND "${UNI20_SANITIZE}" "address" _has_asan)
    string(FIND "${UNI20_SANITIZE}" "undefined" _has_ubsan)
    string(FIND "${UNI20_SANITIZE}" "thread" _has_tsan)
    if(NOT _has_asan EQUAL -1 OR NOT _has_ubsan EQUAL -1 OR NOT _has_tsan EQUAL -1)
      message(WARNING "Building uni20 with sanitizers (${UNI20_SANITIZE}), "
      "but disabling sanitizer instrumentation for oneTBB.")

      # Apply -fno-sanitize=all to all TBB sources
      get_target_property(tbb_sources TBB::tbb SOURCES)
      if(tbb_sources)
        set_source_files_properties(${tbb_sources} PROPERTIES COMPILE_FLAGS "-fno-sanitize=all")
      endif()
    endif()
  endif()

  # Optionally fetch GoogleTest for unit testing
  if(UNI20_BUILD_TESTS)
    set(_gtest_settings)
    if(MSVC)
      # GoogleTest defaults to the static MSVC runtime (/MT); opt into the shared
      # CRT so that gtest links cleanly with uni20's /MD configuration.
      set(_gtest_settings SETTINGS "gtest_force_shared_crt:BOOL=ON")
    endif()

    uni20_add_dependency(
      NAME GTest
      VERSION 1.12.0
      TARGET GTest::gtest
      REPO https://github.com/google/googletest.git
      TAG v1.17.0
      ${_gtest_settings}
    )
  endif()

  # Optionally fetch Google Benchmark for performance testing
  if(UNI20_BUILD_BENCH)
    uni20_add_dependency(
      NAME benchmark
      VERSION 1.5.2
      TARGET benchmark::benchmark
      REPO https://github.com/google/benchmark.git
      TAG v1.9.0
      SETTINGS
        "BENCHMARK_ENABLE_TESTING=OFF"
        "BENCHMARK_ENABLE_GTEST_TESTS=OFF"
        "BENCHMARK_ENABLE_INSTALL=OFF"
        "BENCHMARK_ENABLE_EXCEPTIONS=ON"
    )
  endif()

  # Enable link-time optimizations
  include(CheckIPOSupported)
  check_ipo_supported(
    RESULT ipo_supported
    OUTPUT ipo_error    # optional: captures the reason if it fails
    LANGUAGES CXX
  )

  set(_uni20_lto_allowed FALSE)
  get_property(_uni20_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

  if(_uni20_is_multi_config)
    set(_uni20_lto_allowed TRUE)
  else()
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _uni20_build_type_upper)
    if(_uni20_build_type_upper STREQUAL "RELEASE"
       OR _uni20_build_type_upper STREQUAL "RELWITHDEBINFO"
       OR _uni20_build_type_upper STREQUAL "MINSIZEREL"
       OR _uni20_build_type_upper STREQUAL "DEBUGOPT")
      set(_uni20_lto_allowed TRUE)
    endif()
  endif()

  set(_uni20_enable_ipo_for_targets FALSE)
  if(UNI20_ENABLE_LTO AND ipo_supported AND _uni20_lto_allowed)
    set(_uni20_enable_ipo_for_targets TRUE)
  endif()

  if(_uni20_enable_ipo_for_targets)
    if(_uni20_is_multi_config)
      message(STATUS "LTO/IPO enabled for configs: Release, RelWithDebInfo, MinSizeRel, DebugOpt")
    else()
      message(STATUS "LTO/IPO enabled for build type: ${CMAKE_BUILD_TYPE}")
    endif()
  elseif(NOT UNI20_ENABLE_LTO)
    message(STATUS "LTO/IPO disabled: UNI20_ENABLE_LTO=OFF")
  elseif(NOT ipo_supported)
    message(STATUS "LTO/IPO disabled: toolchain does not support IPO (${ipo_error})")
  else()
    if(CMAKE_BUILD_TYPE)
      message(STATUS
        "LTO/IPO disabled for build type '${CMAKE_BUILD_TYPE}' "
        "(enabled for Release, RelWithDebInfo, MinSizeRel, DebugOpt)")
    else()
      message(STATUS "LTO/IPO disabled: no build type selected")
    endif()
  endif()

  if(_uni20_enable_ipo_for_targets)
    if(_uni20_is_multi_config)
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUGOPT TRUE)
    else()
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
  endif()

  # Create an interface library for common dependencies
  add_library(uni20_deps INTERFACE)
  target_link_libraries(uni20_deps INTERFACE fmt::fmt)
  if(UNI20_ENABLE_MPLAPACK)
    target_link_libraries(uni20_deps INTERFACE mplapack::mpblas_binary128 mplapack::mplapack_binary128)
  endif()

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

  if(UNI20_BUILD_EXAMPLES)
    add_subdirectory(examples)
  endif()

  if(UNI20_BUILD_ASM)
    add_subdirectory(asm)
  endif()
endif()

# Doxygen documentation target
find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)
  # Where to look for our sources and markdown files
  set(DOXYGEN_INPUT
      "${CMAKE_SOURCE_DIR}/README.md"
      "${CMAKE_SOURCE_DIR}/src"
      "${CMAKE_SOURCE_DIR}/examples"
      "${CMAKE_SOURCE_DIR}/docs"
  )
  list(JOIN DOXYGEN_INPUT " " DOXYGEN_INPUT)

  set(DOXYGEN_EXCLUDE
      "${CMAKE_SOURCE_DIR}/docs/latex"
  )
  list(JOIN DOXYGEN_EXCLUDE " " DOXYGEN_EXCLUDE)

  set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs")
  if(UNI20_DOCS_WEB)
    set(DOXYGEN_HTML_EXTRA_STYLESHEET "${CMAKE_SOURCE_DIR}/docs/doxygen-custom.css")
  else()
    set(DOXYGEN_HTML_EXTRA_STYLESHEET "")
  endif()

  # Configure the Doxyfile from a template
  set(DOXYFILE_IN   "${CMAKE_SOURCE_DIR}/docs/Doxyfile.in")
  set(DOXYFILE_OUT  "${CMAKE_BINARY_DIR}/docs/Doxyfile")

  configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

  # Add a "doc" target
  add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API and Markdown documentation with Doxygen"
    VERBATIM
  )
else()
  message(WARNING "Doxygen not found; 'doc' target will be unavailable.")
endif()

# ------------------------------
# Uni20 option summary
# ------------------------------

# Helper to print all UNI20_* options (excluding advanced unless user asked)
# options starting UNI20_DETECTED_ are part of the environment, not variables
# that we expect users will override.
message(STATUS "")
message(STATUS "===== Uni20 Build Options =====")
get_cmake_property(_vars CACHE_VARIABLES)
foreach(v ${_vars})
  if(v MATCHES "^UNI20_" AND NOT v MATCHES "^UNI20_DETECTED_")
    get_property(_help CACHE ${v} PROPERTY HELPSTRING)
    get_property(_adv  CACHE ${v} PROPERTY ADVANCED)
    if(NOT _adv OR UNI20_SHOW_ADVANCED)
      message(STATUS "${v} = ${${v}}   # ${_help}")
    endif()
  endif()
endforeach()

message(STATUS "")

# Determine current build type (for single-config generators)
if(NOT CMAKE_BUILD_TYPE AND CMAKE_CONFIGURATION_TYPES)
  set(_current_config "Release") # default assumption for multi-config
else()
  set(_current_config "${CMAKE_BUILD_TYPE}")
endif()
string(TOUPPER "${_current_config}" _config_uc)

set(_well_known
  CMAKE_SYSTEM_VERSION
  CMAKE_SYSTEM_PROCESSOR
  CMAKE_BUILD_TYPE
  CMAKE_INSTALL_PREFIX
  BUILD_SHARED_LIBS
  CMAKE_CXX_FLAGS
  CMAKE_CXX_FLAGS_${_config_uc}
)

message(STATUS "===== Common CMake Options =====")
foreach(v IN LISTS _well_known)
  if(DEFINED ${v})
    if(DEFINED CACHE{${v}})
      get_property(_help CACHE ${v} PROPERTY HELPSTRING)
      message(STATUS "${v} = ${${v}}   # ${_help}")
    else()
      message(STATUS "${v} = ${${v}}")
    endif()
  else()
    message(STATUS "${v} = # (not set)")
  endif()
endforeach()
message(STATUS "")

message(STATUS "===== Detected Environment =====")
foreach(v ${_vars})
  if(v MATCHES "^UNI20_DETECTED_")
    get_property(_help CACHE ${v} PROPERTY HELPSTRING)
    message(STATUS "${v} = ${${v}}   # ${_help}")
  endif()
endforeach()

message(STATUS "")
