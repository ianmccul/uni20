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
include(Uni20Dependencies)
#include(cmake/DetectBlasIntSize.cmake)

# Set a centralized directory for fetched content.
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cmake/third_party")

# Build Options (including testing/benchmarking)
option(UNI20_USE_SYSTEM_TBB       "Use system-installed oneTBB if available" ON)
option(UNI20_USE_SYSTEM_BENCHMARK "Use system-installed Google Benchmark if available" ON)
option(UNI20_BUILD_EXTERNAL_TESTS "Build tests for external dependencies" OFF)
option(UNI20_ILP64                "Enable ILP64 integer type for BLAS (64-bit integers)" OFF)
option(UNI20_BUILD_PYTHON         "Build the Python bindings" ON)
option(UNI20_ENABLE_CUDA          "Enable CUDA backend support" OFF)
option(UNI20_ENABLE_MPI           "Enable MPI support" OFF)
option(UNI20_BUILD_TESTS          "Build unit tests" ON)
option(UNI20_BUILD_COMBINED_TESTS "Build combined test executable (in addition to per-module tests)" ON)
option(UNI20_BUILD_BENCH          "Build benchmarks" ON)
option(UNI20_ENABLE_WARNINGS      "Enable compiler warnings" ON)
option(UNI20_BUILD_ASM            "Build stand-alone assembly dumps for snippets in asm/" ON)
option(UNI20_DEBUG_DAG            "Enable DAG debug info (for Async buffer tracking and visualization)" OFF)
option(UNI20_ENABLE_STACKTRACE    "Enable C++20 <stacktrace> header (requires library support)" OFF)
option(UNI20_SANITIZE             "Enable sanitizers (comma-separated list: address,undefined,thread,memory,leak)" "")
option(UNI20_EXTERNAL_NO_WERROR   "Disable -Werror for external libraries" ON)
option(UNI20_EXTERNAL_NO_WARN     "Disable all warnings for external libraries" OFF)
option(UNI20_ENABLE_COVERAGE      "Enable code coverage instrumentation" OFF)

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

# If user did not specify, set default based on build type
if(NOT DEFINED UNI20_DEBUG_DAG)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(UNI20_DEBUG_DAG ON CACHE BOOL "Enable DAG debug info (for Async buffer tracking and visualization)" FORCE)
    else()
        set(UNI20_DEBUG_DAG OFF CACHE BOOL "Enable DAG debug info (for Async buffer tracking and visualization)" FORCE)
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

  if(UNI20_EXTERNAL_NO_WERROR)
    message(STATUS "Relaxing warnings for external target: ${real_target}")
    set_target_properties(${real_target} PROPERTIES COMPILE_OPTIONS "${UNI20_WARN_FLAGS_EXTERNAL}")
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

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.2.1
)
FetchContent_MakeAvailable(fmt)
relax_warnings(fmt)

# mdspan reference implementation, put it into stdex namespace to prevent possible conflict with standard library version
# and set up an interface library uni20_mdspan to set up the include directories
FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG stable
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
  set(_gtest_settings)
  if(MSVC)
    # GoogleTest defaults to the static MSVC runtime (/MT); opt into the shared
    # CRT so that gtest links cleanly with uni20's /MD configuration.
    set(_gtest_settings SETTINGS "gtest_force_shared_crt:BOOL=ON")
  endif()

  uni20_add_dependency(
    NAME gtest
    VERSION 1.10.0
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

# Create an interface library for common dependencies
add_library(uni20_deps INTERFACE)
target_link_libraries(uni20_deps INTERFACE fmt::fmt)

# Backends
option(UNI20_BACKEND_BLAS     "Enable the BLAS backend" ON)
option(UNI20_BACKEND_CUDA     "Enable the CUDA backend" ${UNI20_BACKEND_CUDA_DEFAULT})
option(UNI20_BACKEND_CUSOLVER "Enable the cuSOLVER backend" OFF)
option(UNI20_BACKEND_MKL      "Enable the MKL backend" OFF)

# Dependency: MKL requires BLAS
if(UNI20_BACKEND_MKL)
  if(DEFINED UNI20_BACKEND_BLAS AND NOT UNI20_BACKEND_BLAS)
    message(FATAL_ERROR "UNI20_BACKEND_MKL requires UNI20_BACKEND_BLAS to be enabled.")
  endif()
  set(UNI20_BACKEND_BLAS ON CACHE BOOL "Enable the BLAS backend (required by MKL)" FORCE)
endif()

# Dependency: cuSOLVER requires CUDA
if(UNI20_BACKEND_CUSOLVER)
  if(DEFINED UNI20_BACKEND_CUDA AND NOT UNI20_BACKEND_CUDA)
    message(FATAL_ERROR "UNI20_BACKEND_CUSOLVER requires UNI20_BACKEND_CUDA to be enabled.")
  endif()
  set(UNI20_BACKEND_CUDA ON CACHE BOOL "Enable the CUDA backend (required by cuSOLVER)" FORCE)
endif()

# Enable link-time optimizations
include(CheckIPOSupported)
check_ipo_supported(
  RESULT ipo_supported
  OUTPUT ipo_error    # optional: captures the reason if it fails
  LANGUAGES CXX
)
if(ipo_supported)
  message(STATUS "Enabling full Link-Time Optimization (LTO)")
  # Globally turn on IPO for *all* targets and all configs
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
  message(WARNING "IPO/LTO not supported: ${ipo_error}")
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
  CMAKE_BUILD_TYPE
  CMAKE_INSTALL_PREFIX
  BUILD_SHARED_LIBS
  CMAKE_CXX_FLAGS
  CMAKE_CXX_FLAGS_${_config_uc}
)

message(STATUS "===== Common CMake Options =====")
foreach(v IN LISTS _well_known)
  if(DEFINED ${v})
    get_property(_help CACHE ${v} PROPERTY HELPSTRING)
    message(STATUS "${v} = ${${v}}   # ${_help}")
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
