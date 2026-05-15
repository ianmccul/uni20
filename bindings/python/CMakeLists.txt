cmake_minimum_required(VERSION 3.18)

set( Python_FIND_UNVERSIONED_NAMES FIRST ) # FIXME: we probably want to require a clean python environment one day
set(Python_FIND_STRATEGY LOCATION)
# Ensure Python headers/libraries are available before loading nanobind's CMake
# helpers.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
set(UNI20_BINDINGS_PYTHON_EXECUTABLE "${Python_EXECUTABLE}" CACHE FILEPATH
    "Python interpreter used to build uni20 Python bindings" FORCE)

# Ensure nanobind is available via the repository-wide dependency helper, which
# first looks for an existing installation before falling back to FetchContent.
uni20_add_dependency(
  NAME nanobind
  VERSION 1.8.0
  TARGET nanobind::nanobind
  REPO https://github.com/wjakob/nanobind.git
  TAG v1.8.0
)

# Build the Python extension module using nanobind. NB_SHARED ensures we emit a
# loadable extension rather than a static archive.
nanobind_add_module(
  uni20_python
  NB_SHARED
  uni20_python.cpp
  core_bindings.cpp
  greet.cpp
)

# Link against the core uni20 library (declared in src/CMakeLists.txt).
target_link_libraries(uni20_python PRIVATE uni20)

# Set the name of the module to "uni20"
set_target_properties(uni20_python PROPERTIES OUTPUT_NAME "uni20")

set_property(TARGET uni20_python PROPERTY CXX_STANDARD 23)
set_property(TARGET uni20_python PROPERTY CXX_STANDARD_REQUIRED ON)

# nanobind relies on zero-sized arrays for some of its internals, which trips the
# project's global -pedantic-errors setting. Relax the check locally for the
# extension module and silence GCC 15's deprecated literal operator warning
# emitted from nanobind headers.
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-Wno-deprecated-literal-operator" NB_HAS_WNO_DEPRECATED_LITERAL_OPERATOR)
check_cxx_compiler_flag("-Wno-error=deprecated-literal-operator" NB_HAS_WNO_ERROR_DEPRECATED_LITERAL_OPERATOR)

set(NB_WARNING_FLAGS
    -Wno-pedantic
    -Wno-error=pedantic)
if(NB_HAS_WNO_DEPRECATED_LITERAL_OPERATOR)
  list(APPEND NB_WARNING_FLAGS -Wno-deprecated-literal-operator)
endif()
if(NB_HAS_WNO_ERROR_DEPRECATED_LITERAL_OPERATOR)
  list(APPEND NB_WARNING_FLAGS -Wno-error=deprecated-literal-operator)
endif()

target_compile_options(
  uni20_python PRIVATE
    ${NB_WARNING_FLAGS}
)

foreach(nb_target IN ITEMS nanobind nanobind-static nanobind-abi3)
  if(TARGET ${nb_target})
    relax_warnings(${nb_target})
    target_compile_options(
      ${nb_target} PRIVATE
        ${NB_WARNING_FLAGS}
    )
  endif()
endforeach()
