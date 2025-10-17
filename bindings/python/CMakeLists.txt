cmake_minimum_required(VERSION 3.18)

# Ensure Python headers/libraries are available before loading nanobind's CMake
# helpers.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

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
)

# Link against the core uni20 library (declared in src/CMakeLists.txt).
target_link_libraries(uni20_python PRIVATE uni20)

set_property(TARGET uni20_python PROPERTY CXX_STANDARD 23)
set_property(TARGET uni20_python PROPERTY CXX_STANDARD_REQUIRED ON)

# nanobind relies on zero-sized arrays for some of its internals, which trips the
# project's global -pedantic-errors setting. Relax the check locally for the
# extension module.
target_compile_options(uni20_python PRIVATE -Wno-pedantic -Wno-error=pedantic)

foreach(nb_target IN ITEMS nanobind nanobind-static nanobind-abi3)
  if(TARGET ${nb_target})
    target_compile_options(${nb_target} PRIVATE -Wno-pedantic -Wno-error=pedantic)
  endif()
endforeach()
