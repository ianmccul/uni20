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

# Compute human-readable build metadata for the Python bindings.
if(CMAKE_BUILD_TYPE)
  set(UNI20_PY_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
elseif(CMAKE_CONFIGURATION_TYPES)
  string(REPLACE ";" ", " UNI20_PY_CONFIGS "${CMAKE_CONFIGURATION_TYPES}")
  set(UNI20_PY_BUILD_TYPE "Multi-config (${UNI20_PY_CONFIGS})")
else()
  set(UNI20_PY_BUILD_TYPE "Not specified")
endif()

set(UNI20_PY_SYSTEM_VERSION "${CMAKE_SYSTEM_VERSION}")
if(NOT UNI20_PY_SYSTEM_VERSION)
  set(UNI20_PY_SYSTEM_VERSION "Unknown")
endif()

set(UNI20_PY_SYSTEM_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(NOT UNI20_PY_SYSTEM_PROCESSOR)
  set(UNI20_PY_SYSTEM_PROCESSOR "Unknown")
endif()

function(uni20_escape_for_c output_var value)
  string(REPLACE "\\" "\\\\" _escaped "${value}")
  string(REPLACE "\"" "\\\"" _escaped "${_escaped}")
  set(${output_var} "${_escaped}" PARENT_SCOPE)
endfunction()

uni20_escape_for_c(UNI20_PY_GENERATOR_ESCAPED "${CMAKE_GENERATOR}")
uni20_escape_for_c(UNI20_PY_BUILD_TYPE_ESCAPED "${UNI20_PY_BUILD_TYPE}")
uni20_escape_for_c(UNI20_PY_SYSTEM_NAME_ESCAPED "${CMAKE_SYSTEM_NAME}")
uni20_escape_for_c(UNI20_PY_SYSTEM_VERSION_ESCAPED "${UNI20_PY_SYSTEM_VERSION}")
uni20_escape_for_c(UNI20_PY_SYSTEM_PROCESSOR_ESCAPED "${UNI20_PY_SYSTEM_PROCESSOR}")
uni20_escape_for_c(UNI20_PY_COMPILER_ID_ESCAPED "${CMAKE_CXX_COMPILER_ID}")
uni20_escape_for_c(UNI20_PY_COMPILER_VERSION_ESCAPED "${CMAKE_CXX_COMPILER_VERSION}")
uni20_escape_for_c(UNI20_PY_COMPILER_PATH_ESCAPED "${CMAKE_CXX_COMPILER}")

set(UNI20_PY_BUILD_OPTIONS_INITS "")
set(UNI20_PY_BUILD_OPTIONS_COUNT 0)
set(UNI20_PY_DETECTED_ENV_INITS "")
set(UNI20_PY_DETECTED_ENV_COUNT 0)

get_cmake_property(_uni20_py_cache_vars CACHE_VARIABLES)
foreach(v ${_uni20_py_cache_vars})
  if(v MATCHES "^UNI20_" AND NOT v MATCHES "^UNI20_DETECTED_")
    get_property(_help CACHE ${v} PROPERTY HELPSTRING)
    if(NOT _help)
      set(_help "")
    endif()

    set(_value "${${v}}")
    string(REPLACE ";" ", " _value "${_value}")

    uni20_escape_for_c(_name_escaped "${v}")
    uni20_escape_for_c(_value_escaped "${_value}")
    uni20_escape_for_c(_help_escaped "${_help}")

    string(APPEND UNI20_PY_BUILD_OPTIONS_INITS
           "    {\"${_name_escaped}\", \"${_value_escaped}\", \"${_help_escaped}\"},\n")
    math(EXPR UNI20_PY_BUILD_OPTIONS_COUNT "${UNI20_PY_BUILD_OPTIONS_COUNT} + 1")
  elseif(v MATCHES "^UNI20_DETECTED_")
    get_property(_help CACHE ${v} PROPERTY HELPSTRING)
    if(NOT _help)
      set(_help "")
    endif()

    set(_value "${${v}}")
    string(REPLACE ";" ", " _value "${_value}")

    uni20_escape_for_c(_name_escaped "${v}")
    uni20_escape_for_c(_value_escaped "${_value}")
    uni20_escape_for_c(_help_escaped "${_help}")

    string(APPEND UNI20_PY_DETECTED_ENV_INITS
           "    {\"${_name_escaped}\", \"${_value_escaped}\", \"${_help_escaped}\"},\n")
    math(EXPR UNI20_PY_DETECTED_ENV_COUNT "${UNI20_PY_DETECTED_ENV_COUNT} + 1")
  endif()
endforeach()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/buildinfo.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/buildinfo.hpp
  @ONLY
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

target_include_directories(uni20_python PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Set the name of the module to "uni20"
set_target_properties(uni20_python PROPERTIES OUTPUT_NAME "uni20")

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
