# src/uni20/core/CMakeLists.txt

add_library(uni20_core INTERFACE)

target_link_libraries(uni20_core INTERFACE uni20_deps)

function(uni20_escape_for_c output_var value)
  string(REPLACE "\\" "\\\\" _escaped "${value}")
  string(REPLACE "\"" "\\\"" _escaped "${_escaped}")
  string(REPLACE "\n" "\\n" _escaped "${_escaped}")
  string(REPLACE "\r" "\\r" _escaped "${_escaped}")
  string(REPLACE "\t" "\\t" _escaped "${_escaped}")
  set(${output_var} "${_escaped}" PARENT_SCOPE)
endfunction()

function(uni20_configure_buildinfo_header)
  if(CMAKE_BUILD_TYPE)
    set(UNI20_BUILDINFO_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
  elseif(CMAKE_CONFIGURATION_TYPES)
    string(REPLACE ";" ", " UNI20_BUILDINFO_CONFIGS "${CMAKE_CONFIGURATION_TYPES}")
    set(UNI20_BUILDINFO_BUILD_TYPE "Multi-config (${UNI20_BUILDINFO_CONFIGS})")
  else()
    set(UNI20_BUILDINFO_BUILD_TYPE "Not specified")
  endif()

  set(UNI20_BUILDINFO_SYSTEM_VERSION "${CMAKE_SYSTEM_VERSION}")
  if(NOT UNI20_BUILDINFO_SYSTEM_VERSION)
    set(UNI20_BUILDINFO_SYSTEM_VERSION "Unknown")
  endif()

  set(UNI20_BUILDINFO_SYSTEM_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
  if(NOT UNI20_BUILDINFO_SYSTEM_PROCESSOR)
    set(UNI20_BUILDINFO_SYSTEM_PROCESSOR "Unknown")
  endif()

  uni20_escape_for_c(UNI20_BUILDINFO_GENERATOR_ESCAPED "${CMAKE_GENERATOR}")
  uni20_escape_for_c(UNI20_BUILDINFO_BUILD_TYPE_ESCAPED "${UNI20_BUILDINFO_BUILD_TYPE}")
  uni20_escape_for_c(UNI20_BUILDINFO_SYSTEM_NAME_ESCAPED "${CMAKE_SYSTEM_NAME}")
  uni20_escape_for_c(UNI20_BUILDINFO_SYSTEM_VERSION_ESCAPED "${UNI20_BUILDINFO_SYSTEM_VERSION}")
  uni20_escape_for_c(UNI20_BUILDINFO_SYSTEM_PROCESSOR_ESCAPED "${UNI20_BUILDINFO_SYSTEM_PROCESSOR}")
  uni20_escape_for_c(UNI20_BUILDINFO_COMPILER_ID_ESCAPED "${CMAKE_CXX_COMPILER_ID}")
  uni20_escape_for_c(UNI20_BUILDINFO_COMPILER_VERSION_ESCAPED "${CMAKE_CXX_COMPILER_VERSION}")
  uni20_escape_for_c(UNI20_BUILDINFO_COMPILER_PATH_ESCAPED "${CMAKE_CXX_COMPILER}")

  set(UNI20_BUILDINFO_BUILD_OPTIONS_INITS "")
  set(UNI20_BUILDINFO_BUILD_OPTIONS_COUNT 0)
  set(UNI20_BUILDINFO_DETECTED_ENV_INITS "")
  set(UNI20_BUILDINFO_DETECTED_ENV_COUNT 0)

  get_cmake_property(_uni20_buildinfo_cache_vars CACHE_VARIABLES)
  foreach(v IN LISTS _uni20_buildinfo_cache_vars)
    if(v MATCHES "^UNI20_")
      get_property(_help CACHE ${v} PROPERTY HELPSTRING)
      if(NOT _help)
        set(_help "")
      endif()

      set(_value "${${v}}")
      string(REPLACE ";" ", " _value "${_value}")

      uni20_escape_for_c(_name_escaped "${v}")
      uni20_escape_for_c(_value_escaped "${_value}")
      uni20_escape_for_c(_help_escaped "${_help}")

      if(v MATCHES "^UNI20_DETECTED_")
        string(APPEND UNI20_BUILDINFO_DETECTED_ENV_INITS
               "    {\"${_name_escaped}\", \"${_value_escaped}\", \"${_help_escaped}\"},\n")
        math(EXPR UNI20_BUILDINFO_DETECTED_ENV_COUNT "${UNI20_BUILDINFO_DETECTED_ENV_COUNT} + 1")
      else()
        string(APPEND UNI20_BUILDINFO_BUILD_OPTIONS_INITS
               "    {\"${_name_escaped}\", \"${_value_escaped}\", \"${_help_escaped}\"},\n")
        math(EXPR UNI20_BUILDINFO_BUILD_OPTIONS_COUNT "${UNI20_BUILDINFO_BUILD_OPTIONS_COUNT} + 1")
      endif()
    endif()
  endforeach()

  file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/src/uni20")
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/buildinfo.hpp.in"
    "${PROJECT_BINARY_DIR}/src/uni20/buildinfo.hpp"
    @ONLY
  )
endfunction()

# Generate after all subdirectories have populated UNI20_* cache metadata.
cmake_language(DEFER DIRECTORY "${PROJECT_SOURCE_DIR}" CALL uni20_configure_buildinfo_header)
