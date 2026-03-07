# -----------------------------------------------------------------------------
# \brief Unified dependency handler for uni20 external projects.
#
# Usage:
#   uni20_add_dependency(NAME <name>
#                        VERSION <version>
#                        TARGET <imported-target>
#                        REPO <git-url>
#                        TAG <git-tag>
#                        [COMPONENTS <comp1;comp2>]
#                        [SETTINGS <key1=value1;key2=value2>])
#
# Behavior:
#   1. UNI20_USE_SYSTEM_<NAME>=AUTO (default): try system package, then fetch.
#   2. UNI20_USE_SYSTEM_<NAME>=ON: require a system package (fail if missing).
#   3. UNI20_USE_SYSTEM_<NAME>=OFF: skip system package lookup and fetch.
#   4. Apply optional SETTINGS before FetchContent_MakeAvailable().
#   5. Exports two cache variables:
#        UNI20_<NAME>_SOURCE  ("system" | "fetched")
#        UNI20_<NAME>_TARGET  (<imported target name>)
# -----------------------------------------------------------------------------

function(_uni20_normalize_dependency_mode input_value output_var)
  string(TOUPPER "${input_value}" _mode)

  if(_mode STREQUAL "TRUE" OR _mode STREQUAL "YES" OR _mode STREQUAL "1")
    set(_mode "ON")
  elseif(_mode STREQUAL "FALSE" OR _mode STREQUAL "NO" OR _mode STREQUAL "0")
    set(_mode "OFF")
  endif()

  if(NOT (_mode STREQUAL "AUTO" OR _mode STREQUAL "ON" OR _mode STREQUAL "OFF"))
    message(FATAL_ERROR
      "Invalid dependency mode '${input_value}'. Expected one of: AUTO, ON, OFF.")
  endif()

  set(${output_var} "${_mode}" PARENT_SCOPE)
endfunction()

function(uni20_dependency_option option_var dependency_label default_value)
  if(NOT option_var)
    message(FATAL_ERROR "uni20_dependency_option() requires option_var")
  endif()

  if(NOT dependency_label)
    set(dependency_label "${option_var}")
  endif()

  if(NOT default_value)
    set(default_value "AUTO")
  endif()

  _uni20_normalize_dependency_mode("${default_value}" _default_mode)

  set(_help
      "How to resolve ${dependency_label}: AUTO (prefer system, fallback fetch), ON (require system), OFF (always fetch)")

  if(DEFINED ${option_var})
    set(_raw_mode "${${option_var}}")
  else()
    set(_raw_mode "${_default_mode}")
  endif()

  _uni20_normalize_dependency_mode("${_raw_mode}" _current_mode)
  set(${option_var} "${_current_mode}" CACHE STRING "${_help}" FORCE)

  set_property(CACHE ${option_var} PROPERTY STRINGS AUTO ON OFF)
endfunction()

function(uni20_add_dependency)
  cmake_parse_arguments(DEP "" "NAME;VERSION;TARGET;REPO;TAG" "COMPONENTS;SETTINGS" ${ARGN})

  if(NOT DEP_NAME)
    message(FATAL_ERROR "uni20_add_dependency() requires NAME parameter")
  endif()

  string(TOUPPER "${DEP_NAME}" NAME_UPPER)

  set(use_system_var "UNI20_USE_SYSTEM_${NAME_UPPER}")
  if(NOT DEFINED ${use_system_var})
    uni20_dependency_option(${use_system_var} "${DEP_NAME}" AUTO)
  else()
    _uni20_normalize_dependency_mode("${${use_system_var}}" _uni20_normalized_mode)
    set(${use_system_var} "${_uni20_normalized_mode}")
  endif()

  set(source_var   "UNI20_${NAME_UPPER}_SOURCE")
  set(target_var   "UNI20_${NAME_UPPER}_TARGET")
  set(version_var  "UNI20_${NAME_UPPER}_VERSION")
  set(dir_var      "UNI20_${NAME_UPPER}_DIR")
  set(detected_var "UNI20_DETECTED_${NAME_UPPER}")

  set(_uni20_use_system_mode "${${use_system_var}}")

  set(_uni20_found_system_package FALSE)
  set(_uni20_try_system_lookup FALSE)
  set(_uni20_require_system FALSE)

  if(_uni20_use_system_mode STREQUAL "AUTO")
    set(_uni20_try_system_lookup TRUE)
  elseif(_uni20_use_system_mode STREQUAL "ON")
    set(_uni20_try_system_lookup TRUE)
    set(_uni20_require_system TRUE)
  endif()

  if(_uni20_try_system_lookup)
    # When a minimum version is requested, use CONFIG mode so CMake can verify
    # the package version from a *ConfigVersion.cmake file. Module-mode finders
    # (for example FindGTest) may report FOUND without providing version metadata.
    # In that case we cannot verify compatibility and should fall back to fetching.
    if(DEP_COMPONENTS)
      find_package(${DEP_NAME} ${DEP_VERSION} QUIET CONFIG COMPONENTS ${DEP_COMPONENTS})
    else()
      find_package(${DEP_NAME} ${DEP_VERSION} QUIET CONFIG)
    endif()

    # If no explicit version was requested and config mode did not find a package,
    # allow a module-mode fallback.
    if((NOT DEP_VERSION) AND NOT (DEFINED ${DEP_NAME}_FOUND AND ${DEP_NAME}_FOUND))
      if(DEP_COMPONENTS)
        find_package(${DEP_NAME} QUIET MODULE COMPONENTS ${DEP_COMPONENTS})
      else()
        find_package(${DEP_NAME} QUIET MODULE)
      endif()
    endif()

    if(TARGET ${DEP_TARGET})
      set(_uni20_found_system_package TRUE)
    elseif(DEFINED ${DEP_NAME}_FOUND AND ${DEP_NAME}_FOUND)
      set(_uni20_found_system_package TRUE)
    endif()
  endif()

  if(_uni20_found_system_package)
    message(STATUS "Using system ${DEP_NAME}: ${${DEP_NAME}_DIR}")

    set(${source_var} "system" CACHE STRING "Source type for ${DEP_NAME} (system or fetched)" FORCE)
    set(${target_var} ${DEP_TARGET} CACHE STRING "CMake imported target name for ${DEP_NAME}" FORCE)
    if(DEFINED ${DEP_NAME}_VERSION)
      set(${version_var} ${${DEP_NAME}_VERSION} CACHE STRING "Detected ${DEP_NAME} version" FORCE)
    endif()
    if(DEFINED ${DEP_NAME}_DIR)
      set(${dir_var} ${${DEP_NAME}_DIR} CACHE PATH "Install directory for ${DEP_NAME}" FORCE)
    endif()
    if(DEFINED ${DEP_NAME}_DIR)
      set(_uni20_detected_help "Found via ${${DEP_NAME}_DIR}")
    else()
      set(_uni20_detected_help "Found system installation")
    endif()
    set(${detected_var} "system" CACHE STRING "${_uni20_detected_help}" FORCE)

  else()
    if(_uni20_require_system)
      message(FATAL_ERROR
        "${use_system_var}=ON requires a system installation of ${DEP_NAME}, "
        "but target '${DEP_TARGET}' was not found. "
        "Use ${use_system_var}=AUTO for fallback fetch, or OFF to force fetch.")
    endif()

    if(DEP_SETTINGS)
      foreach(setting IN LISTS DEP_SETTINGS)
        if(NOT setting MATCHES "^([^:=]+)(:([^=]+))?=(.*)$")
          message(FATAL_ERROR "Invalid SETTINGS entry '${setting}' for ${DEP_NAME}; expected VAR=VALUE or VAR:TYPE=VALUE")
        endif()

        set(var "${CMAKE_MATCH_1}")
        set(type "${CMAKE_MATCH_3}")
        set(value "${CMAKE_MATCH_4}")

        if(NOT type)
          string(TOUPPER "${value}" value_upper)
          if(value_upper STREQUAL "ON" OR value_upper STREQUAL "OFF"
             OR value_upper STREQUAL "TRUE" OR value_upper STREQUAL "FALSE")
            set(type BOOL)
          else()
            set(type STRING)
          endif()
        endif()

        set(${var} "${value}" CACHE ${type} "Auto-configured by uni20_add_dependency(${DEP_NAME})" FORCE)
      endforeach()
    endif()

    include(FetchContent)
    FetchContent_Declare(
      ${DEP_NAME}
      GIT_REPOSITORY ${DEP_REPO}
      GIT_TAG ${DEP_TAG}
    )
    FetchContent_MakeAvailable(${DEP_NAME})

    if(DEFINED DEP_REPO)
      set(repo_info "${DEP_REPO}")
      if(DEFINED DEP_TAG)
        string(APPEND repo_info " (tag ${DEP_TAG})")
      endif()
    else()
      set(repo_info "(no repository URL)")
    endif()

    set(help_text "Cloned from ${repo_info}")

    if(_uni20_use_system_mode STREQUAL "AUTO")
      message(STATUS "System ${DEP_NAME} not found or version is too old. Fetching from ${repo_info}")
    else()
      message(STATUS "Fetching ${DEP_NAME} from ${repo_info}")
    endif()

    set(${source_var} "fetched" CACHE STRING "Source type for ${DEP_NAME} (system or fetched)" FORCE)
    set(${target_var} ${DEP_TARGET} CACHE STRING "CMake imported target name for ${DEP_NAME}" FORCE)
    set(${version_var} ${DEP_VERSION} CACHE STRING "Requested version for ${DEP_NAME}" FORCE)
    set(${detected_var} "fetched" CACHE STRING "${help_text}" FORCE)
  endif()
endfunction()
