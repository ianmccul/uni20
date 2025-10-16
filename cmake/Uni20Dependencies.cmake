include(FetchContent)

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
#   1. If UNI20_USE_SYSTEM_<NAME> is ON (default), try find_package().
#   2. If not found, fetch from the specified repo+tag using FetchContent.
#   3. Apply optional SETTINGS before FetchContent_MakeAvailable().
#   4. Exports two cache variables:
#        UNI20_<NAME>_SOURCE  ("system" | "fetched")
#        UNI20_<NAME>_TARGET  (<imported target name>)
# -----------------------------------------------------------------------------
function(uni20_add_dependency)
  cmake_parse_arguments(DEP "" "NAME;VERSION;TARGET;REPO;TAG" "COMPONENTS;SETTINGS" ${ARGN})

  if(NOT DEP_NAME)
    message(FATAL_ERROR "uni20_add_dependency() requires NAME parameter")
  endif()

  string(TOUPPER "${DEP_NAME}" NAME_UPPER)

  set(use_system_var "UNI20_USE_SYSTEM_${NAME_UPPER}")
  option(${use_system_var} "Use system-installed ${DEP_NAME} if available" ON)

  set(source_var   "UNI20_${NAME_UPPER}_SOURCE")
  set(target_var   "UNI20_${NAME_UPPER}_TARGET")
  set(version_var  "UNI20_${NAME_UPPER}_VERSION")
  set(dir_var      "UNI20_${NAME_UPPER}_DIR")
  set(detected_var "UNI20_DETECTED_${NAME_UPPER}")

  if(${use_system_var})
    if(DEP_COMPONENTS)
      find_package(${DEP_NAME} ${DEP_VERSION} QUIET COMPONENTS ${DEP_COMPONENTS})
    else()
      find_package(${DEP_NAME} ${DEP_VERSION} QUIET)
    endif()
  endif()

  if(${DEP_NAME}_FOUND)
    message(STATUS "Using system ${DEP_NAME}: ${${DEP_NAME}_DIR}")

    set(${source_var} "system" CACHE STRING "Source type for ${DEP_NAME} (system or fetched)" FORCE)
    set(${target_var} ${DEP_TARGET} CACHE STRING "CMake imported target name for ${DEP_NAME}" FORCE)
    if(DEFINED ${DEP_NAME}_VERSION)
      set(${version_var} ${${DEP_NAME}_VERSION} CACHE STRING "Detected ${DEP_NAME} version" FORCE)
    endif()
    if(DEFINED ${DEP_NAME}_DIR)
      set(${dir_var} ${${DEP_NAME}_DIR} CACHE PATH "Install directory for ${DEP_NAME}" FORCE)
    endif()
    set(${detected_var} "system" CACHE STRING "Found via ${${DEP_NAME}_DIR}" FORCE)

  else()
    message(STATUS "System ${DEP_NAME} not found — fetching via FetchContent")

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
      set(help_text "Cloned from ${repo_info}")
    else()
      set(help_text "Fetched via FetchContent (no repository URL)")
    endif()

    set(${source_var} "fetched" CACHE STRING "Source type for ${DEP_NAME} (system or fetched)" FORCE)
    set(${target_var} ${DEP_TARGET} CACHE STRING "CMake imported target name for ${DEP_NAME}" FORCE)
    set(${version_var} ${DEP_VERSION} CACHE STRING "Requested version for ${DEP_NAME}" FORCE)
    set(${detected_var} "fetched" CACHE STRING "${help_text}" FORCE)
  endif()
endfunction()
