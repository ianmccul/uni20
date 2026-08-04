#[[
Detect whether the active C++ standard library provides `std::stacktrace`.

Output variables:
  - UNI20_STACKTRACE_AVAILABLE
  - UNI20_STACKTRACE_LIBRARIES

Output cache variables:
  - UNI20_DETECTED_STACKTRACE_PROVIDER
]]

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

function(uni20_detect_stacktrace)
  set(_stacktrace_probe [=[
    #include <stacktrace>

    #if !defined(__cpp_lib_stacktrace) || __cpp_lib_stacktrace < 202011L
    #error "std::stacktrace feature macro is unavailable"
    #endif

    int main()
    {
      return static_cast<int>(std::stacktrace::current().size());
    }
  ]=])

  cmake_push_check_state(RESET)
  check_cxx_source_compiles("${_stacktrace_probe}" _UNI20_STACKTRACE_WITHOUT_EXTRA_LIBRARY)
  cmake_pop_check_state()

  if(_UNI20_STACKTRACE_WITHOUT_EXTRA_LIBRARY)
    set(_available ON)
    set(_libraries "")
    set(_provider "standard library")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    find_library(
      _uni20_stdcxxexp_library
      NAMES stdc++exp
      PATHS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}
      NO_DEFAULT_PATH
      NO_CACHE
    )
    if(NOT _uni20_stdcxxexp_library)
      set(_uni20_stdcxxexp_library stdc++exp)
    endif()

    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES "${_uni20_stdcxxexp_library}")
    check_cxx_source_compiles("${_stacktrace_probe}" _UNI20_STACKTRACE_WITH_STDCXXEXP)
    cmake_pop_check_state()

    if(_UNI20_STACKTRACE_WITH_STDCXXEXP)
      set(_available ON)
      set(_libraries "${_uni20_stdcxxexp_library}")
      set(_provider "stdc++exp")
    endif()
  endif()

  if(NOT _available)
    set(_available OFF)
    set(_libraries "")
    set(_provider "unavailable")
  endif()

  set(UNI20_DETECTED_STACKTRACE_PROVIDER "${_provider}"
      CACHE INTERNAL "Provider used for std::stacktrace support" FORCE)
  set(UNI20_STACKTRACE_AVAILABLE "${_available}" PARENT_SCOPE)
  set(UNI20_STACKTRACE_LIBRARIES "${_libraries}" PARENT_SCOPE)

  message(STATUS "Detected std::stacktrace provider: ${_provider}")
endfunction()
