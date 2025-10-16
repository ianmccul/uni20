# src/common/CMakeLists.txt

include(CheckIncludeFile)
check_include_file_cxx("termios.h" HAVE_TERMIOS_H)
check_include_file_cxx("sys/ioctl.h" HAVE_SYS_IOCTL_H)

if(HAVE_TERMIOS_H)
  add_compile_definitions(HAVE_TERMIOS_H=1)
endif()

if(HAVE_SYS_IOCTL_H)
  add_compile_definitions(HAVE_SYS_IOCTL_H=1)
endif()


add_library(uni20_common STATIC
    terminal.cpp
)
target_link_libraries(uni20_common PUBLIC mdspan uni20_deps)

# If we are using GCC and havae stacktrace library available, then trace.hpp can make use of it.
# But we also need to link against libstdc++exp.
# First, detect the compiler and version. gcc-14 on ubuntu is configured with stacktrace lib
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Get GCC version
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14)
        # Test if stdc++exp is available
        include(CheckLibraryExists)
        check_library_exists(stdc++exp __cxa_throw "" HAS_LIBSTDCXXEXP)
        if(HAS_LIBSTDCXXEXP)
            target_link_libraries(uni20_common PUBLIC stdc++exp)
        endif()
    endif()
endif()
