

include(CheckIncludeFile)
check_include_file("termios.h" HAVE_TERMIOS_H)
check_include_file("sys/ioctl.h" HAVE_SYS_IOCTL_H)

if(HAVE_TERMIOS_H)
  add_compile_definitions(HAVE_TERMIOS_H=1)
endif()

if(HAVE_SYS_IOCTL_H)
  add_compile_definitions(HAVE_SYS_IOCTL_H=1)
endif()


add_library(uni20_common STATIC
    common.hpp
    terminal.cpp
)
target_link_libraries(uni20_common PUBLIC uni20_deps)
target_include_directories(uni20_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
