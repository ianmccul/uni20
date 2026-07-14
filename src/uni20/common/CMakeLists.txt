# src/uni20/common/CMakeLists.txt

include(CheckIncludeFile)
include(CheckIncludeFileCXX)
check_include_file_cxx("termios.h" HAVE_TERMIOS_H)
check_include_file_cxx("sys/ioctl.h" HAVE_SYS_IOCTL_H)

if(HAVE_TERMIOS_H)
  add_compile_definitions(HAVE_TERMIOS_H=1)
endif()

if(HAVE_SYS_IOCTL_H)
  add_compile_definitions(HAVE_SYS_IOCTL_H=1)
endif()


add_library(uni20_common STATIC
    display.cpp
    presentation.cpp
    terminal.cpp
)
target_link_libraries(uni20_common PUBLIC uni20_mdspan uni20_deps)

if(UNI20_ENABLE_STACKTRACE AND UNI20_STACKTRACE_LIBRARIES)
  target_link_libraries(uni20_common PUBLIC ${UNI20_STACKTRACE_LIBRARIES})
endif()
