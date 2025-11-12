# src/CMakeLists.txt

# Build the definitions string for submodule tracing for the config.hpp file
set(TRACE_MODULE_DEFINITIONS "")
foreach(module IN LISTS SUBMODULES)
  string(TOUPPER "${module}" MODULE_UPPER)
  # The variable ENABLE_TRACE_MODULE_UPPER is defined by the option.
  if(${UNI20_ENABLE_TRACE_${MODULE_UPPER}})
    set(TRACE_MODULE_DEFINITIONS "${TRACE_MODULE_DEFINITIONS}#define ENABLE_TRACE_${MODULE_UPPER} 1\n")
  else()
    set(TRACE_MODULE_DEFINITIONS "${TRACE_MODULE_DEFINITIONS}#define ENABLE_TRACE_${MODULE_UPPER} 0\n")
  endif()
endforeach()

# Make sure the directory containing config.h is on the include path.
target_include_directories(uni20_deps INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(uni20_deps INTERFACE ${PROJECT_SOURCE_DIR}/src)

# Add the core library with its source files.
add_library(uni20 STATIC uni20.cpp)

# Specify include directories for the public interface.
#target_include_directories(uni20 PUBLIC
#    ${CMAKE_CURRENT_SOURCE_DIR}/core
#    ${CMAKE_CURRENT_SOURCE_DIR}/backends
#)

# Link external libraries (for example, fmt)
target_link_libraries(uni20 PUBLIC uni20_deps uni20_core uni20_common uni20_linalg)

add_subdirectory(core)
add_subdirectory(common)
add_subdirectory(backend)
add_subdirectory(level1)
add_subdirectory(mdspan)
add_subdirectory(async)
add_subdirectory(kernel)
add_subdirectory(linalg)
add_subdirectory(expokit)

target_link_libraries(uni20 PUBLIC uni20_expokit)

# Configure the header file. We need to do this last, because subdirectories may set their own variables.
configure_file(
    config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/config.hpp
    @ONLY
)
