# src/uni20/CMakeLists.txt

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

# Make sure generated config.hpp and source headers are on the include path.
target_include_directories(uni20_deps INTERFACE ${PROJECT_BINARY_DIR}/src)
target_include_directories(uni20_deps INTERFACE ${PROJECT_SOURCE_DIR}/src)

# Add the core library with its source files.
add_library(uni20 STATIC uni20.cpp)

# Link external libraries (for example, fmt)
target_link_libraries(uni20 PUBLIC uni20_deps uni20_core uni20_common uni20_linalg uni20_symmetry)
enable_ipo_if_supported(uni20)

add_subdirectory(core)
add_subdirectory(mdspan)
add_subdirectory(common)
target_link_libraries(uni20_mdspan INTERFACE uni20_common uni20_core)
add_subdirectory(symmetry)
add_subdirectory(backend)
add_subdirectory(async)
add_subdirectory(kernel)
add_subdirectory(linalg)
add_subdirectory(krylov)
add_subdirectory(tensor_network)
add_subdirectory(models)

target_link_libraries(uni20 PUBLIC uni20_krylov uni20_models uni20_tensor_network)

# Configure the header file. We need to do this last, because subdirectories may set their own variables.
if(NOT UNI20_BACKEND_BLAS OR
   NOT DEFINED UNI20_DETECTED_BLAS_VENDOR OR
   UNI20_DETECTED_BLAS_VENDOR STREQUAL "")
  set(UNI20_DETECTED_BLAS_VENDOR "None")
endif()
if(NOT UNI20_BACKEND_BLAS OR NOT DEFINED UNI20_DETECTED_BLAS_VENDOR_MACRO OR
   UNI20_DETECTED_BLAS_VENDOR_MACRO STREQUAL "")
  set(UNI20_DETECTED_BLAS_VENDOR_MACRO "UNI20_BLAS_VENDOR_NONE")
endif()
if(NOT UNI20_BACKEND_BLAS OR NOT DEFINED BLA_SIZEOF_INTEGER OR BLA_SIZEOF_INTEGER STREQUAL "")
  if(UNI20_ILP64)
    set(BLA_SIZEOF_INTEGER 8)
  else()
    set(BLA_SIZEOF_INTEGER 4)
  endif()
endif()

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/src/uni20")
configure_file(
  config.hpp.in
  ${PROJECT_BINARY_DIR}/src/uni20/config.hpp
  @ONLY
)
