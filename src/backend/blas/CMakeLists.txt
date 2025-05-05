
if(UNI20_ILP64)
  set(BLA_SIZEOF_INTEGER 8 CACHE INTERNAL "BLAS integer size (8 bytes for ILP64)")
else()
  set(BLA_SIZEOF_INTEGER 4 CACHE INTERNAL "BLAS integer size (4 bytes for LP64)")
endif()

message(STATUS "UNI20_ILP64 is set to: ${UNI20_ILP64}")
message(STATUS "Using BLA_SIZEOF_INTEGER = ${BLA_SIZEOF_INTEGER}")

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(NOT BLAS_FOUND)
  message(FATAL_ERROR "BLAS not found.")
endif()

if(NOT LAPACK_FOUND)
  message(FATAL_ERROR "LAPACK not found.")
endif()

# set variables for the BLAS configuration
set(BLAS_FOUND "${BLAS_FOUND}" CACHE INTERNAL "BLAS was found")
include(DetectBlasVendor)
detect_blas_vendor()

add_library(uni20_interface_blas INTERFACE)

# Add the BLAS libraries to the interface link line
target_link_libraries(uni20_interface_blas INTERFACE ${BLAS_LIBRARIES})

# If FindBLAS sets BLAS_LINKER_FLAGS, attach them to link options:
if(BLAS_LINKER_FLAGS)
  target_link_options(uni20_interface_blas INTERFACE ${BLAS_LINKER_FLAGS})
endif()

add_library(uni20_backend_blas STATIC
    blas_vendor.cpp
)
target_link_libraries(uni20_backend_blas PUBLIC uni20_deps uni20_interface_blas)

# Handle optional MKL
if(UNI20_BACKEND_MKL)
  add_subdirectory(mkl)
  target_link_libraries(uni20_backend_blas INTERFACE uni20_backend_mkl)
endif()

# Handle optional OPENBLAS
if(UNI20_BACKEND_OPENBLAS)
  add_subdirectory(openblas)
  target_link_libraries(uni20_backend_blas INTERFACE uni20_backend_openblas)
endif()
