
if(UNI20_ILP64)
  set(BLA_SIZEOF_INTEGER 8 CACHE INTERNAL "BLAS integer size (8 bytes for ILP64)")
else()
  set(BLA_SIZEOF_INTEGER 4 CACHE INTERNAL "BLAS integer size (4 bytes for LP64)")
endif()

message(STATUS "UNI20_ILP64 is set to: ${UNI20_ILP64}")
message(STATUS "Using BLA_SIZEOF_INTEGER = ${BLA_SIZEOF_INTEGER}")

if(UNI20_BLAS_VENDOR STREQUAL "")
  set(UNI20_BLAS_VENDOR "All" CACHE STRING "BLAS vendor hint passed to FindBLAS as BLA_VENDOR" FORCE)
endif()

set(BLA_VENDOR "${UNI20_BLAS_VENDOR}" CACHE STRING "BLAS vendor hint used by FindBLAS" FORCE)
message(STATUS "Using BLA_VENDOR = ${BLA_VENDOR} (from UNI20_BLAS_VENDOR)")

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(NOT BLAS_FOUND)
  message(FATAL_ERROR "BLAS not found.")
endif()

if(NOT LAPACK_FOUND)
  message(FATAL_ERROR "LAPACK not found.")
endif()

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

add_subdirectory(reference)
target_link_libraries(uni20_backend_blas PUBLIC uni20_backend_blas_reference)

# Handle optional MKL
if(UNI20_BACKEND_MKL)
  add_subdirectory(mkl)
  target_link_libraries(uni20_backend_blas INTERFACE uni20_backend_mkl)
endif()

# Handle optional OpenBLAS extension API
if(UNI20_BACKEND_OPENBLAS)
  if(NOT UNI20_DETECTED_BLAS_VENDOR STREQUAL "OpenBLAS")
    message(FATAL_ERROR
      "UNI20_BACKEND_OPENBLAS requires UNI20_DETECTED_BLAS_VENDOR=OpenBLAS, "
      "but detected '${UNI20_DETECTED_BLAS_VENDOR}'. Set UNI20_BLAS_VENDOR=OpenBLAS "
      "or disable UNI20_BACKEND_OPENBLAS.")
  endif()
  add_subdirectory(openblas)
  target_link_libraries(uni20_backend_blas INTERFACE uni20_backend_openblas)
endif()
