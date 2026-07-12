
message(STATUS "Using BLA_VENDOR = ${BLA_VENDOR} (from UNI20_BLAS_VENDOR)")

find_package(BLAS REQUIRED)

if(NOT BLAS_FOUND)
  message(FATAL_ERROR "BLAS not found.")
endif()

include(DetectBlasVendor)
detect_blas_vendor()

add_library(uni20_interface_blas INTERFACE)

if(TARGET BLAS::BLAS)
  target_link_libraries(uni20_interface_blas INTERFACE BLAS::BLAS)
else()
  target_link_libraries(uni20_interface_blas INTERFACE ${BLAS_LIBRARIES})
endif()

# If FindBLAS sets BLAS_LINKER_FLAGS, attach them to link options:
if(BLAS_LINKER_FLAGS)
  target_link_options(uni20_interface_blas INTERFACE ${BLAS_LINKER_FLAGS})
endif()

add_library(uni20_backend_blas STATIC
    blas_vendor.cpp
)

add_subdirectory(reference)
target_link_libraries(uni20_backend_blas PUBLIC uni20_common uni20_backend_blas_reference)

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
