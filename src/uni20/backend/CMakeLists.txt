# src/uni20/backend/CMakeLists.txt

add_library(uni20_backend INTERFACE)
target_link_libraries(uni20_backend INTERFACE uni20_common)

if(UNI20_DOCS_WEB)
  return()
endif()

if(UNI20_ILP64)
  set(BLA_SIZEOF_INTEGER 8 CACHE INTERNAL "BLAS integer size (8 bytes for ILP64)")
else()
  set(BLA_SIZEOF_INTEGER 4 CACHE INTERNAL "BLAS integer size (4 bytes for LP64)")
endif()

if(UNI20_BLAS_VENDOR STREQUAL "")
  set(UNI20_BLAS_VENDOR "All" CACHE STRING "BLAS vendor hint passed to FindBLAS as BLA_VENDOR" FORCE)
endif()
set(BLA_VENDOR "${UNI20_BLAS_VENDOR}" CACHE STRING "BLAS vendor hint used by FindBLAS" FORCE)

message(STATUS "UNI20_ILP64 is set to: ${UNI20_ILP64}")
message(STATUS "Using BLA_SIZEOF_INTEGER = ${BLA_SIZEOF_INTEGER}")

add_subdirectory(lapack)
target_link_libraries(uni20_backend INTERFACE uni20_backend_lapack)

if(UNI20_BACKEND_BLAS)
  add_subdirectory(blas)
  target_link_libraries(uni20_backend INTERFACE uni20_backend_blas)
endif()

if(UNI20_BACKEND_CUDA)
  add_subdirectory(cuda)
  target_link_libraries(uni20_backend INTERFACE uni20_backend_cuda)
endif()

if(UNI20_BACKEND_CUBLAS)
  add_subdirectory(cublas)
  target_link_libraries(uni20_backend INTERFACE uni20_backend_cublas)
endif()
