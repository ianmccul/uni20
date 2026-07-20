find_package(CUDAToolkit REQUIRED)

if(NOT TARGET CUDA::cublas)
  message(FATAL_ERROR "cuBLAS backend requested but the cuBLAS library is not available.")
endif()

add_library(uni20_backend_cublas STATIC
    cublas_error.cpp
    execution.cpp
)
target_link_libraries(uni20_backend_cublas PUBLIC uni20_backend_cuda CUDA::cublas)
