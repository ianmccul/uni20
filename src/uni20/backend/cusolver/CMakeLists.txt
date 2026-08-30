find_package(CUDAToolkit REQUIRED)

if(NOT TARGET CUDA::cusolver)
  message(FATAL_ERROR "cuSOLVER backend requested but cuSOLVER library is not available.")
endif()

add_library(uni20_backend_cusolver STATIC
    cusolver_error.cpp
    execution.cpp
)
target_link_libraries(uni20_backend_cusolver PUBLIC uni20_backend_cuda CUDA::cusolver)
