if(UNI20_BACKEND_CUDA AND (NOT UNI20_ENABLE_CUDA))
  message(FATAL_ERROR "CUDA backend requested but CUDA support is disabled (UNI20_ENABLE_CUDA is OFF)")
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT TARGET CUDA::cublas)
  message(FATAL_ERROR "CUDA backend requested but cuBLAS library is not available.")
endif()

add_library(uni20_backend_cuda STATIC
    cuda_backend.cpp
)
target_link_libraries(uni20_backend_cuda PRIVATE uni20_common)
#target_include_directories(uni20_backend_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# You might also need to enable CUDA language and link CUDA libraries if required.
