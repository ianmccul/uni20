find_package(CUDAToolkit REQUIRED)

if(NOT TARGET CUDA::cusolver)
  message(FATAL_ERROR "cuSOLVER backend requested but cuSOLVER library is not available.")
endif()

add_library(uni20_backend_cusolver STATIC
    cusolver_backend.cpp
)
# Link against the CUDA backend since it provides the lower-level routines.
target_link_libraries(uni20_backend_cusolver PRIVATE uni20_backend_cuda)
#target_include_directories(uni20_backend_cusolver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link external NVIDIA cuSOLVER library
target_link_libraries(uni20_backend_cusolver PRIVATE CUDA::cusolver)
