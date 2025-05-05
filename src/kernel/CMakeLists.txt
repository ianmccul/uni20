
# src/kernel/CMakeLists.txt

add_library(uni20_kernel INTERFACE)

# the CPU backend is always available
add_subdirectory(cpu)
target_link_libraries(uni20_kernel INTERFACE uni20_kernel_cpu)

if(UNI20_BACKEND_BLAS)
  add_subdirectory(blas)
  target_link_libraries(uni20_kernel INTERFACE uni20_kernel_blas)
endif()

if(UNI20_BACKEND_CUDA)
  add_subdirectory(cuda)
  target_link_libraries(uni20_kernel INTERFACE uni20_kernel_cuda)
endif()
