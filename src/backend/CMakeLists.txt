# src/backend/CMakeLists.txt

add_library(uni20_backend INTERFACE)

if(UNI20_BACKEND_BLAS)
  add_subdirectory(blas)
  target_link_libraries(uni20_backend INTERFACE uni20_backend_blas)
endif()

if(UNI20_BACKEND_CUDA)
  add_subdirectory(cuda)
  target_link_libraries(uni20_backend INTERFACE uni20_backend_cuda)
endif()
