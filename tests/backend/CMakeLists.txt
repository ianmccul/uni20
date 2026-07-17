# tests/backend/CMakeLists.txt

if(TARGET uni20_backend_blas)
  add_subdirectory(blas)
endif()

if(TARGET uni20_backend_cuda)
  add_subdirectory(cuda)
endif()
