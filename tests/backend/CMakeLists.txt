# tests/backend/CMakeLists.txt

if(TARGET uni20_backend_blas)
  add_subdirectory(blas)
endif()

if(TARGET uni20_backend_cuda)
  add_subdirectory(cuda)
endif()

if(TARGET uni20_backend_cublas)
  add_subdirectory(cublas)
endif()

if(TARGET uni20_backend_cusolver)
  add_subdirectory(cusolver)
endif()
