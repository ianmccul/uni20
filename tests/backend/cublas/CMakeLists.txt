add_test_module(cublas
  SOURCES test_cublas_execution.cpp
  LIBS uni20_async uni20_backend_cublas uni20_linalg
)
