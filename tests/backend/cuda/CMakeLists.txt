add_test_module(cuda_runtime
  SOURCES test_cuda_buffer.cpp test_cuda_runtime.cpp test_debug_cuda_scheduler.cpp
  LIBS uni20_common uni20_async uni20_backend_cuda
)
