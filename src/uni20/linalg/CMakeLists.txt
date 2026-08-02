add_library(uni20_linalg STATIC
    backends/cpu/matrix_exponential.cpp
    dispatch_diagnostics.cpp
)

if(TARGET uni20_backend_cuda)
  target_sources(uni20_linalg PRIVATE
      backends/cuda/elementwise_copy.cu
  )
  set_target_properties(uni20_linalg PROPERTIES
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
  )
endif()

target_include_directories(uni20_linalg
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(uni20_linalg
    PUBLIC
        uni20_backend
        uni20_core
        uni20_common
)
