add_library(uni20_backend_blas STATIC
    blas_backend.cpp
)
target_link_libraries(uni20_backend_blas PRIVATE uni20_common)
#target_include_directories(uni20_backend_blas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
