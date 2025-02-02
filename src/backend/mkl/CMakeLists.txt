find_package(MKL REQUIRED)

add_library(uni20_backend_mkl STATIC
    mkl_backend.cpp
)
# Assume MKL uses the BLAS routines from uni20_backend_blas.
target_link_libraries(uni20_backend_mkl PRIVATE uni20_backend_blas)
#target_include_directories(uni20_backend_mkl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(uni20_backend_mkl PRIVATE MKL::MKL)
