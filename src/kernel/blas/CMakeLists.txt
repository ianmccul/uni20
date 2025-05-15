include(FindBLAS)  # 通常你前面就有
find_package(BLAS REQUIRED)

add_library(uni20_kernel_blas INTERFACE)
target_link_libraries(uni20_kernel_blas INTERFACE ${BLAS_LIBRARIES})
