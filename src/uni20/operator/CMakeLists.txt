# src/uni20/operator/CMakeLists.txt

add_library(uni20_operator INTERFACE)

target_link_libraries(uni20_operator
    INTERFACE
        uni20_common
        uni20_core
        uni20_matrix
        uni20_symmetry
)
