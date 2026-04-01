# src/uni20/models/CMakeLists.txt

add_library(uni20_models INTERFACE)

target_link_libraries(uni20_models
    INTERFACE
        uni20_common
        uni20_core
        uni20_matrix
        uni20_operator
        uni20_symmetry
)
