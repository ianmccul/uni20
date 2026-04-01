# src/uni20/matrix/CMakeLists.txt

add_library(uni20_matrix INTERFACE)

target_link_libraries(uni20_matrix
    INTERFACE
        uni20_common
        uni20_core
)
