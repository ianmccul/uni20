# src/uni20/symmetry/CMakeLists.txt

add_library(uni20_symmetry INTERFACE)

target_link_libraries(uni20_symmetry
    INTERFACE
        uni20_common
        uni20_core
)
