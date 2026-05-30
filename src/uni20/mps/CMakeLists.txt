# src/uni20/mps/CMakeLists.txt

add_library(uni20_mps INTERFACE)

target_link_libraries(uni20_mps
  INTERFACE
    uni20_core
    uni20_operator
    uni20_symmetry
    uni20_tensorcontraction_runtime
)
