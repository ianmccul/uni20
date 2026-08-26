add_library(uni20_tensor_network INTERFACE)

target_link_libraries(uni20_tensor_network
    INTERFACE
        uni20_krylov
        uni20_symmetry
)
