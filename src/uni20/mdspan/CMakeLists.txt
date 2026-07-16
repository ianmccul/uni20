
# src/uni20/mdspan/CMakeLists.txt

add_library(uni20_mdspan_base INTERFACE)
target_link_libraries(uni20_mdspan_base INTERFACE mdspan uni20_deps)

add_library(uni20_mdspan INTERFACE)
target_link_libraries(uni20_mdspan INTERFACE uni20_mdspan_base)
