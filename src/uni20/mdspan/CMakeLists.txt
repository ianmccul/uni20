
# src/uni20/mdspan/CMakeLists.txt

add_library(uni20_mdspan INTERFACE)
target_link_libraries(uni20_mdspan INTERFACE mdspan uni20_deps)
