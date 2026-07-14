
# src/uni20/kernel/CMakeLists.txt

add_library(uni20_kernel INTERFACE)

# The CPU reference implementation is always available.
add_subdirectory(cpu)
target_link_libraries(uni20_kernel INTERFACE uni20_kernel_cpu)
