# examples/CMakeLists.txt

# Add an executable target for the dependencies example.
add_executable(mdspan_example mdspan_example.cpp)

# Link the dependencies example with fmt and, if necessary, the uni20 library.
# (If mdspan is fetched via FetchContent, its include directories should already be set.)
target_link_libraries(mdspan_example PRIVATE fmt::fmt mdspan)
