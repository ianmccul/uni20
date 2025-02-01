# src/CMakeLists.txt

# Add the core library with its source files.
add_library(uni20
    core/dummy.cpp           # <-- Include the dummy source file
    # ... add other source files as needed
)

# Specify include directories for the public interface.
target_include_directories(uni20 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/backends
)

# Link external libraries (for example, fmt)
target_link_libraries(uni20 PUBLIC fmt::fmt)
