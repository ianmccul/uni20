# tests/CMakeLists.txt

enable_testing()

# Create a test executable.
add_executable(uni20_tests
    test_terminal_color.cpp
    test_dummy.cpp   # Include other test files if necessary
)

target_link_libraries(uni20_tests
    PRIVATE uni20 uni20_common
            gtest_main
)

include(GoogleTest)
gtest_discover_tests(uni20_tests)
