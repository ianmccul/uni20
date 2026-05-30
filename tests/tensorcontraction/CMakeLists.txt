# tests/tensorcontraction/CMakeLists.txt

add_executable(uni20_tensorcontraction_tests
  test_main.cpp
  test_effective_hamiltonian_plan.cpp
  test_matrix_family.cpp
)

target_link_libraries(uni20_tensorcontraction_tests
  PRIVATE
    uni20_common
    uni20_core
    uni20_tensorcontraction_runtime
    GTest::gtest
)

target_compile_definitions(uni20_tensorcontraction_tests PRIVATE UNIT_TEST)

gtest_discover_tests(uni20_tensorcontraction_tests)
