# tests/core/CMakeLists.txt

add_test_module(core
  SOURCES test_types.cpp test_numeric_limits.cpp test_scalar_traits.cpp test_scalar_concepts.cpp test_scalar_io.cpp
          test_scalar_precision.cpp test_math.cpp test_buildinfo.cpp
  LIBS uni20_core
)

if(UNI20_ENABLE_MPLAPACK)
  add_executable(uni20_mplapack_binary128_tests
    test_mplapack_binary128.cpp
    "${PROJECT_SOURCE_DIR}/tests/uni20_tests_main.cpp"
  )
  target_link_libraries(uni20_mplapack_binary128_tests
    PRIVATE
      uni20_core
      uni20_backend_blas
      mplapack::mplapack_binary128
      GTest::gtest
  )
  target_compile_definitions(uni20_mplapack_binary128_tests PRIVATE UNIT_TEST)
  gtest_discover_tests(uni20_mplapack_binary128_tests)
endif()
