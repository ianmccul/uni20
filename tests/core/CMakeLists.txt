# tests/core/CMakeLists.txt

add_test_module(core
  SOURCES test_types.cpp test_numeric_limits.cpp test_scalar_traits.cpp test_scalar_concepts.cpp test_scalar_io.cpp test_buildinfo.cpp
  LIBS uni20_core
)

if(UNI20_ENABLE_MPLAPACK)
  add_executable(uni20_mplapack_binary128_tests test_mplapack_binary128.cpp)
  target_link_libraries(uni20_mplapack_binary128_tests
    PRIVATE
      uni20_core
      mplapack::mplapack_binary128
      GTest::gtest_main
  )
  target_compile_definitions(uni20_mplapack_binary128_tests PRIVATE UNIT_TEST)
  gtest_discover_tests(uni20_mplapack_binary128_tests)
endif()
