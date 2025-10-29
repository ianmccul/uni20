# tests/level1/CMakeLists.txt

add_test_module(level1
  SOURCES test_apply_unary.cpp test_sum.cpp test_zip_transform.cpp test_assign.cpp
  LIBS uni20_common uni20_level1
)
