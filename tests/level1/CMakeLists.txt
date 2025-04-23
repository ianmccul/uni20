# tests/level1/CMakeLists.txt

# suppress “unused-function” warnings in all translation units in this directory
add_compile_options(-Wno-unused-function)

add_test_module(level1
  SOURCES test_apply_unary.cpp test_sum.cpp test_zip_layout.cpp
  LIBS uni20_common uni20_level1
)
