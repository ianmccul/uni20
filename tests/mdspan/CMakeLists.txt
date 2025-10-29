# tests/mdspan/CMakeLists.txt

add_test_module(mdspan
  SOURCES test_iteration_plan.cpp test_zip_layout.cpp
  LIBS uni20_common uni20_level1 uni20_mdspan mdspan
)
