# tests/mdspan/CMakeLists.txt

add_test_module(mdspan
  SOURCES
    test_concepts.cpp
    test_conjugate_accessor.cpp
    test_iteration_plan.cpp
    test_strides.cpp
    test_transform_view.cpp
    test_zip_layout.cpp
  LIBS uni20_mdspan
)
