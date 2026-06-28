function(uni20_add_example target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE ${ARGN})
endfunction()

uni20_add_example(buildinfo_example common/buildinfo_example.cpp fmt::fmt uni20_core uni20_common)
uni20_add_example(gtest_floating_eq_example common/gtest_floating_eq_example.cpp gtest fmt::fmt uni20_common)
uni20_add_example(trace_example common/trace_example.cpp fmt::fmt mdspan uni20_common)

uni20_add_example(mdspan_example mdspan/mdspan_example.cpp fmt::fmt mdspan uni20_common)

if(TARGET uni20_backend_blas)
  uni20_add_example(blas_example linalg/blas_example.cpp fmt::fmt uni20_backend_blas)
endif()

uni20_add_example(presentation_example presentation/table_layout_example.cpp fmt::fmt uni20_common)
uni20_add_example(presentation_table_layout_example presentation/table_layout_example.cpp fmt::fmt uni20_common)
uni20_add_example(presentation_glyph_policy_example presentation/glyph_policy_example.cpp fmt::fmt uni20_common)
uni20_add_example(presentation_text_art_example presentation/text_art_example.cpp fmt::fmt uni20_common)
uni20_add_example(presentation_diagnostic_region_example presentation/diagnostic_region_example.cpp fmt::fmt uni20_common)

uni20_add_example(async_example async/async_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_example2 async/async_example2.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_ops_example async/async_ops_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_buffer_semantics_example async/async_buffer_semantics_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_kernel_shapes_example async/async_kernel_shapes_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_tbb_reduction_example async/async_tbb_reduction_example.cpp fmt::fmt uni20_common uni20_async
                  TBB::tbb)
uni20_add_example(async_fib_example async/async_fib_example.cpp fmt::fmt uni20_common uni20_async TBB::tbb)
uni20_add_example(async_dag_debug_example async/async_dag_debug_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_dag_gallery_example async/async_dag_gallery_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_dag_deadlock_tbb_example async/async_dag_deadlock_tbb_example.cpp fmt::fmt uni20_common
                  uni20_async TBB::tbb)
uni20_add_example(future_example async/future_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(bug async/bug.cpp fmt::fmt uni20_common uni20_async TBB::tbb)

uni20_add_example(ad_example ad/ad_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(gradient_solver ad/gradient_solver.cpp fmt::fmt uni20_common uni20_async)

enable_testing()
include(GoogleTest)
