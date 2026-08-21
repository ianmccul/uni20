function(uni20_add_example target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE ${ARGN})
endfunction()

uni20_add_example(buildinfo_example common/buildinfo_example.cpp fmt::fmt uni20_core uni20_common)
uni20_add_example(gtest_floating_eq_example common/gtest_floating_eq_example.cpp gtest fmt::fmt uni20_common)
uni20_add_example(trace_example common/trace_example.cpp fmt::fmt mdspan uni20_common)

uni20_add_example(mdspan_example mdspan/mdspan_example.cpp fmt::fmt mdspan uni20_common)

uni20_add_example(block_tensor_example symmetry/block_tensor_example.cpp uni20_symmetry)
add_test(NAME BlockTensorExample COMMAND block_tensor_example)

if(TARGET uni20_backend_cuda)
  uni20_add_example(cuda_hello_world_example cuda/cuda_hello_world_example.cpp fmt::fmt uni20_core uni20_common
                    uni20_backend_cuda)
else()
  uni20_add_example(cuda_hello_world_example cuda/cuda_hello_world_example.cpp fmt::fmt uni20_core uni20_common)
endif()

if(TARGET uni20_backend_blas)
  uni20_add_example(blas_example linalg/blas_example.cpp fmt::fmt uni20_backend_blas)
endif()

uni20_add_example(kernel_dispatch_example linalg/kernel_dispatch_example.cpp fmt::fmt uni20_linalg)
uni20_add_example(kernel_dispatch_error_example linalg/kernel_dispatch_error_example.cpp fmt::fmt uni20_linalg)
if(TARGET uni20_backend_blas)
  uni20_add_example(gemm_dispatch_example linalg/gemm_dispatch_example.cpp fmt::fmt uni20_linalg
                    uni20_backend_blas)
else()
  uni20_add_example(gemm_dispatch_example linalg/gemm_dispatch_example.cpp fmt::fmt uni20_linalg)
endif()
add_test(NAME KernelDispatchExample.OrderedFallback COMMAND kernel_dispatch_example)
add_test(NAME KernelDispatchExample.ErrorReporting COMMAND kernel_dispatch_error_example)
add_test(NAME KernelDispatchExample.TensorGemmFp32
    COMMAND gemm_dispatch_example --precision=fp32 --diagnostics=on)
add_test(NAME KernelDispatchExample.TensorGemmFp64
    COMMAND gemm_dispatch_example --precision=fp64 --diagnostics=on)
if(UNI20_HAS_FLOAT128)
  add_test(NAME KernelDispatchExample.TensorGemmFp128
      COMMAND gemm_dispatch_example --precision=fp128 --diagnostics=on)
endif()

uni20_add_example(krylov_matrix_market_example krylov/krylov_matrix_market_example.cpp fmt::fmt uni20_common
                  uni20_krylov)
target_compile_definitions(krylov_matrix_market_example
    PRIVATE
        UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR="${PROJECT_SOURCE_DIR}/tests/krylov/matrix_market"
)
add_test(NAME KrylovMatrixMarketExample.Float
    COMMAND krylov_matrix_market_example
        "${PROJECT_SOURCE_DIR}/tests/krylov/matrix_market/path_laplacian_30.mtx"
        --precision=float --which=LA --nev=2 --ncv=8 --tol=1e-5 --max-iters=100 --diagnostics=none)
add_test(NAME KrylovMatrixMarketExample.Double
    COMMAND krylov_matrix_market_example
        "${PROJECT_SOURCE_DIR}/tests/krylov/matrix_market/path_laplacian_30.mtx"
        --precision=double --which=LA --nev=2 --ncv=8 --tol=1e-10 --max-iters=100 --diagnostics=none)
add_test(NAME KrylovMatrixMarketExample.ShiftInvertDouble
    COMMAND krylov_matrix_market_example
        "${PROJECT_SOURCE_DIR}/tests/krylov/matrix_market/path_laplacian_30.mtx"
        --precision=double --mode=shift-invert --which=LM --nev=2 --ncv=8 --tol=1e-10 --max-iters=100
        --diagnostics=summary)
if(UNI20_HAS_FLOAT128)
  add_test(NAME KrylovMatrixMarketExample.Float128
      COMMAND krylov_matrix_market_example
          "${PROJECT_SOURCE_DIR}/tests/krylov/matrix_market/path_laplacian_30.mtx"
          --precision=float128 --which=LA --nev=2 --ncv=8 --tol=1e-20 --max-iters=100 --diagnostics=none)
endif()

uni20_add_example(krylov_nonsymmetric_matrix_market_example krylov/krylov_nonsymmetric_matrix_market_example.cpp
                  fmt::fmt uni20_common uni20_krylov)
target_compile_definitions(krylov_nonsymmetric_matrix_market_example
    PRIVATE
        UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR="${PROJECT_SOURCE_DIR}/tests/krylov/matrix_market"
)
add_test(NAME KrylovNonsymmetricMatrixMarketExample.Lop163
    COMMAND krylov_nonsymmetric_matrix_market_example
        "${PROJECT_SOURCE_DIR}/tests/krylov/matrix_market/nep/stoch/lop163.mtx"
        --which=LM --nev=3 --nkeep=6 --ncv=24 --tol=1e-10 --diagnostics=summary --eigenvectors=false)
if(UNI20_HAS_FLOAT128)
  add_test(NAME KrylovNonsymmetricMatrixMarketExample.Lop163Float128
      COMMAND krylov_nonsymmetric_matrix_market_example
          "${PROJECT_SOURCE_DIR}/tests/krylov/matrix_market/nep/stoch/lop163.mtx"
          --precision=float128 --which=LM --nev=3 --nkeep=6 --ncv=24 --tol=1e-20
          --max-iters=500 --diagnostics=summary --eigenvectors=false)
endif()

uni20_add_example(krylov_exponential_probe_example krylov/krylov_exponential_probe_example.cpp fmt::fmt uni20_common
                  uni20_krylov)
add_test(NAME KrylovExponentialProbeExample.Float
    COMMAND krylov_exponential_probe_example --precision=float --tol=1e-4 --check)
add_test(NAME KrylovExponentialProbeExample.Double
    COMMAND krylov_exponential_probe_example --precision=double --tol=1e-8 --check)

uni20_add_example(krylov_exponential_orthogonality_example krylov/krylov_exponential_orthogonality_example.cpp
                  fmt::fmt uni20_common uni20_krylov)
add_test(NAME KrylovExponentialOrthogonalityExample.Float
    COMMAND krylov_exponential_orthogonality_example --precision=float --tol=1e-8 --check)

uni20_add_example(krylov_exponential_matrix_market_probe_example
                  krylov/krylov_exponential_matrix_market_probe_example.cpp fmt::fmt uni20_common uni20_krylov)
target_compile_definitions(krylov_exponential_matrix_market_probe_example
    PRIVATE
        UNI20_KRYLOV_EXAMPLE_MATRIX_MARKET_DIR="${PROJECT_SOURCE_DIR}/tests/krylov/matrix_market"
)
add_test(NAME KrylovExponentialMatrixMarketProbeExample.TdvpDouble
    COMMAND krylov_exponential_matrix_market_probe_example --check)
add_test(NAME KrylovExponentialMatrixMarketProbeExample.TdvpFloat
    COMMAND krylov_exponential_matrix_market_probe_example
        --precision=float --tol=1e-8 --time-imag=-0.01968473663975394 --max-dim=32 --check)
if(UNI20_HAS_FLOAT128)
  add_test(NAME KrylovExponentialMatrixMarketProbeExample.TdvpFloat128
      COMMAND krylov_exponential_matrix_market_probe_example
          --precision=float128 --tol=1e-20 --time-imag=-0.01968473663975394 --max-dim=32)
endif()

uni20_add_example(presentation_example presentation/table_layout_example.cpp fmt::fmt uni20_common)
uni20_add_example(presentation_table_layout_example presentation/table_layout_example.cpp fmt::fmt uni20_common)
uni20_add_example(presentation_glyph_policy_example presentation/glyph_policy_example.cpp fmt::fmt uni20_common)
uni20_add_example(presentation_text_art_example presentation/text_art_example.cpp fmt::fmt uni20_common)
uni20_add_example(presentation_diagnostic_region_example presentation/diagnostic_region_example.cpp fmt::fmt uni20_common)
uni20_add_example(presentation_mdspan_example presentation/mdspan_example.cpp fmt::fmt mdspan uni20_common)
uni20_add_example(display_example presentation/display_example.cpp fmt::fmt uni20_common)

uni20_add_example(async_example async/async_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_example2 async/async_example2.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_ops_example async/async_ops_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_buffer_semantics_example async/async_buffer_semantics_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_buffer_await_paths_example async/async_buffer_await_paths_example.cpp fmt::fmt uni20_common
                  uni20_async)
add_test(NAME AsyncBufferAwaitPathsExample COMMAND async_buffer_await_paths_example)
uni20_add_example(async_kernel_shapes_example async/async_kernel_shapes_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_tensor_matrix_product_example async/async_tensor_matrix_product_example.cpp fmt::fmt
                  uni20_linalg uni20_async)
add_test(NAME AsyncTensorMatrixProductExample COMMAND async_tensor_matrix_product_example)
uni20_add_example(async_tbb_matrix_product_batch_example async/async_tbb_matrix_product_batch_example.cpp fmt::fmt
                  uni20_linalg uni20_async TBB::tbb)
add_test(NAME AsyncTbbMatrixProductBatchExample
    COMMAND async_tbb_matrix_product_batch_example
        --size=16 --products=6 --threads=3 --backend=cpu --precision=fp64)
if(UNI20_HAS_FLOAT128)
  add_test(NAME AsyncTbbMatrixProductBatchExample.Float128
      COMMAND async_tbb_matrix_product_batch_example
          --size=12 --products=4 --threads=3 --backend=auto --precision=fp128)
endif()
uni20_add_example(async_tbb_reduction_example async/async_tbb_reduction_example.cpp fmt::fmt uni20_common uni20_async
                  TBB::tbb)
uni20_add_example(async_ad_stress_example async/async_ad_stress_example.cpp fmt::fmt uni20_common uni20_async TBB::tbb)
add_test(NAME AsyncAdStressExample.Smoke
    COMMAND async_ad_stress_example
        --threads=2 --iterations=1 --depth=3 --dead-branches=4 --failing-branches=1 --seed=1234)
uni20_add_example(async_fib_example async/async_fib_example.cpp fmt::fmt uni20_common uni20_async TBB::tbb)
uni20_add_example(async_dag_debug_example async/async_dag_debug_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_dag_gallery_example async/async_dag_gallery_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(async_dag_deadlock_tbb_example async/async_dag_deadlock_tbb_example.cpp fmt::fmt uni20_common
                  uni20_async TBB::tbb)
uni20_add_example(async_diagnostics_guide_example async/async_diagnostics_guide_example.cpp fmt::fmt uni20_common
                  uni20_async)
uni20_add_example(async_coroutine_failure_example async/async_coroutine_failure_example.cpp fmt::fmt uni20_common
                  uni20_async)
add_test(NAME AsyncCoroutineFailureExample COMMAND async_coroutine_failure_example --no-dot)
uni20_add_example(future_example async/future_example.cpp fmt::fmt uni20_common uni20_async)

uni20_add_example(ad_example ad/ad_example.cpp fmt::fmt uni20_common uni20_async)
uni20_add_example(gradient_solver ad/gradient_solver.cpp fmt::fmt uni20_common uni20_async)

enable_testing()
include(GoogleTest)
