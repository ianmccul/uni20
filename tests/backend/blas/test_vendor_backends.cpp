#include <uni20/backend/blas/backend_blas.hpp>

#include <uni20/config.hpp>

#include <gtest/gtest.h>

#if UNI20_BACKEND_MKL
TEST(MKLBackend, ExposesVersionAndThreadControls)
{
  EXPECT_FALSE(uni20::blas::mkl::version_string().empty());
  EXPECT_GT(uni20::blas::mkl::max_threads(), 0);
  EXPECT_GT(uni20::blas::mkl::max_threads(uni20::blas::mkl::Domain::blas), 0);
  EXPECT_NE(uni20::blas::mkl::sequential_backend(), uni20::blas::mkl::threaded_backend());
}
#endif

#if UNI20_BACKEND_OPENBLAS
TEST(OpenBLASBackend, ExposesConfigAndThreadControls)
{
  EXPECT_FALSE(uni20::blas::openblas::config().empty());
  EXPECT_FALSE(uni20::blas::openblas::core_name().empty());
  EXPECT_GT(uni20::blas::openblas::num_threads(), 0);
  EXPECT_GT(uni20::blas::openblas::num_procs(), 0);

  auto const mode = uni20::blas::openblas::parallel_mode();
  EXPECT_TRUE(mode == uni20::blas::openblas::ParallelMode::sequential ||
              mode == uni20::blas::openblas::ParallelMode::thread ||
              mode == uni20::blas::openblas::ParallelMode::openmp);
}
#endif
