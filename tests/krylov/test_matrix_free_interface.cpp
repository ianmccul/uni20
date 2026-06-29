#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/symmetric_lanczos.hpp>

#include "krylov_test_types.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

template <typename Scalar> class KrylovMatrixFreeInterfaceRealTypedTest : public ::testing::Test {};

using RealTypes = uni20::krylov::test::KrylovRealTestTypes;
TYPED_TEST_SUITE(KrylovMatrixFreeInterfaceRealTypedTest, RealTypes);

template <typename Scalar> double scalar_tolerance()
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return 1.0e-4;
  }
  else
  {
    return 1.0e-12;
  }
}

class VectorOnlyOps {
  public:
    explicit VectorOnlyOps(std::size_t dimension) : dimension_(dimension) {}

    [[nodiscard]] std::size_t problem_dimension() const noexcept { return dimension_; }

    [[nodiscard]] std::size_t vector_dimension(uni20::krylov::DenseHostVector<double> const& x) const noexcept
    {
      return x.values.size();
    }

    [[nodiscard]] uni20::krylov::DenseHostVector<double> allocate_like(uni20::krylov::DenseHostVector<double> const& x)
    {
      return uni20::krylov::DenseHostVector<double>{std::vector<double>(x.values.size())};
    }

    void copy(uni20::krylov::DenseHostVector<double>& dst, uni20::krylov::DenseHostVector<double> const& src)
    {
      dst.values = src.values;
    }

    void axpy(uni20::krylov::DenseHostVector<double>& y, double alpha, uni20::krylov::DenseHostVector<double> const& x)
    {
      for (std::size_t i = 0; i < y.values.size(); ++i)
      {
        y.values[i] += alpha * x.values[i];
      }
    }

    void scal(uni20::krylov::DenseHostVector<double>& x, double alpha)
    {
      for (double& value : x.values)
      {
        value *= alpha;
      }
    }

    void set_zero(uni20::krylov::DenseHostVector<double>& x)
    {
      for (double& value : x.values)
      {
        value = 0.0;
      }
    }

    [[nodiscard]] double inner_product(uni20::krylov::DenseHostVector<double> const& x,
                                       uni20::krylov::DenseHostVector<double> const& y)
    {
      double result = 0.0;
      for (std::size_t i = 0; i < x.values.size(); ++i)
      {
        result += x.values[i] * y.values[i];
      }
      return result;
    }

  private:
    std::size_t dimension_ = 0;
};

static_assert(uni20::krylov::KrylovVectorOps<VectorOnlyOps, uni20::krylov::DenseHostVector<double>, double>);
static_assert(!uni20::krylov::KrylovOperator<VectorOnlyOps, uni20::krylov::DenseHostVector<double>>);
static_assert(!uni20::krylov::KrylovMatrixFreeOperator<VectorOnlyOps, uni20::krylov::DenseHostVector<double>, double>);

TEST(KrylovMatrixFreeInterface, DenseHostOpsUseDeclaredOperations)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;

  DenseHostVectorOps<double> ops(3, {
                                        2.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        3.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        5.0,
                                    });

  DenseHostVector<double> x{{1.0, 2.0, 3.0}};
  auto y = ops.allocate_like(x);

  EXPECT_EQ(ops.problem_dimension(), 3);
  EXPECT_EQ(ops.vector_dimension(x), 3);

  ops.set_zero(y);
  ops.matvec(y, x);
  EXPECT_EQ(y.values, (std::vector<double>{2.0, 6.0, 15.0}));

  ops.axpy(y, -1.0, x);
  EXPECT_EQ(y.values, (std::vector<double>{1.0, 4.0, 12.0}));

  ops.scal(y, 0.5);
  EXPECT_EQ(y.values, (std::vector<double>{0.5, 2.0, 6.0}));

  EXPECT_DOUBLE_EQ(ops.inner_product(x, y), 22.5);
  EXPECT_DOUBLE_EQ(ops.norm(x), std::sqrt(14.0));
  EXPECT_DOUBLE_EQ(uni20::krylov::norm_or_inner_product<double>(ops, x), std::sqrt(14.0));

  auto z = ops.allocate_like(x);
  ops.copy(z, y);
  EXPECT_EQ(z.values, y.values);

  EXPECT_EQ(ops.allocation_count(), 2);
  EXPECT_EQ(ops.copy_count(), 1);
  EXPECT_EQ(ops.axpy_count(), 1);
  EXPECT_EQ(ops.scal_count(), 1);
  EXPECT_EQ(ops.set_zero_count(), 1);
  EXPECT_EQ(ops.inner_product_count(), 1);
  EXPECT_EQ(ops.norm_count(), 2);
  EXPECT_EQ(ops.matvec_count(), 1);
}

TEST(KrylovMatrixFreeInterface, NormHelperFallsBackToInnerProduct)
{
  VectorOnlyOps ops(3);
  uni20::krylov::DenseHostVector<double> x{{2.0, -3.0, 6.0}};

  static_assert(!uni20::krylov::KrylovNorm<VectorOnlyOps, uni20::krylov::DenseHostVector<double>, double>);
  EXPECT_DOUBLE_EQ(uni20::krylov::norm_or_inner_product<double>(ops, x), 7.0);
}

TEST(KrylovMatrixFreeInterface, NonsymmetricDefaultsUseRealFirstPolicy)
{
  uni20::krylov::NonsymmetricEigenParams<double> params;

  EXPECT_EQ(params.eigenvalue_count, 1);
  EXPECT_EQ(params.spectrum, uni20::krylov::SpectrumPart::LargestMagnitude);
  EXPECT_EQ(params.real_policy, uni20::krylov::RealNonsymmetricPolicy::RequireRealEigenpairs);
  EXPECT_EQ(params.complex_pair_tolerance, 0.0);

  uni20::krylov::NonsymmetricEigenResult<double, uni20::krylov::DenseHostVector<double>> result;
  EXPECT_EQ(result.status, uni20::krylov::NonsymmetricStatus::Converged);
}

TEST(KrylovMatrixFreeInterface, ComputesSymmetricDefaultRestartDimensions)
{
  uni20::krylov::SymmetricEigenParams<double> params;
  params.eigenvalue_count = 4;

  EXPECT_EQ(uni20::krylov::default_symmetric_krylov_dimension(params.eigenvalue_count, 512), 20);
  EXPECT_EQ(uni20::krylov::effective_symmetric_krylov_dimension(params, 512), 20);
  EXPECT_EQ(uni20::krylov::effective_symmetric_retained_ritz_count(params, 20), 4);

  params.eigenvalue_count = 12;
  EXPECT_EQ(uni20::krylov::effective_symmetric_krylov_dimension(params, 512), 25);

  params.eigenvalue_count = 4;
  params.krylov_dimension = 18;
  params.retained_ritz_count = 6;
  EXPECT_EQ(uni20::krylov::effective_symmetric_krylov_dimension(params, 512), 18);
  EXPECT_EQ(uni20::krylov::effective_symmetric_retained_ritz_count(params, 18), 6);
}

TEST(KrylovMatrixFreeInterface, ComputesNonsymmetricDefaultRestartDimensions)
{
  uni20::krylov::NonsymmetricEigenParams<double> params;
  params.eigenvalue_count = 4;

  EXPECT_EQ(uni20::krylov::default_nonsymmetric_krylov_dimension(params.eigenvalue_count, 512), 32);
  EXPECT_EQ(uni20::krylov::effective_nonsymmetric_krylov_dimension(params, 512), 32);
  EXPECT_EQ(uni20::krylov::effective_nonsymmetric_retained_ritz_count(params, 32), 8);

  params.eigenvalue_count = 1;
  EXPECT_EQ(uni20::krylov::effective_nonsymmetric_krylov_dimension(params, 512), 20);
  EXPECT_EQ(uni20::krylov::effective_nonsymmetric_retained_ritz_count(params, 20), 5);

  params.krylov_dimension = 12;
  params.retained_ritz_count = 3;
  EXPECT_EQ(uni20::krylov::effective_nonsymmetric_krylov_dimension(params, 512), 12);
  EXPECT_EQ(uni20::krylov::effective_nonsymmetric_retained_ritz_count(params, 12), 3);
}

TEST(KrylovMatrixFreeInterface, ClassifiesRealArnoldiRitzRealityWithScaleAwareTolerance)
{
  using uni20::krylov::RitzReality;

  EXPECT_EQ((uni20::krylov::classify_ritz_reality<double>({2.0, 1.0e-13}, 1.0e-12, 2.0)), RitzReality::Real);
  EXPECT_EQ((uni20::krylov::classify_ritz_reality<double>({2.0, 5.0e-12}, 1.0e-12, 2.0)), RitzReality::Ambiguous);
  EXPECT_EQ((uni20::krylov::classify_ritz_reality<double>({2.0, 5.0e-10}, 1.0e-12, 2.0)), RitzReality::Complex);
}

TEST(KrylovMatrixFreeInterface, RitzRealityDefaultToleranceScalesWithPrecision)
{
  using uni20::krylov::RitzReality;

  double const double_tol = static_cast<double>(uni20::krylov::default_complex_pair_tolerance<double>());
  double const float_tol = static_cast<double>(uni20::krylov::default_complex_pair_tolerance<float>());

  EXPECT_LT(double_tol, 1.0e-7);
  EXPECT_GT(float_tol, double_tol);
  EXPECT_EQ((uni20::krylov::classify_ritz_reality<double>({1.0, 0.5 * double_tol})), RitzReality::Real);
  EXPECT_EQ((uni20::krylov::classify_ritz_reality<double>({1.0, 5.0 * double_tol})), RitzReality::Ambiguous);
}

template <typename Scalar> class CountingMetricOps {
  public:
    explicit CountingMetricOps(std::size_t dimension) : dimension_(dimension) {}

    [[nodiscard]] int metric_inner_product_count() const noexcept { return metric_inner_product_count_; }

    [[nodiscard]] Scalar metric_inner_product(uni20::krylov::DenseHostVector<Scalar> const& x,
                                              uni20::krylov::DenseHostVector<Scalar> const& y)
    {
      if (x.values.size() != dimension_ || y.values.size() != dimension_)
      {
        throw std::invalid_argument("metric test vectors have the wrong size");
      }
      ++metric_inner_product_count_;
      Scalar result{};
      for (std::size_t i = 0; i < dimension_; ++i)
      {
        result += x.values[i] * y.values[i];
      }
      return result;
    }

  private:
    std::size_t dimension_ = 0;
    int metric_inner_product_count_ = 0;
};

TYPED_TEST(KrylovMatrixFreeInterfaceRealTypedTest, MetricInnerProductCustomizationBypassesDefaultBApply)
{
  using Scalar = TypeParam;
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SymmetricSpectralTransform;
  using uni20::krylov::SymmetricSpectralTransformOptions;

  DenseHostVectorOps<Scalar> op_ops(3, {
                                           Scalar{2.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{3.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{5.0},
                                       });
  DenseHostVectorOps<Scalar> b_ops(3, {
                                          Scalar{1.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{1.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{1.0},
                                      });
  CountingMetricOps<Scalar> metric_ops(3);
  DenseHostVector<Scalar> initial{{Scalar{1.0}, Scalar{1.0}, Scalar{1.0}}};

  uni20::krylov::SymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = 1;
  params.krylov_dimension = 3;
  params.spectrum = uni20::krylov::SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  SymmetricSpectralTransformOptions<Scalar> options;
  options.transform = SymmetricSpectralTransform::GeneralizedInverse;
  auto result = uni20::krylov::symmetric_lanczos_restarted_generalized_transformed<Scalar>(op_ops, b_ops, metric_ops,
                                                                                           initial, params, options);

  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.eigenvalues.size(), 1);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 5.0, scalar_tolerance<Scalar>());
  EXPECT_GT(metric_ops.metric_inner_product_count(), 0);
  EXPECT_EQ(b_ops.matvec_count(), 0);
}

TYPED_TEST(KrylovMatrixFreeInterfaceRealTypedTest, GeneralizedMetricFallbackAppliesBIntoScratch)
{
  using Scalar = TypeParam;
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::DenseHostVectorOps;
  using uni20::krylov::SymmetricSpectralTransform;
  using uni20::krylov::SymmetricSpectralTransformOptions;

  DenseHostVectorOps<Scalar> op_ops(3, {
                                           Scalar{2.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{3.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{0.0},
                                           Scalar{5.0},
                                       });
  DenseHostVectorOps<Scalar> b_ops(3, {
                                          Scalar{1.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{1.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{0.0},
                                          Scalar{1.0},
                                      });
  DenseHostVector<Scalar> initial{{Scalar{1.0}, Scalar{1.0}, Scalar{1.0}}};

  uni20::krylov::SymmetricEigenParams<Scalar> params;
  params.eigenvalue_count = 1;
  params.krylov_dimension = 3;
  params.spectrum = uni20::krylov::SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  SymmetricSpectralTransformOptions<Scalar> options;
  options.transform = SymmetricSpectralTransform::GeneralizedInverse;
  auto result = uni20::krylov::symmetric_lanczos_restarted_generalized_transformed<Scalar>(op_ops, b_ops, initial,
                                                                                           params, options);

  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.eigenvalues.size(), 1);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 5.0, scalar_tolerance<Scalar>());
  EXPECT_GT(b_ops.matvec_count(), 0);
}

class NestedSolveMatvecOps {
  public:
    NestedSolveMatvecOps(std::size_t dimension, std::vector<double> matrix)
        : dimension_(dimension), matrix_(std::move(matrix))
    {
      if (matrix_.size() != dimension_ * dimension_)
      {
        throw std::invalid_argument("nested-solve test matrix has the wrong size");
      }
    }

    [[nodiscard]] std::size_t problem_dimension() const noexcept { return dimension_; }

    [[nodiscard]] std::size_t vector_dimension(uni20::krylov::DenseHostVector<double> const& x) const noexcept
    {
      return x.values.size();
    }

    [[nodiscard]] int matvec_count() const noexcept { return matvec_count_; }
    [[nodiscard]] bool nested_solve_ran() const noexcept { return nested_solve_ran_; }

    [[nodiscard]] uni20::krylov::DenseHostVector<double> allocate_like(uni20::krylov::DenseHostVector<double> const& x)
    {
      return uni20::krylov::DenseHostVector<double>{std::vector<double>(x.values.size())};
    }

    void copy(uni20::krylov::DenseHostVector<double>& dst, uni20::krylov::DenseHostVector<double> const& src)
    {
      require_same_size(dst, src);
      dst.values = src.values;
    }

    void axpy(uni20::krylov::DenseHostVector<double>& y, double alpha, uni20::krylov::DenseHostVector<double> const& x)
    {
      require_same_size(y, x);
      for (std::size_t i = 0; i < y.values.size(); ++i)
      {
        y.values[i] += alpha * x.values[i];
      }
    }

    void scal(uni20::krylov::DenseHostVector<double>& x, double alpha)
    {
      for (double& value : x.values)
      {
        value *= alpha;
      }
    }

    void set_zero(uni20::krylov::DenseHostVector<double>& x)
    {
      for (double& value : x.values)
      {
        value = 0.0;
      }
    }

    [[nodiscard]] double inner_product(uni20::krylov::DenseHostVector<double> const& x,
                                       uni20::krylov::DenseHostVector<double> const& y)
    {
      require_same_size(x, y);
      double result = 0.0;
      for (std::size_t i = 0; i < x.values.size(); ++i)
      {
        result += x.values[i] * y.values[i];
      }
      return result;
    }

    void matvec(uni20::krylov::DenseHostVector<double>& y, uni20::krylov::DenseHostVector<double> const& x)
    {
      if (!nested_solve_ran_)
      {
        nested_solve_ran_ = true;
        uni20::krylov::DenseHostVectorOps<double> inner_ops(3, {
                                                                   2.0,
                                                                   0.0,
                                                                   0.0,
                                                                   0.0,
                                                                   3.0,
                                                                   0.0,
                                                                   0.0,
                                                                   0.0,
                                                                   5.0,
                                                               });
        uni20::krylov::DenseHostVector<double> inner_initial{{1.0, 1.0, 1.0}};
        uni20::krylov::SymmetricEigenParams<double> inner_params;
        inner_params.eigenvalue_count = 1;
        inner_params.krylov_dimension = 3;
        inner_params.spectrum = uni20::krylov::SpectrumPart::LargestAlgebraic;
        inner_params.compute_eigenvectors = false;

        auto inner_result = uni20::krylov::symmetric_lanczos_standard<double>(inner_ops, inner_initial, inner_params);
        if (inner_result.status != 0 || inner_result.eigenvalues.empty() ||
            std::abs(inner_result.eigenvalues[0] - 5.0) > 1.0e-12)
        {
          throw std::runtime_error("nested native Krylov solve failed");
        }
      }

      if (x.values.size() != dimension_ || y.values.size() != dimension_)
      {
        throw std::invalid_argument("nested-solve test matvec vector has the wrong size");
      }

      ++matvec_count_;
      for (std::size_t row = 0; row < dimension_; ++row)
      {
        double value = 0.0;
        for (std::size_t col = 0; col < dimension_; ++col)
        {
          value += matrix_[row * dimension_ + col] * x.values[col];
        }
        y.values[row] = value;
      }
    }

  private:
    static void require_same_size(uni20::krylov::DenseHostVector<double> const& lhs,
                                  uni20::krylov::DenseHostVector<double> const& rhs)
    {
      if (lhs.values.size() != rhs.values.size())
      {
        throw std::invalid_argument("nested-solve test vectors have different sizes");
      }
    }

    std::size_t dimension_ = 0;
    std::vector<double> matrix_;
    int matvec_count_ = 0;
    bool nested_solve_ran_ = false;
};

static_assert(
    uni20::krylov::KrylovMatrixFreeOperator<NestedSolveMatvecOps, uni20::krylov::DenseHostVector<double>, double>);

TEST(KrylovMatrixFreeInterface, NativeSolveCanBeReenteredFromMatvecCallback)
{
  using uni20::krylov::DenseHostVector;
  using uni20::krylov::SpectrumPart;
  using uni20::krylov::SymmetricEigenParams;

  NestedSolveMatvecOps ops(4, {
                                  1.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  2.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  4.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  8.0,
                              });
  DenseHostVector<double> initial{{1.0, 1.0, 1.0, 1.0}};

  SymmetricEigenParams<double> params;
  params.eigenvalue_count = 1;
  params.krylov_dimension = 4;
  params.spectrum = SpectrumPart::LargestAlgebraic;
  params.compute_eigenvectors = false;

  auto result = uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params);

  ASSERT_EQ(result.status, 0);
  ASSERT_EQ(result.eigenvalues.size(), 1);
  EXPECT_NEAR(result.eigenvalues[0], 8.0, 1.0e-12);
  EXPECT_TRUE(ops.nested_solve_ran());
  EXPECT_EQ(ops.matvec_count(), 4);
}

} // namespace
