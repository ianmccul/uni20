#pragma once

/**
 * \file truncated_svd.hpp
 * \ingroup linalg
 * \brief Tensor-facing truncated dense singular value decompositions.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/math.hpp>
#include <uni20/linalg/ops/svd.hpp>
#include <uni20/tensor/tensor.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::linalg
{

/// \brief Rank-selection policy for a truncated singular value decomposition.
/// \details Active cutoff and discarded-weight criteria each impose a minimum
///          rank. Their maximum is then clamped by the explicit extent bounds.
///          The maximum retained extent is a hard cap and may therefore prevent
///          another requested criterion from being satisfied.
template <uni20::Real Real> struct SvdTruncationPolicy
{
    std::size_t minimum_retained_extent = 0;
    std::size_t maximum_retained_extent = uni20::numeric_limits<std::size_t>::max();
    std::optional<Real> singular_value_cutoff = std::nullopt;
    std::optional<Real> normalized_squared_singular_value_cutoff = std::nullopt;
    std::optional<Real> maximum_discarded_weight = std::nullopt;
};

/// \brief Statistics describing a completed SVD truncation.
template <uni20::Real Real> struct SvdTruncationInfo
{
    std::size_t available_rank;
    std::size_t retained_rank;
    Real original_squared_norm;
    Real discarded_weight;
    std::optional<Real> smallest_retained_singular_value;
    std::optional<Real> largest_discarded_singular_value;
};

/// \brief Owning factors and statistics returned by `truncated_svd`.
/// \details Aggregate member order supports
///          `auto [u, s, vh, info] = truncated_svd(matrix, policy)`.
template <class LeftTensor, class SingularValueTensor, class RightAdjointTensor, class TruncationInfo>
struct TruncatedSvdResult
{
    using left_singular_vector_tensor_type = LeftTensor;
    using singular_value_tensor_type = SingularValueTensor;
    using right_singular_vector_adjoint_tensor_type = RightAdjointTensor;
    using truncation_info_type = TruncationInfo;

    LeftTensor left_singular_vectors;
    SingularValueTensor singular_values;
    RightAdjointTensor right_singular_vectors_adjoint;
    TruncationInfo truncation;
};

namespace detail
{
template <uni20::Real Real> class CompensatedNonnegativeSum {
  public:
    void add(Real value)
    {
      Real const next = sum_ + value;
      if (sum_ >= value)
        correction_ += (sum_ - next) + value;
      else
        correction_ += (value - next) + sum_;
      sum_ = next;
    }

    [[nodiscard]] Real value() const { return sum_ + correction_; }

  private:
    Real sum_{};
    Real correction_{};
};

template <uni20::Real Real> void validate_svd_truncation_policy(SvdTruncationPolicy<Real> const& policy)
{
  ERROR_IF(policy.minimum_retained_extent > policy.maximum_retained_extent,
           "SVD minimum retained extent exceeds maximum retained extent", policy.minimum_retained_extent,
           policy.maximum_retained_extent);

  auto validate_nonnegative = [](std::optional<Real> const& value, char const* name) {
    ERROR_IF(value && (!uni20::isfinite(*value) || *value < Real{}), name, "must be finite and nonnegative");
  };
  validate_nonnegative(policy.singular_value_cutoff, "SVD singular-value cutoff");
  validate_nonnegative(policy.normalized_squared_singular_value_cutoff, "SVD normalized squared singular-value cutoff");
  validate_nonnegative(policy.maximum_discarded_weight, "SVD maximum discarded weight");

  ERROR_IF(
      policy.normalized_squared_singular_value_cutoff && *policy.normalized_squared_singular_value_cutoff > Real{1},
      "SVD normalized squared singular-value cutoff exceeds one", *policy.normalized_squared_singular_value_cutoff);
  ERROR_IF(policy.maximum_discarded_weight && *policy.maximum_discarded_weight > Real{1},
           "SVD maximum discarded weight exceeds one", *policy.maximum_discarded_weight);
}

template <uni20::RankedImmediateTensorView<1> SingularValueTensor>
[[nodiscard]] auto
select_svd_truncation(SingularValueTensor const& singular_values,
                      SvdTruncationPolicy<uni20::tensor_element_t<SingularValueTensor>> const& policy)
{
  using real_type = uni20::tensor_element_t<SingularValueTensor>;
  validate_svd_truncation_policy(policy);

  std::size_t const available_rank = static_cast<std::size_t>(singular_values.extent(0));
  std::vector<real_type> scaled_squares(available_rank);
  std::vector<real_type> discarded_scaled_squares(available_rank + 1, real_type{});

  real_type scale{};
  if (available_rank > 0)
  {
    scale = singular_values[0];
    CHECK(uni20::isfinite(scale), scale);
    CHECK(scale >= real_type{}, scale);

    for (std::size_t index = 1; index < available_rank; ++index)
    {
      real_type const previous = singular_values[static_cast<uni20::index_type>(index - 1)];
      real_type const value = singular_values[static_cast<uni20::index_type>(index)];
      CHECK(uni20::isfinite(value), value);
      CHECK(value >= real_type{}, value);
      CHECK(previous >= value, previous, value);
    }
  }

  if (scale != real_type{})
  {
    CompensatedNonnegativeSum<real_type> sum;
    for (std::size_t index = available_rank; index > 0; --index)
    {
      std::size_t const value_index = index - 1;
      real_type const ratio = singular_values[static_cast<uni20::index_type>(value_index)] / scale;
      real_type const square = ratio * ratio;
      scaled_squares[value_index] = square;
      sum.add(square);
      discarded_scaled_squares[value_index] = sum.value();
    }
  }

  real_type const total_scaled_squared_norm = discarded_scaled_squares[0];
  bool const has_accuracy_criterion = policy.singular_value_cutoff || policy.normalized_squared_singular_value_cutoff ||
                                      policy.maximum_discarded_weight;
  std::size_t retained_rank = has_accuracy_criterion ? 0 : available_rank;

  if (policy.singular_value_cutoff)
  {
    std::size_t cutoff_rank = 0;
    while (cutoff_rank < available_rank &&
           singular_values[static_cast<uni20::index_type>(cutoff_rank)] >= *policy.singular_value_cutoff)
      ++cutoff_rank;
    retained_rank = std::max(retained_rank, cutoff_rank);
  }

  if (policy.normalized_squared_singular_value_cutoff)
  {
    std::size_t cutoff_rank = 0;
    if (total_scaled_squared_norm == real_type{})
    {
      if (*policy.normalized_squared_singular_value_cutoff == real_type{}) cutoff_rank = available_rank;
    }
    else
    {
      real_type const threshold = *policy.normalized_squared_singular_value_cutoff * total_scaled_squared_norm;
      while (cutoff_rank < available_rank && scaled_squares[cutoff_rank] >= threshold)
        ++cutoff_rank;
    }
    retained_rank = std::max(retained_rank, cutoff_rank);
  }

  if (policy.maximum_discarded_weight)
  {
    std::size_t error_rank = 0;
    if (total_scaled_squared_norm != real_type{})
    {
      real_type const maximum_discarded = *policy.maximum_discarded_weight * total_scaled_squared_norm;
      while (error_rank < available_rank && discarded_scaled_squares[error_rank] > maximum_discarded)
        ++error_rank;
    }
    retained_rank = std::max(retained_rank, error_rank);
  }

  retained_rank = std::max(retained_rank, std::min(policy.minimum_retained_extent, available_rank));
  retained_rank = std::min(retained_rank, std::min(policy.maximum_retained_extent, available_rank));

  real_type original_squared_norm{};
  real_type discarded_weight{};
  if (scale != real_type{})
  {
    original_squared_norm = scale * scale * total_scaled_squared_norm;
    discarded_weight = discarded_scaled_squares[retained_rank] / total_scaled_squared_norm;
  }

  return SvdTruncationInfo<real_type>{
      .available_rank = available_rank,
      .retained_rank = retained_rank,
      .original_squared_norm = original_squared_norm,
      .discarded_weight = discarded_weight,
      .smallest_retained_singular_value =
          retained_rank == 0
              ? std::nullopt
              : std::optional<real_type>{singular_values[static_cast<uni20::index_type>(retained_rank - 1)]},
      .largest_discarded_singular_value =
          retained_rank == available_rank
              ? std::nullopt
              : std::optional<real_type>{singular_values[static_cast<uni20::index_type>(retained_rank)]}};
}

template <class ExactResult>
[[nodiscard]] auto truncate_svd_result(
    ExactResult&& exact,
    SvdTruncationPolicy<
        uni20::tensor_element_t<typename std::remove_cvref_t<ExactResult>::singular_value_tensor_type>> const& policy)
{
  using exact_type = std::remove_cvref_t<ExactResult>;
  using scalar_type = uni20::tensor_element_t<typename exact_type::left_singular_vector_tensor_type>;
  using real_type = uni20::tensor_element_t<typename exact_type::singular_value_tensor_type>;

  auto truncation = select_svd_truncation(exact.singular_values, policy);
  auto const retained_rank = static_cast<uni20::index_type>(truncation.retained_rank);
  auto const rows = exact.left_singular_vectors.extent(0);
  auto const cols = exact.right_singular_vectors_adjoint.extent(1);

  uni20::Tensor<scalar_type, 2> left_singular_vectors(rows, retained_rank);
  uni20::Tensor<real_type, 1> singular_values(retained_rank);
  uni20::Tensor<scalar_type, 2> right_singular_vectors_adjoint(retained_rank, cols);

  for (uni20::index_type column = 0; column < retained_rank; ++column)
  {
    singular_values[column] = exact.singular_values[column];
    for (uni20::index_type row = 0; row < rows; ++row)
      left_singular_vectors[row, column] = exact.left_singular_vectors[row, column];
    for (uni20::index_type output_column = 0; output_column < cols; ++output_column)
      right_singular_vectors_adjoint[column, output_column] =
          exact.right_singular_vectors_adjoint[column, output_column];
  }

  return TruncatedSvdResult<decltype(left_singular_vectors), decltype(singular_values),
                            decltype(right_singular_vectors_adjoint), decltype(truncation)>{
      .left_singular_vectors = std::move(left_singular_vectors),
      .singular_values = std::move(singular_values),
      .right_singular_vectors_adjoint = std::move(right_singular_vectors_adjoint),
      .truncation = std::move(truncation)};
}
} // namespace detail

/// \brief Preserve a matrix and return a truncated reduced SVD through an explicit selector.
template <class BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
[[nodiscard]] auto
truncated_svd(BackendSelector&& selector, MatrixTensor const& matrix,
              SvdTruncationPolicy<uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>> policy = {})
{
  auto exact = svd(std::forward<BackendSelector>(selector), matrix);
  return detail::truncate_svd_result(std::move(exact), policy);
}

/// \brief Preserve a matrix and return a truncated reduced SVD.
template <uni20::RankedTensorView<2> MatrixTensor>
[[nodiscard]] auto
truncated_svd(MatrixTensor const& matrix,
              SvdTruncationPolicy<uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>> policy = {})
{
  auto exact = svd(matrix);
  return detail::truncate_svd_result(std::move(exact), policy);
}

/// \brief Consume an owning matrix and return a truncated reduced SVD through an explicit selector.
/// \details The exact SVD may reuse compatible input storage as destructive
///          workspace. Truncation returns right-sized owning factors and does
///          not promise that the transferred allocation is retained.
template <class BackendSelector, class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2> &&
           (!std::is_lvalue_reference_v<MatrixTensor>) && (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto
truncated_svd(BackendSelector&& selector, MatrixTensor&& matrix,
              SvdTruncationPolicy<uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>> policy = {})
{
  auto exact = svd(std::forward<BackendSelector>(selector), std::forward<MatrixTensor>(matrix));
  return detail::truncate_svd_result(std::move(exact), policy);
}

/// \brief Consume an owning matrix and return a truncated reduced SVD.
template <class MatrixTensor>
  requires uni20::OwningTensor<MatrixTensor> && uni20::MutableRankedImmediateTensorView<MatrixTensor, 2> &&
           (!std::is_lvalue_reference_v<MatrixTensor>) && (!std::is_const_v<std::remove_reference_t<MatrixTensor>>)
[[nodiscard]] auto
truncated_svd(MatrixTensor&& matrix,
              SvdTruncationPolicy<uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>> policy = {})
{
  auto exact = svd(std::forward<MatrixTensor>(matrix));
  return detail::truncate_svd_result(std::move(exact), policy);
}

} // namespace uni20::linalg
