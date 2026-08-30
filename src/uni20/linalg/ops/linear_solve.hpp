#pragma once

/**
 * \file linear_solve.hpp
 * \ingroup linalg
 * \brief Destructive-workspace and value-producing dense general solves.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/backends/cpu/linear_solve.hpp>
#include <uni20/linalg/backends/lapack/linear_solve.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>

#include <concepts>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class CoefficientTensor, class RhsTensor>
concept CompatibleLinearSolveTensors =
    uni20::MutableRankedTensorView<CoefficientTensor, 2> && uni20::MutableRankedTensorView<RhsTensor, 2> &&
    uni20::RealOrComplex<uni20::tensor_element_t<CoefficientTensor>> &&
    std::same_as<uni20::tensor_element_t<CoefficientTensor>, uni20::tensor_element_t<RhsTensor>>;

template <uni20::RankedTensorView<2> CoefficientTensor, uni20::RankedTensorView<2> RhsTensor>
void require_linear_solve_shape(CoefficientTensor const& coefficients, RhsTensor const& right_hand_sides)
{
  ERROR_IF(coefficients.extent(0) != coefficients.extent(1), "solve requires a square coefficient matrix");
  ERROR_IF(coefficients.extent(0) != right_hand_sides.extent(0),
           "solve coefficient and right-hand-side row counts do not agree");
}
} // namespace detail

/// \brief Solve a general system in destructive coefficient and RHS workspaces.
/// \details On return, `right_hand_sides` contains the solution. The
///          coefficient workspace is overwritten with backend factorization
///          data. A provider-reported singular matrix is a terminal error.
/// \pre The two workspaces do not overlap.
template <KernelBackendSelector BackendSelector, class CoefficientTensor, class RhsTensor>
  requires detail::CompatibleLinearSolveTensors<CoefficientTensor, RhsTensor>
void solve_inplace(BackendSelector&& selector, CoefficientTensor&& coefficients, RhsTensor&& right_hand_sides)
{
  detail::require_linear_solve_shape(coefficients, right_hand_sides);
  auto coefficient_descriptor = uni20::mdspec_of(coefficients);
  auto rhs_descriptor = uni20::mdspec_of(right_hand_sides);
  dispatch_kernel(std::forward<BackendSelector>(selector), linear_solve_op{}, coefficient_descriptor, rhs_descriptor);
}

/// \brief Solve a general system in destructive workspaces using storage policy.
template <class CoefficientTensor, class RhsTensor>
  requires detail::CompatibleLinearSolveTensors<CoefficientTensor, RhsTensor>
void solve_inplace(CoefficientTensor&& coefficients, RhsTensor&& right_hand_sides)
{
  detail::require_linear_solve_shape(coefficients, right_hand_sides);
  auto selector = select_backend(linear_solve_op{}, coefficients, right_hand_sides);
  solve_inplace(selector, std::forward<CoefficientTensor>(coefficients), std::forward<RhsTensor>(right_hand_sides));
}

/// \brief Preserve inputs and return the dense solution through an explicit selector.
/// \details Coefficients and right-hand sides are materialized into owning
///          column-major host work matrices before destructive dispatch.
template <KernelBackendSelector BackendSelector, uni20::RankedTensorView<2> CoefficientTensor,
          uni20::RankedTensorView<2> RhsTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<CoefficientTensor>> &&
           std::same_as<uni20::tensor_element_t<CoefficientTensor>, uni20::tensor_element_t<RhsTensor>>
[[nodiscard]] auto solve(BackendSelector&& selector, CoefficientTensor const& coefficients,
                         RhsTensor const& right_hand_sides)
{
  detail::require_linear_solve_shape(coefficients, right_hand_sides);
  auto coefficient_work = uni20::make_tensor<uni20::ColumnMajor>(coefficients);
  auto solution = uni20::make_tensor<uni20::ColumnMajor>(right_hand_sides);
  solve_inplace(std::forward<BackendSelector>(selector), coefficient_work, solution);
  return solution;
}

/// \brief Preserve inputs and return an owning column-major host solution.
template <uni20::RankedTensorView<2> CoefficientTensor, uni20::RankedTensorView<2> RhsTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<CoefficientTensor>> &&
           std::same_as<uni20::tensor_element_t<CoefficientTensor>, uni20::tensor_element_t<RhsTensor>>
[[nodiscard]] auto solve(CoefficientTensor const& coefficients, RhsTensor const& right_hand_sides)
{
  detail::require_linear_solve_shape(coefficients, right_hand_sides);
  auto coefficient_work = uni20::make_tensor<uni20::ColumnMajor>(coefficients);
  auto solution = uni20::make_tensor<uni20::ColumnMajor>(right_hand_sides);
  auto selector = select_backend(linear_solve_op{}, coefficient_work, solution);
  solve_inplace(selector, coefficient_work, solution);
  return solution;
}

} // namespace uni20::linalg
