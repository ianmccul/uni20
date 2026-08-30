#pragma once

/**
 * \file linear_solve.hpp
 * \ingroup linalg
 * \brief Async Tensor wrappers for dense general linear solves.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async/detail/output.hpp>
#include <uni20/linalg/ops/linear_solve.hpp>

#include <concepts>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class CoefficientTensor, class RhsTensor>
using preserving_solve_result_t = std::remove_cvref_t<decltype(uni20::linalg::solve(
    std::declval<CoefficientTensor const&>(), std::declval<RhsTensor const&>()))>;

template <class CoefficientTensor, class RhsTensor> [[nodiscard]] constexpr auto select_async_solve_backend()
{
  using work_type = preserving_solve_result_t<CoefficientTensor, RhsTensor>;
  return select_backend_for<work_type, work_type>(linear_solve_op{});
}

template <class BackendSelector, class ResultTensor, uni20::RankedTensorView<2> CoefficientTensor,
          uni20::RankedTensorView<2> RhsTensor>
async::AsyncTask co_preserving_solve(BackendSelector const selector, async::WriteBuffer<ResultTensor> output,
                                     async::ReadBuffer<CoefficientTensor> coefficients,
                                     async::ReadBuffer<RhsTensor> right_hand_sides)
{
  auto output_storage = output.storage();
  auto awaited = co_await async::all(output_storage, coefficients, right_hand_sides);
  auto& storage = std::get<0>(awaited);
  auto const& coefficient_value = std::get<1>(awaited);
  auto const& rhs_value = std::get<2>(awaited);

  auto result = uni20::linalg::solve(selector, coefficient_value, rhs_value);
  storage.emplace(std::move(result));
  co_return;
}

template <class BackendSelector, uni20::AsyncTensorOutput CoefficientTensor, uni20::AsyncTensorOutput RhsTensor>
async::AsyncTask co_solve_inplace(BackendSelector const selector, async::WriteBuffer<CoefficientTensor> coefficients,
                                  async::WriteBuffer<RhsTensor> right_hand_sides)
{
  auto coefficient_awaiter = uni20::detail::mutable_async_tensor_awaiter(coefficients);
  auto rhs_awaiter = uni20::detail::mutable_async_tensor_awaiter(right_hand_sides);
  auto awaited = co_await async::all(coefficient_awaiter, rhs_awaiter);
  decltype(auto) coefficient_value = uni20::detail::mutable_async_tensor_value<CoefficientTensor>(std::get<0>(awaited));
  decltype(auto) rhs_value = uni20::detail::mutable_async_tensor_value<RhsTensor>(std::get<1>(awaited));

  uni20::linalg::solve_inplace(selector, coefficient_value, rhs_value);
  co_return;
}

template <class BackendSelector, uni20::RankedTensorView<2> CoefficientTensor, uni20::RankedTensorView<2> RhsTensor>
[[nodiscard]] auto schedule_preserving_solve(BackendSelector selector,
                                             async::Async<CoefficientTensor> const& coefficients,
                                             async::Async<RhsTensor> const& right_hand_sides)
{
  using result_type = preserving_solve_result_t<CoefficientTensor, RhsTensor>;
  async::Async<result_type> output;
  output.debug_name("solve.solution");
  auto task = co_preserving_solve(std::move(selector), output.write(), coefficients.read(), right_hand_sides.read());
  task.debug_name("solve");
  async::schedule(std::move(task));
  return output;
}

template <class BackendSelector, uni20::AsyncTensorOutput CoefficientTensor, uni20::AsyncTensorOutput RhsTensor>
void schedule_solve_inplace(BackendSelector selector, async::Async<CoefficientTensor>& coefficients,
                            async::Async<RhsTensor>& right_hand_sides)
{
  ERROR_IF(std::addressof(coefficients.queue()) == std::addressof(right_hand_sides.queue()),
           "async solve workspaces must not share an epoch queue");
  auto task = co_solve_inplace(std::move(selector), coefficients.write(), right_hand_sides.write());
  task.debug_name("solve_inplace");
  async::schedule(std::move(task));
}
} // namespace detail

/// \brief Schedule a preserving dense general solve through an explicit selector.
/// \details Both inputs remain readable. The result is an independent owning
///          column-major host matrix whose failure state is published by the task.
template <KernelBackendSelector BackendSelector, uni20::RankedTensorView<2> CoefficientTensor,
          uni20::RankedTensorView<2> RhsTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<CoefficientTensor>> &&
           std::same_as<uni20::tensor_element_t<CoefficientTensor>, uni20::tensor_element_t<RhsTensor>>
[[nodiscard]] auto solve(BackendSelector selector, async::Async<CoefficientTensor> const& coefficients,
                         async::Async<RhsTensor> const& right_hand_sides)
{
  return detail::schedule_preserving_solve(std::move(selector), coefficients, right_hand_sides);
}

/// \brief Schedule a preserving dense general solve using static Tensor policy.
template <uni20::RankedTensorView<2> CoefficientTensor, uni20::RankedTensorView<2> RhsTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<CoefficientTensor>> &&
           std::same_as<uni20::tensor_element_t<CoefficientTensor>, uni20::tensor_element_t<RhsTensor>>
[[nodiscard]] auto solve(async::Async<CoefficientTensor> const& coefficients,
                         async::Async<RhsTensor> const& right_hand_sides)
{
  auto selector = detail::select_async_solve_backend<CoefficientTensor, RhsTensor>();
  return detail::schedule_preserving_solve(std::move(selector), coefficients, right_hand_sides);
}

/// \brief Schedule a destructive dense general solve through an explicit selector.
/// \details On success, `right_hand_sides` contains the solution and
///          `coefficients` contains backend factorization data. Both writer
///          epochs receive any task failure.
/// \pre The two async workspaces have distinct epoch queues and nonoverlapping storage.
template <KernelBackendSelector BackendSelector, uni20::AsyncTensorOutput CoefficientTensor,
          uni20::AsyncTensorOutput RhsTensor>
  requires detail::CompatibleLinearSolveTensors<CoefficientTensor, RhsTensor>
void solve_inplace(BackendSelector selector, async::Async<CoefficientTensor>& coefficients,
                   async::Async<RhsTensor>& right_hand_sides)
{
  detail::schedule_solve_inplace(std::move(selector), coefficients, right_hand_sides);
}

/// \brief Schedule a destructive dense general solve using static Tensor policy.
/// \pre The two async workspaces have distinct epoch queues and nonoverlapping storage.
template <uni20::AsyncTensorOutput CoefficientTensor, uni20::AsyncTensorOutput RhsTensor>
  requires detail::CompatibleLinearSolveTensors<CoefficientTensor, RhsTensor>
void solve_inplace(async::Async<CoefficientTensor>& coefficients, async::Async<RhsTensor>& right_hand_sides)
{
  auto selector = select_backend_for<CoefficientTensor, RhsTensor>(linear_solve_op{});
  detail::schedule_solve_inplace(std::move(selector), coefficients, right_hand_sides);
}

} // namespace uni20::linalg
