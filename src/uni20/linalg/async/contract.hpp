#pragma once

/**
 * \file contract.hpp
 * \ingroup linalg
 * \brief Async Tensor wrapper for fixed-output pairwise contraction.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async/concepts.hpp>
#include <uni20/linalg/async/detail/output.hpp>
#include <uni20/linalg/async/dispatch.hpp>
#include <uni20/linalg/ops/contract.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class BackendSelector, uni20::AsyncTensorOutput OutputTensor, uni20::TensorView LhsTensor,
          uni20::TensorView RhsTensor, class AlphaAwaiter, class BetaAwaiter, std::size_t LhsRank, std::size_t RhsRank,
          std::size_t ContractedRank>
async::AsyncTask co_contract(BackendSelector const selector, async::WriteBuffer<OutputTensor> output,
                             async::ReadBuffer<LhsTensor> lhs, async::ReadBuffer<RhsTensor> rhs, AlphaAwaiter alpha,
                             BetaAwaiter beta, ContractionAxes<LhsRank, RhsRank, ContractedRank> axes)
{
  auto output_awaiter = uni20::detail::mutable_async_tensor_awaiter(output);
  auto awaited = co_await async::all(output_awaiter, lhs, rhs, alpha, beta);
  decltype(auto) output_value = uni20::detail::mutable_async_tensor_value<OutputTensor>(std::get<0>(awaited));
  auto const& lhs_value = std::get<1>(awaited);
  auto const& rhs_value = std::get<2>(awaited);
  using scalar_type = uni20::tensor_element_t<OutputTensor>;
  scalar_type const alpha_value = static_cast<scalar_type>(std::get<3>(awaited));
  scalar_type const beta_value = static_cast<scalar_type>(std::get<4>(awaited));

  auto const required_extents = contraction_output_extents(lhs_value, rhs_value, axes);
  uni20::require_output(output_value, required_extents);
  auto output_descriptor = uni20::mdspec_of(output_value);
  auto lhs_descriptor = uni20::mdspec_of(lhs_value);
  auto rhs_descriptor = uni20::mdspec_of(rhs_value);
  auto operation = contract_op<LhsRank, RhsRank, ContractedRank>{.axes = std::move(axes)};
  co_await co_dispatch_kernel(selector, std::move(operation), output_descriptor, alpha_value, lhs_descriptor,
                              rhs_descriptor, beta_value);
  co_return;
}

template <class BackendSelector, uni20::AsyncTensorOutput OutputTensor, uni20::TensorView LhsTensor,
          uni20::TensorView RhsTensor, class AlphaAwaiter, class BetaAwaiter, std::size_t LhsRank, std::size_t RhsRank,
          std::size_t ContractedRank>
void schedule_async_contract(BackendSelector selector, async::Async<OutputTensor>& output,
                             async::Async<LhsTensor> const& lhs, async::Async<RhsTensor> const& rhs, AlphaAwaiter alpha,
                             BetaAwaiter beta, ContractionAxes<LhsRank, RhsRank, ContractedRank> axes)
{
  auto const* const output_queue = std::addressof(output.queue());
  ERROR_IF(output_queue == std::addressof(lhs.queue()) || output_queue == std::addressof(rhs.queue()),
           "async contraction output must not share an epoch queue with an input");
  auto task = co_contract(std::move(selector), output.write(), lhs.read(), rhs.read(), std::move(alpha),
                          std::move(beta), std::move(axes));
  task.debug_name("contract");
  async::schedule(std::move(task));
}
} // namespace detail

/// \brief Schedule a fixed-output pairwise contraction through an explicit selector.
/// \details Tensor operands are asynchronous; `alpha` and `beta` may be immediate or Async scalars.
/// \pre The output is constructed, has a distinct epoch queue from both inputs, and has compatible shape.
template <KernelBackendSelector BackendSelector, uni20::AsyncTensorOutput OutputTensor, uni20::TensorView LhsTensor,
          uni20::TensorView RhsTensor, class Alpha, class Beta, std::size_t LhsRank, std::size_t RhsRank,
          std::size_t ContractedRank>
  requires detail::CompatibleContractionTensors<OutputTensor, LhsTensor, RhsTensor> &&
           uni20::MutableRankedTensorView<OutputTensor,
                                          ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> &&
           uni20::RankedTensorView<LhsTensor, LhsRank> && uni20::RankedTensorView<RhsTensor, RhsRank> &&
           AsyncOperationScalar<Alpha, uni20::tensor_element_t<OutputTensor>> &&
           AsyncOperationScalar<Beta, uni20::tensor_element_t<OutputTensor>>
void contract(BackendSelector selector, async::Async<OutputTensor>& output, Alpha&& alpha,
              async::Async<LhsTensor> const& lhs, async::Async<RhsTensor> const& rhs,
              ContractionAxes<LhsRank, RhsRank, ContractedRank> axes, Beta&& beta)
{
  detail::schedule_async_contract(std::move(selector), output, lhs, rhs, async::read(std::forward<Alpha>(alpha)),
                                  async::read(std::forward<Beta>(beta)), std::move(axes));
}

/// \brief Schedule a fixed-output pairwise contraction using static Tensor policy.
template <uni20::AsyncTensorOutput OutputTensor, uni20::TensorView LhsTensor, uni20::TensorView RhsTensor, class Alpha,
          class Beta, std::size_t LhsRank, std::size_t RhsRank, std::size_t ContractedRank>
  requires detail::CompatibleContractionTensors<OutputTensor, LhsTensor, RhsTensor> &&
           uni20::MutableRankedTensorView<OutputTensor,
                                          ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> &&
           uni20::RankedTensorView<LhsTensor, LhsRank> && uni20::RankedTensorView<RhsTensor, RhsRank> &&
           AsyncOperationScalar<Alpha, uni20::tensor_element_t<OutputTensor>> &&
           AsyncOperationScalar<Beta, uni20::tensor_element_t<OutputTensor>>
void contract(async::Async<OutputTensor>& output, Alpha&& alpha, async::Async<LhsTensor> const& lhs,
              async::Async<RhsTensor> const& rhs, ContractionAxes<LhsRank, RhsRank, ContractedRank> axes, Beta&& beta)
{
  auto operation = contract_op<LhsRank, RhsRank, ContractedRank>{.axes = axes};
  auto selector = select_backend_for<OutputTensor, LhsTensor, RhsTensor>(operation);
  contract(std::move(selector), output, std::forward<Alpha>(alpha), lhs, rhs, std::move(axes),
           std::forward<Beta>(beta));
}

/// \brief Schedule a fixed-output pairwise contraction from raw axis pairs through an explicit selector.
template <KernelBackendSelector BackendSelector, uni20::AsyncTensorOutput OutputTensor, uni20::TensorView LhsTensor,
          uni20::TensorView RhsTensor, class Alpha, class Beta, std::size_t ContractedRank>
  requires detail::CompatibleContractionTensors<OutputTensor, LhsTensor, RhsTensor> &&
           (ContractedRank <= uni20::tensor_mdspec_t<LhsTensor>::rank()) &&
           (ContractedRank <= uni20::tensor_mdspec_t<RhsTensor>::rank()) &&
           (uni20::tensor_mdspec_t<OutputTensor>::rank() == uni20::tensor_mdspec_t<LhsTensor>::rank() +
                                                                uni20::tensor_mdspec_t<RhsTensor>::rank() -
                                                                2 * ContractedRank) &&
           AsyncOperationScalar<Alpha, uni20::tensor_element_t<OutputTensor>> &&
           AsyncOperationScalar<Beta, uni20::tensor_element_t<OutputTensor>>
void contract(BackendSelector selector, async::Async<OutputTensor>& output, Alpha&& alpha,
              async::Async<LhsTensor> const& lhs, async::Async<RhsTensor> const& rhs,
              std::array<std::pair<std::size_t, std::size_t>, ContractedRank> requested_axes, Beta&& beta)
{
  constexpr std::size_t lhs_rank = uni20::tensor_mdspec_t<LhsTensor>::rank();
  constexpr std::size_t rhs_rank = uni20::tensor_mdspec_t<RhsTensor>::rank();
  auto axes = make_contraction_axes<lhs_rank, rhs_rank>(std::move(requested_axes));
  contract(std::move(selector), output, std::forward<Alpha>(alpha), lhs, rhs, std::move(axes),
           std::forward<Beta>(beta));
}

/// \brief Schedule a fixed-output pairwise contraction from raw axis pairs using static Tensor policy.
template <uni20::AsyncTensorOutput OutputTensor, uni20::TensorView LhsTensor, uni20::TensorView RhsTensor, class Alpha,
          class Beta, std::size_t ContractedRank>
  requires detail::CompatibleContractionTensors<OutputTensor, LhsTensor, RhsTensor> &&
           (ContractedRank <= uni20::tensor_mdspec_t<LhsTensor>::rank()) &&
           (ContractedRank <= uni20::tensor_mdspec_t<RhsTensor>::rank()) &&
           (uni20::tensor_mdspec_t<OutputTensor>::rank() == uni20::tensor_mdspec_t<LhsTensor>::rank() +
                                                                uni20::tensor_mdspec_t<RhsTensor>::rank() -
                                                                2 * ContractedRank) &&
           AsyncOperationScalar<Alpha, uni20::tensor_element_t<OutputTensor>> &&
           AsyncOperationScalar<Beta, uni20::tensor_element_t<OutputTensor>>
void contract(async::Async<OutputTensor>& output, Alpha&& alpha, async::Async<LhsTensor> const& lhs,
              async::Async<RhsTensor> const& rhs,
              std::array<std::pair<std::size_t, std::size_t>, ContractedRank> requested_axes, Beta&& beta)
{
  constexpr std::size_t lhs_rank = uni20::tensor_mdspec_t<LhsTensor>::rank();
  constexpr std::size_t rhs_rank = uni20::tensor_mdspec_t<RhsTensor>::rank();
  auto axes = make_contraction_axes<lhs_rank, rhs_rank>(std::move(requested_axes));
  contract(output, std::forward<Alpha>(alpha), lhs, rhs, std::move(axes), std::forward<Beta>(beta));
}

} // namespace uni20::linalg
