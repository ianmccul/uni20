#pragma once

/**
 * \file contract.hpp
 * \ingroup linalg
 * \brief Fixed-output pairwise Tensor contraction front end.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/backends/cpu/contract.hpp>
#include <uni20/linalg/backends/direct_gemm/contract.hpp>
#include <uni20/linalg/contraction_axes.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/output.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{

template <class OutputTensor, class InputTensor>
[[nodiscard]] constexpr bool is_obvious_contraction_alias(OutputTensor& output, InputTensor const& input) noexcept
{
  if constexpr (std::same_as<std::remove_cvref_t<OutputTensor>, std::remove_cvref_t<InputTensor>>)
  {
    return static_cast<void const*>(std::addressof(output)) == static_cast<void const*>(std::addressof(input));
  }
  return false;
}

template <class OutputTensor, class LhsTensor, class RhsTensor>
concept CompatibleContractionTensors =
    uni20::MutableTensorView<OutputTensor> && uni20::TensorView<LhsTensor> && uni20::TensorView<RhsTensor> &&
    std::same_as<uni20::tensor_element_t<OutputTensor>, uni20::tensor_element_t<LhsTensor>> &&
    std::same_as<uni20::tensor_element_t<OutputTensor>, uni20::tensor_element_t<RhsTensor>> &&
    uni20::Scalar<uni20::tensor_element_t<OutputTensor>>;

template <class OutputTensor, class LhsTensor, class RhsTensor>
void validate_contraction_aliasing(OutputTensor& output, LhsTensor const& lhs, RhsTensor const& rhs)
{
  ERROR_IF(is_obvious_contraction_alias(output, lhs) || is_obvious_contraction_alias(output, rhs),
           "tensor contraction output must not alias an input tensor");
}

} // namespace detail

/// \brief Contract two tensors into an existing fixed-storage output.
/// \details Computes `output = alpha * contract(lhs, rhs) + beta * output`.
///          Surviving left axes precede surviving right axes in their original
///          order. Backend selection occurs before operands are normalized to
///          mdspec values.
/// \pre Output storage does not overlap either input, beyond the cheap
///      same-object aliases rejected by this front end.
template <class BackendSelector, class OutputTensor, class LhsTensor, class RhsTensor, std::size_t LhsRank,
          std::size_t RhsRank, std::size_t ContractedRank>
  requires detail::CompatibleContractionTensors<OutputTensor, LhsTensor, RhsTensor> &&
           uni20::MutableRankedTensorView<OutputTensor,
                                          ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> &&
           uni20::RankedTensorView<LhsTensor, LhsRank> && uni20::RankedTensorView<RhsTensor, RhsRank>
void contract(BackendSelector&& selector, OutputTensor&& output, uni20::tensor_element_t<OutputTensor> alpha,
              LhsTensor const& lhs, RhsTensor const& rhs, ContractionAxes<LhsRank, RhsRank, ContractedRank> axes,
              uni20::tensor_element_t<OutputTensor> beta)
{
  detail::validate_contraction_aliasing(output, lhs, rhs);
  auto const required_extents = contraction_output_extents(lhs, rhs, axes);
  uni20::require_output(output, required_extents);

  auto output_descriptor = uni20::mdspec_of(output);
  auto lhs_descriptor = uni20::mdspec_of(lhs);
  auto rhs_descriptor = uni20::mdspec_of(rhs);
  auto operation = contract_op<LhsRank, RhsRank, ContractedRank>{.axes = std::move(axes)};
  dispatch_kernel(std::forward<BackendSelector>(selector), std::move(operation), output_descriptor, alpha,
                  lhs_descriptor, rhs_descriptor, beta);
}

/// \brief Contract two tensors using their default backend selector.
template <class OutputTensor, class LhsTensor, class RhsTensor, std::size_t LhsRank, std::size_t RhsRank,
          std::size_t ContractedRank>
  requires detail::CompatibleContractionTensors<OutputTensor, LhsTensor, RhsTensor> &&
           uni20::MutableRankedTensorView<OutputTensor,
                                          ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> &&
           uni20::RankedTensorView<LhsTensor, LhsRank> && uni20::RankedTensorView<RhsTensor, RhsRank>
void contract(OutputTensor&& output, uni20::tensor_element_t<OutputTensor> alpha, LhsTensor const& lhs,
              RhsTensor const& rhs, ContractionAxes<LhsRank, RhsRank, ContractedRank> axes,
              uni20::tensor_element_t<OutputTensor> beta)
{
  auto operation = contract_op<LhsRank, RhsRank, ContractedRank>{.axes = axes};
  auto selector = select_backend(operation, output, lhs, rhs);
  contract(std::move(selector), std::forward<OutputTensor>(output), alpha, lhs, rhs, std::move(axes), beta);
}

/// \brief Contract two tensors using an explicit selector and raw axis pairs.
template <class BackendSelector, class OutputTensor, class LhsTensor, class RhsTensor, std::size_t ContractedRank>
  requires detail::CompatibleContractionTensors<OutputTensor, LhsTensor, RhsTensor> &&
           (ContractedRank <= uni20::tensor_mdspec_t<LhsTensor>::rank()) &&
           (ContractedRank <= uni20::tensor_mdspec_t<RhsTensor>::rank()) &&
           (uni20::tensor_mdspec_t<OutputTensor>::rank() ==
            uni20::tensor_mdspec_t<LhsTensor>::rank() + uni20::tensor_mdspec_t<RhsTensor>::rank() - 2 * ContractedRank)
void contract(BackendSelector&& selector, OutputTensor&& output, uni20::tensor_element_t<OutputTensor> alpha,
              LhsTensor const& lhs, RhsTensor const& rhs,
              std::array<std::pair<std::size_t, std::size_t>, ContractedRank> requested_axes,
              uni20::tensor_element_t<OutputTensor> beta)
{
  constexpr std::size_t lhs_rank = uni20::tensor_mdspec_t<LhsTensor>::rank();
  constexpr std::size_t rhs_rank = uni20::tensor_mdspec_t<RhsTensor>::rank();
  auto axes = make_contraction_axes<lhs_rank, rhs_rank>(std::move(requested_axes));
  contract(std::forward<BackendSelector>(selector), std::forward<OutputTensor>(output), alpha, lhs, rhs,
           std::move(axes), beta);
}

/// \brief Contract two tensors using storage policy and raw axis pairs.
template <class OutputTensor, class LhsTensor, class RhsTensor, std::size_t ContractedRank>
  requires detail::CompatibleContractionTensors<OutputTensor, LhsTensor, RhsTensor> &&
           (ContractedRank <= uni20::tensor_mdspec_t<LhsTensor>::rank()) &&
           (ContractedRank <= uni20::tensor_mdspec_t<RhsTensor>::rank()) &&
           (uni20::tensor_mdspec_t<OutputTensor>::rank() ==
            uni20::tensor_mdspec_t<LhsTensor>::rank() + uni20::tensor_mdspec_t<RhsTensor>::rank() - 2 * ContractedRank)
void contract(OutputTensor&& output, uni20::tensor_element_t<OutputTensor> alpha, LhsTensor const& lhs,
              RhsTensor const& rhs, std::array<std::pair<std::size_t, std::size_t>, ContractedRank> requested_axes,
              uni20::tensor_element_t<OutputTensor> beta)
{
  constexpr std::size_t lhs_rank = uni20::tensor_mdspec_t<LhsTensor>::rank();
  constexpr std::size_t rhs_rank = uni20::tensor_mdspec_t<RhsTensor>::rank();
  auto axes = make_contraction_axes<lhs_rank, rhs_rank>(std::move(requested_axes));
  contract(std::forward<OutputTensor>(output), alpha, lhs, rhs, std::move(axes), beta);
}

} // namespace uni20::linalg
