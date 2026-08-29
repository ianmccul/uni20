/**
 * \file rabc_contraction.hpp
 * \ingroup tensor_network
 * \brief Dispatchable sparse R/A/B/C block contraction.
 */

#pragma once

#include <uni20/linalg/backend_selector.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/symmetry/block_tensor_concepts.hpp>
#include <uni20/tensor_network/backends/host_right_first_rabc.hpp>
#include <uni20/tensor_network/rabc_contraction_plan.hpp>
#include <uni20/tensor_network/rabc_operation.hpp>

#include <concepts>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{

template <class StoragePolicy> [[nodiscard]] auto make_rabc_backend_selector()
{
  using leaf_storage_policy = typename StoragePolicy::leaf_storage_policy;
  auto contraction_selector = linalg::select_backend_for_storage<leaf_storage_policy>(contract_op<2, 2, 1>{});
  return backend_list{tensor_network::HostRightFirstRabcBackend{std::move(contraction_selector)}};
}

} // namespace detail

/// \brief Install the host right-first R/A/B/C strategy around a block storage selector.
/// \details The supplied selector is retained for nested dense contractions.
///          Future storage-specific defaults may add CUDA, hybrid, or
///          distributed candidates ahead of this host implementation.
template <class StoragePolicy> struct backend_selector_default<tensor_network::rabc_contract_op, StoragePolicy>
{
    template <class StorageSelector>
    static auto select(tensor_network::rabc_contract_op const&, StorageSelector storage_selector)
    {
      static_cast<void>(storage_selector);
      return detail::make_rabc_backend_selector<StoragePolicy>();
    }
};

} // namespace uni20::linalg

namespace uni20::tensor_network
{
namespace detail
{

template <class Output, class B> [[nodiscard]] constexpr bool is_obvious_rabc_alias(Output& output, B const& b) noexcept
{
  if constexpr (std::same_as<std::remove_cvref_t<Output>, std::remove_cvref_t<B>>)
  {
    return static_cast<void const*>(std::addressof(output)) == static_cast<void const*>(std::addressof(b));
  }
  return false;
}

} // namespace detail

/// \brief Prepare the single host right-first backend selected for fixed R/A/B/C operands.
/// \details Preparation validates the fixed block structures, derives the
///          backend-specific schedule, and allocates reusable intermediate
///          storage. This overload intentionally accepts the currently
///          supported one-entry prepared selector; future backend lists must
///          define how preparation chooses and retains one accepted backend.
/// \param selector Selected host R/A/B/C backend and nested dense selector.
/// \param output Prototype fixed output structure.
/// \param plan Execution-neutral coefficient plan transferred into the executor.
/// \param a Prototype left block family.
/// \param b Prototype center block family.
/// \param c Prototype right block family.
/// \pre The plan's coordinate keys were constructed for the symmetry, domain,
///      and codomain semantics of these four operand families.
/// \return Prepared host executor owning the plan, schedule, and workspace.
template <class ContractionSelector, detail::HostWritableRabcTensor Output, RabcPlan Plan,
          detail::HostReadableRabcTensor A, detail::HostReadableRabcTensor B, detail::HostReadableRabcTensor C>
  requires detail::CompatibleHostRabcOperands<Output, Plan, A, B, C>
[[nodiscard]] auto prepare_rabc_contract(linalg::backend_list<HostRightFirstRabcBackend<ContractionSelector>> selector,
                                         Output const& output, Plan plan, A const& a, B const& b, C const& c)
{
  auto backend = std::move(std::get<0>(selector.entries));
  return PreparedHostRightFirstRabcContraction(std::move(backend.contraction_selector), output, std::move(plan), a, b,
                                               c);
}

/// \brief Select and prepare the R/A/B/C backend for one fixed block structure.
/// \param output Prototype fixed output structure and storage policy.
/// \param plan Execution-neutral coefficient plan transferred into the executor.
/// \param a Prototype left block family.
/// \param b Prototype center block family.
/// \param c Prototype right block family.
/// \pre The plan's coordinate keys were constructed for the symmetry, domain,
///      and codomain semantics of these four operand families.
/// \return Prepared executor selected from the output storage policy.
template <detail::HostWritableRabcTensor Output, RabcPlan Plan, detail::HostReadableRabcTensor A,
          detail::HostReadableRabcTensor B, detail::HostReadableRabcTensor C>
  requires detail::CompatibleHostRabcOperands<Output, Plan, A, B, C>
[[nodiscard]] auto prepare_rabc_contract(Output const& output, Plan plan, A const& a, B const& b, C const& c)
{
  auto selector = linalg::select_backend(rabc_contract_op{}, output);
  return prepare_rabc_contract(std::move(selector), output, std::move(plan), a, b, c);
}

/// \brief Execute a fixed-output sparse R/A/B/C contraction with an explicit backend selector.
/// \details Computes `R_r = sum(f(r,a,b,c) A_a B_b transpose(C_c))`.
///          The sparse plan contains only mathematical coefficients and block
///          keys. The selected backend resolves keys to storage bindings and
///          chooses contraction order, intermediate reuse, placement, and
///          communication. Stored output blocks with no contributing term are
///          overwritten with zero.
/// \pre Output numerical storage does not overlap any input family.
/// \pre The plan's coordinate keys were constructed for the symmetry, domain,
///      and codomain semantics of the supplied operand families.
/// \tparam BackendSelector Explicit R/A/B/C backend selector.
/// \param selector Ordered backend selector.
/// \param output Existing fixed-structure R block family.
/// \param plan Immutable sparse coefficient tensor `f`.
/// \param a Left environment block family.
/// \param b Input center block family.
/// \param c Right environment block family, stored before transposition.
template <linalg::KernelBackendSelector BackendSelector, MutableBlockTensorView Output, RabcPlan Plan,
          BlockTensorView A, BlockTensorView B, BlockTensorView C>
void rabc_contract(BackendSelector&& selector, Output& output, Plan const& plan, A const& a, B const& b, C const& c)
{
  if (detail::is_obvious_rabc_alias(output, b))
    throw std::invalid_argument("R/A/B/C contraction output must not alias its input center");
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), rabc_contract_op{}, output, plan, a, b, c);
}

/// \brief Execute a sparse R/A/B/C contraction using the output storage policy.
/// \details Backend selection occurs while output storage and execution policy
///          remain available. Input placement remains visible to the selected
///          backend through the fixed block-tensor views.
/// \pre The plan's coordinate keys were constructed for the symmetry, domain,
///      and codomain semantics of the supplied operand families.
template <MutableBlockTensorView Output, RabcPlan Plan, BlockTensorView A, BlockTensorView B, BlockTensorView C>
void rabc_contract(Output& output, Plan const& plan, A const& a, B const& b, C const& c)
{
  auto selector = linalg::select_backend(rabc_contract_op{}, output);
  rabc_contract(std::move(selector), output, plan, a, b, c);
}

} // namespace uni20::tensor_network
