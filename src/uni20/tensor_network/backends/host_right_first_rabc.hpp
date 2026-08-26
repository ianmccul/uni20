/**
 * \file host_right_first_rabc.hpp
 * \ingroup tensor_network
 * \brief Host right-first backend for sparse R/A/B/C contractions.
 */

#pragma once

#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/elementwise_functions.hpp>
#include <uni20/linalg/ops/contract.hpp>
#include <uni20/symmetry/block_tensor_concepts.hpp>
#include <uni20/symmetry/block_tensor_storage.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/tensor.hpp>
#include <uni20/tensor/transform.hpp>
#include <uni20/tensor_network/rabc_contraction_plan.hpp>
#include <uni20/tensor_network/rabc_operation.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::tensor_network
{

/// \brief Host executor that reuses right-first `(B_b,C_c)` intermediates.
/// \details The retained selector performs each dense pairwise contraction.
///          The sparse logical plan remains order-neutral; this backend derives
///          its right-first groups for each invocation.
/// \tparam ContractionSelector Dense contraction backend selector.
template <linalg::KernelBackendSelector ContractionSelector> struct HostRightFirstRabcBackend
{
    static constexpr std::string_view name = "host_right_first_rabc";

    ContractionSelector contraction_selector;
};

template <class ContractionSelector>
HostRightFirstRabcBackend(ContractionSelector) -> HostRightFirstRabcBackend<std::remove_cvref_t<ContractionSelector>>;

namespace detail
{

template <class Tensor>
concept HostReadableRabcTensor =
    ImmediateBlockTensorView<Tensor> && (block_tensor_type_t<Tensor>::dense_block_order() == 2) &&
    HostReadableTensor<block_tensor_const_block_t<Tensor>>;

template <class Tensor>
concept HostWritableRabcTensor =
    MutableImmediateBlockTensorView<Tensor> && (block_tensor_type_t<Tensor>::dense_block_order() == 2) &&
    HostWritableTensor<block_tensor_mutable_block_t<Tensor>>;

template <class Output, class Plan, class A, class B, class C>
concept CompatibleHostRabcOperands =
    HostWritableRabcTensor<Output> && HostReadableRabcTensor<A> && HostReadableRabcTensor<B> &&
    HostReadableRabcTensor<C> &&
    std::same_as<typename std::remove_cvref_t<Plan>::scalar_type, block_tensor_value_t<Output>> &&
    std::same_as<block_tensor_value_t<Output>, block_tensor_value_t<A>> &&
    std::same_as<block_tensor_value_t<Output>, block_tensor_value_t<B>> &&
    std::same_as<block_tensor_value_t<Output>, block_tensor_value_t<C>>;

template <class Tensor>
using mutable_rabc_mdspec_t = decltype(mdspec_of(std::declval<block_tensor_mutable_block_t<Tensor>&>()));

template <class Tensor>
using const_rabc_mdspec_t = decltype(mdspec_of(std::declval<block_tensor_const_block_t<Tensor> const&>()));

template <class ContractionSelector, class Output, class Scalar, class A, class B, class C>
consteval bool host_right_first_types_compatible()
{
  using intermediate_type = ColumnMajorTensor<Scalar, 2>;
  using intermediate_mutable_mdspec = decltype(mdspec_of(std::declval<intermediate_type&>()));
  using intermediate_const_mdspec = decltype(mdspec_of(std::declval<intermediate_type const&>()));
  using output_mdspec = mutable_rabc_mdspec_t<Output>;
  using a_mdspec = const_rabc_mdspec_t<A>;
  using b_mdspec = const_rabc_mdspec_t<B>;
  using c_mdspec = const_rabc_mdspec_t<C>;

  constexpr bool right_product_is_total =
      linalg::probe_dispatch_kernel_types<ContractionSelector const&, linalg::contract_op<2, 2, 1>,
                                          intermediate_mutable_mdspec&, Scalar&, b_mdspec&, c_mdspec&, Scalar&>() ==
      linalg::KernelTypeAcceptance::yes;
  constexpr bool accumulation_is_total =
      linalg::probe_dispatch_kernel_types<ContractionSelector const&, linalg::contract_op<2, 2, 1>, output_mdspec&,
                                          Scalar&, a_mdspec&, intermediate_const_mdspec&, Scalar&>() ==
      linalg::KernelTypeAcceptance::yes;
  return right_product_is_total && accumulation_is_total;
}

template <class Function> void execute_rabc_batch(SerialBlockExecution, std::size_t size, Function&& function)
{
  for (std::size_t index = 0; index < size; ++index)
    function(index);
}

template <class Function> void execute_rabc_batch(SchedulerBatchBlockExecution, std::size_t size, Function&& function)
{
  async::execute_batch(size, std::forward<Function>(function));
}

template <class Output, class Plan, class A, class B, class C>
  requires CompatibleHostRabcOperands<Output, Plan, A, B, C>
void validate_rabc_operands(Output const& output, Plan const& plan, A const& a, B const& b, C const& c)
{
  for (auto const& term : plan.terms())
  {
    if (term.r_ordinal >= output.stored_block_count() || term.a_ordinal >= a.stored_block_count() ||
        term.b_ordinal >= b.stored_block_count() || term.c_ordinal >= c.stored_block_count())
    {
      throw std::invalid_argument("R/A/B/C contraction term has an out-of-range block ordinal");
    }

    auto const output_block = output.block_by_ordinal(term.r_ordinal);
    auto const a_block = a.block_by_ordinal(term.a_ordinal);
    auto const b_block = b.block_by_ordinal(term.b_ordinal);
    auto const c_block = c.block_by_ordinal(term.c_ordinal);
    if (a_block.extent(1) != b_block.extent(0) || b_block.extent(1) != c_block.extent(1) ||
        output_block.extent(0) != a_block.extent(0) || output_block.extent(1) != c_block.extent(0))
    {
      throw std::invalid_argument("R/A/B/C contraction term has incompatible dense block extents");
    }
  }
}

struct RightFirstRabcGroup
{
    std::size_t b_ordinal;
    std::size_t c_ordinal;
};

template <class Plan>
[[nodiscard]] auto make_right_first_groups(Plan const& plan, std::vector<std::size_t>& term_group)
    -> std::vector<RightFirstRabcGroup>
{
  auto const terms = plan.terms();
  std::vector<std::size_t> order(terms.size());
  std::iota(order.begin(), order.end(), std::size_t{});
  std::ranges::stable_sort(order, [&](std::size_t lhs, std::size_t rhs) {
    return std::tuple{terms[lhs].b_ordinal, terms[lhs].c_ordinal} <
           std::tuple{terms[rhs].b_ordinal, terms[rhs].c_ordinal};
  });

  std::vector<RightFirstRabcGroup> groups;
  groups.reserve(order.size());
  for (std::size_t const term_index : order)
  {
    auto const& term = terms[term_index];
    if (groups.empty() || groups.back().b_ordinal != term.b_ordinal || groups.back().c_ordinal != term.c_ordinal)
      groups.push_back({.b_ordinal = term.b_ordinal, .c_ordinal = term.c_ordinal});
    term_group[term_index] = groups.size() - 1;
  }
  return groups;
}

template <class Output, class Plan, class A, class B, class C, class ContractionSelector>
  requires CompatibleHostRabcOperands<Output, Plan, A, B, C>
void execute_host_right_first_rabc(ContractionSelector const& selector, Output& output, Plan const& plan, A const& a,
                                   B const& b, C const& c)
{
  using scalar_type = block_tensor_value_t<Output>;
  auto const terms = plan.terms();
  std::vector<std::size_t> term_group(terms.size());
  auto const groups = make_right_first_groups(plan, term_group);

  std::vector<ColumnMajorTensor<scalar_type, 2>> intermediates;
  intermediates.reserve(groups.size());
  for (auto const& group : groups)
  {
    auto const b_block = b.block_by_ordinal(group.b_ordinal);
    auto const c_block = c.block_by_ordinal(group.c_ordinal);
    intermediates.emplace_back(b_block.extent(0), c_block.extent(0));
  }

  using execution_policy = typename block_tensor_type_t<Output>::storage_policy::block_execution_policy;
  constexpr std::array<std::pair<std::size_t, std::size_t>, 1> right_dimensions{std::pair{1U, 1U}};
  execute_rabc_batch(execution_policy{}, groups.size(), [&](std::size_t group_index) {
    auto const& group = groups[group_index];
    auto const b_block = b.block_by_ordinal(group.b_ordinal);
    auto const c_block = c.block_by_ordinal(group.c_ordinal);
    linalg::contract(selector, intermediates[group_index], scalar_type{1}, b_block, c_block, right_dimensions,
                     scalar_type{});
  });

  std::vector<std::size_t> output_offsets(output.stored_block_count() + 1, 0);
  for (auto const& term : terms)
    ++output_offsets[term.r_ordinal + 1];
  std::partial_sum(output_offsets.begin(), output_offsets.end(), output_offsets.begin());

  std::vector<std::size_t> output_order(output.stored_block_count());
  std::iota(output_order.begin(), output_order.end(), std::size_t{});
  std::vector<long double> output_cost(output.stored_block_count(), 0);
  for (std::size_t index = 0; index < terms.size(); ++index)
  {
    auto const& term = terms[index];
    auto const a_block = a.block_by_ordinal(term.a_ordinal);
    auto const& intermediate = intermediates[term_group[index]];
    output_cost[term.r_ordinal] +=
        static_cast<long double>(a_block.extent(0)) * a_block.extent(1) * intermediate.extent(1);
  }
  std::ranges::stable_sort(output_order,
                           [&](std::size_t lhs, std::size_t rhs) { return output_cost[lhs] > output_cost[rhs]; });

  constexpr std::array<std::pair<std::size_t, std::size_t>, 1> left_dimensions{std::pair{1U, 0U}};
  execute_rabc_batch(execution_policy{}, output_order.size(), [&](std::size_t ordered_index) {
    std::size_t const output_ordinal = output_order[ordered_index];
    auto output_block = output.block_by_ordinal(output_ordinal);
    std::size_t const first = output_offsets[output_ordinal];
    std::size_t const last = output_offsets[output_ordinal + 1];
    if (first == last)
    {
      uni20::transform_inplace(output_block, linalg::scale{scalar_type{}});
      return;
    }

    for (std::size_t index = first; index < last; ++index)
    {
      auto const& term = terms[index];
      auto const a_block = a.block_by_ordinal(term.a_ordinal);
      scalar_type const beta = index == first ? scalar_type{} : scalar_type{1};
      linalg::contract(selector, output_block, term.coefficient, a_block, intermediates[term_group[index]],
                       left_dimensions, beta);
    }
  });
}

} // namespace detail

/// \brief Report host right-first eligibility for immediate matrix-block operands.
template <class ContractionSelector, detail::HostWritableRabcTensor Output, uni20::Scalar Scalar,
          detail::HostReadableRabcTensor A, detail::HostReadableRabcTensor B, detail::HostReadableRabcTensor C>
  requires detail::CompatibleHostRabcOperands<Output, RabcContractionPlan<Scalar>, A, B, C>
consteval auto kernel_accepts_types(HostRightFirstRabcBackend<ContractionSelector> const&, rabc_contract_op const&,
                                    Output&, RabcContractionPlan<Scalar> const&, A const&, B const&, C const&)
{
  if constexpr (detail::host_right_first_types_compatible<ContractionSelector, Output, Scalar, A, B, C>())
    return linalg::kernel_types_yes;
  else
    return linalg::kernel_types_no;
}

/// \brief Execute a host right-first R/A/B/C contraction.
template <class ContractionSelector, detail::HostWritableRabcTensor Output, uni20::Scalar Scalar,
          detail::HostReadableRabcTensor A, detail::HostReadableRabcTensor B, detail::HostReadableRabcTensor C>
  requires detail::CompatibleHostRabcOperands<Output, RabcContractionPlan<Scalar>, A, B, C> &&
           (detail::host_right_first_types_compatible<ContractionSelector, Output, Scalar, A, B, C>())
auto try_kernel(HostRightFirstRabcBackend<ContractionSelector> const& backend, rabc_contract_op const&, Output& output,
                RabcContractionPlan<Scalar> const& plan, A const& a, B const& b, C const& c) -> linalg::KernelAttempt
{
  detail::validate_rabc_operands(output, plan, a, b, c);
  detail::execute_host_right_first_rabc(backend.contraction_selector, output, plan, a, b, c);
  return linalg::KernelAttempt::success;
}

} // namespace uni20::tensor_network
