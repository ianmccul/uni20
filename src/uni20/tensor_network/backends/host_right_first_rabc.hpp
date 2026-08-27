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
///          The sparse logical plan remains order-neutral. One-shot dispatch
///          prepares transient state, while `prepare_rabc_contract` retains
///          grouping and workspace for repeated applications.
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
    HostReadableRabcTensor<C> && RabcPlan<Plan> &&
    std::same_as<typename std::remove_cvref_t<Plan>::r_key_type, block_tensor_key_t<Output>> &&
    std::same_as<typename std::remove_cvref_t<Plan>::a_key_type, block_tensor_key_t<A>> &&
    std::same_as<typename std::remove_cvref_t<Plan>::b_key_type, block_tensor_key_t<B>> &&
    std::same_as<typename std::remove_cvref_t<Plan>::c_key_type, block_tensor_key_t<C>> &&
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

template <uni20::Scalar Scalar> struct BoundRabcTerm
{
    std::size_t r_ordinal;
    std::size_t a_ordinal;
    std::size_t b_ordinal;
    std::size_t c_ordinal;
    Scalar coefficient;
};

template <class Tensor, class Key>
[[nodiscard]] auto bind_rabc_keys(Tensor const& tensor, std::span<Key const> keys) -> std::vector<std::size_t>
{
  std::vector<std::size_t> result;
  result.reserve(keys.size());
  for (auto const& key : keys)
  {
    auto const found = std::ranges::lower_bound(tensor.stored_keys(), key);
    if (found == tensor.stored_keys().end() || *found != key)
      throw std::invalid_argument("R/A/B/C logical key is not stored by its operand");
    result.push_back(static_cast<std::size_t>(found - tensor.stored_keys().begin()));
  }
  return result;
}

template <class Output, class Plan, class A, class B, class C>
  requires CompatibleHostRabcOperands<Output, Plan, A, B, C>
[[nodiscard]] auto bind_rabc_operands(Output const& output, Plan const& plan, A const& a, B const& b, C const& c)
    -> std::vector<BoundRabcTerm<typename std::remove_cvref_t<Plan>::scalar_type>>
{
  using scalar_type = typename std::remove_cvref_t<Plan>::scalar_type;
  auto const r_ordinals = bind_rabc_keys(output, plan.r_keys());
  auto const a_ordinals = bind_rabc_keys(a, plan.a_keys());
  auto const b_ordinals = bind_rabc_keys(b, plan.b_keys());
  auto const c_ordinals = bind_rabc_keys(c, plan.c_keys());

  std::vector<BoundRabcTerm<scalar_type>> result;
  result.reserve(plan.term_count());
  for (auto const& term : plan.terms())
  {
    BoundRabcTerm<scalar_type> binding{.r_ordinal = r_ordinals[term.r_key_index],
                                      .a_ordinal = a_ordinals[term.a_key_index],
                                      .b_ordinal = b_ordinals[term.b_key_index],
                                      .c_ordinal = c_ordinals[term.c_key_index],
                                      .coefficient = term.coefficient};

    auto const output_block = output.block_by_ordinal(binding.r_ordinal);
    auto const a_block = a.block_by_ordinal(binding.a_ordinal);
    auto const b_block = b.block_by_ordinal(binding.b_ordinal);
    auto const c_block = c.block_by_ordinal(binding.c_ordinal);
    if (a_block.extent(1) != b_block.extent(0) || b_block.extent(1) != c_block.extent(1) ||
        output_block.extent(0) != a_block.extent(0) || output_block.extent(1) != c_block.extent(0))
      throw std::invalid_argument("R/A/B/C contraction term has incompatible dense block extents");
    result.push_back(std::move(binding));
  }

  std::ranges::stable_sort(result, [](auto const& lhs, auto const& rhs) {
    return std::tuple{lhs.r_ordinal, lhs.a_ordinal, lhs.b_ordinal, lhs.c_ordinal} <
           std::tuple{rhs.r_ordinal, rhs.a_ordinal, rhs.b_ordinal, rhs.c_ordinal};
  });
  return result;
}

struct RightFirstRabcGroup
{
    std::size_t b_ordinal;
    std::size_t c_ordinal;
};

template <class Terms>
[[nodiscard]] auto make_right_first_groups(Terms const& terms, std::vector<std::size_t>& term_group)
    -> std::vector<RightFirstRabcGroup>
{
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

} // namespace detail

/// \brief Prepared host right-first executor for one fixed R/A/B/C structure.
/// \details Construction binds logical keys to storage ordinals, validates
///          dense extents, derives the
///          unique `(B_b,C_c)` groups and output execution order, and allocates
///          every intermediate once. Repeated calls reuse that schedule and
///          workspace. The prepared object is tied to the plan and block
///          structures used at construction and is not safe for concurrent
///          execution.
/// \tparam Plan Execution-neutral logical coefficient plan.
/// \tparam ContractionSelector Dense contraction backend selector.
template <RabcPlan Plan, linalg::KernelBackendSelector ContractionSelector>
class PreparedHostRightFirstRabcContraction {
  public:
    using plan_type = std::remove_cvref_t<Plan>;
    using scalar_type = typename plan_type::scalar_type;
    using contraction_selector_type = ContractionSelector;

    /// \brief Compile a fixed host execution schedule and allocate its workspace.
    /// \param selector Dense contraction selector retained for every application.
    /// \param output Prototype defining output block ordinals and extents.
    /// \param plan Execution-neutral coefficient plan retained by value.
    /// \param a Prototype left block family.
    /// \param b Prototype center block family.
    /// \param c Prototype right block family.
    template <detail::HostWritableRabcTensor Output, detail::HostReadableRabcTensor A, detail::HostReadableRabcTensor B,
              detail::HostReadableRabcTensor C>
      requires detail::CompatibleHostRabcOperands<Output, plan_type, A, B, C>
    PreparedHostRightFirstRabcContraction(contraction_selector_type selector, Output const& output,
                                          plan_type plan, A const& a, B const& b, C const& c)
        : selector_(std::move(selector)), plan_(std::move(plan)), term_group_(plan_.term_count())
    {
      bound_terms_ = detail::bind_rabc_operands(output, plan_, a, b, c);
      groups_ = detail::make_right_first_groups(bound_terms_, term_group_);

      intermediates_.reserve(groups_.size());
      for (auto const& group : groups_)
      {
        auto const b_block = b.block_by_ordinal(group.b_ordinal);
        auto const c_block = c.block_by_ordinal(group.c_ordinal);
        intermediates_.emplace_back(b_block.extent(0), c_block.extent(0));
      }

      output_offsets_.assign(output.stored_block_count() + 1, 0);
      for (auto const& term : bound_terms_)
        ++output_offsets_[term.r_ordinal + 1];
      std::partial_sum(output_offsets_.begin(), output_offsets_.end(), output_offsets_.begin());

      output_order_.resize(output.stored_block_count());
      std::iota(output_order_.begin(), output_order_.end(), std::size_t{});
      std::vector<long double> output_cost(output.stored_block_count(), 0);
      for (std::size_t index = 0; index < bound_terms_.size(); ++index)
      {
        auto const& term = bound_terms_[index];
        auto const a_block = a.block_by_ordinal(term.a_ordinal);
        auto const& intermediate = intermediates_[term_group_[index]];
        output_cost[term.r_ordinal] +=
            static_cast<long double>(a_block.extent(0)) * a_block.extent(1) * intermediate.extent(1);
      }
      std::ranges::stable_sort(output_order_,
                               [&](std::size_t lhs, std::size_t rhs) { return output_cost[lhs] > output_cost[rhs]; });
    }

    /// \brief Execute using the retained schedule and workspace.
    /// \pre All operands have the same stored logical-key sequences and dense
    ///      extents supplied at construction, and output storage does not
    ///      overlap an input.
    /// \param output Fixed-structure output block family.
    /// \param a Left block family matching the prepared prototype.
    /// \param b Center block family matching the prepared prototype.
    /// \param c Right block family matching the prepared prototype.
    template <detail::HostWritableRabcTensor Output, detail::HostReadableRabcTensor A, detail::HostReadableRabcTensor B,
              detail::HostReadableRabcTensor C>
      requires detail::CompatibleHostRabcOperands<Output, plan_type, A, B, C>
    void operator()(Output& output, A const& a, B const& b, C const& c)
    {
      using execution_policy = typename block_tensor_type_t<Output>::storage_policy::block_execution_policy;
      constexpr std::array<std::pair<std::size_t, std::size_t>, 1> right_dimensions{std::pair{1U, 1U}};
      detail::execute_rabc_batch(execution_policy{}, groups_.size(), [&](std::size_t group_index) {
        auto const& group = groups_[group_index];
        auto const b_block = b.block_by_ordinal(group.b_ordinal);
        auto const c_block = c.block_by_ordinal(group.c_ordinal);
        linalg::contract(selector_, intermediates_[group_index], scalar_type{1}, b_block, c_block, right_dimensions,
                         scalar_type{});
      });

      constexpr std::array<std::pair<std::size_t, std::size_t>, 1> left_dimensions{std::pair{1U, 0U}};
      detail::execute_rabc_batch(execution_policy{}, output_order_.size(), [&](std::size_t ordered_index) {
        std::size_t const output_ordinal = output_order_[ordered_index];
        auto output_block = output.block_by_ordinal(output_ordinal);
        std::size_t const first = output_offsets_[output_ordinal];
        std::size_t const last = output_offsets_[output_ordinal + 1];
        if (first == last)
        {
          uni20::fill(output_block, scalar_type{});
          return;
        }

        for (std::size_t index = first; index < last; ++index)
        {
          auto const& term = bound_terms_[index];
          auto const a_block = a.block_by_ordinal(term.a_ordinal);
          scalar_type const beta = index == first ? scalar_type{} : scalar_type{1};
          linalg::contract(selector_, output_block, term.coefficient, a_block, intermediates_[term_group_[index]],
                           left_dimensions, beta);
        }
      });
    }

    /// \brief Return the number of retained `(B,C)` intermediate blocks.
    [[nodiscard]] auto intermediate_count() const noexcept -> std::size_t { return intermediates_.size(); }

    /// \brief Return the immutable execution-neutral coefficient plan.
    [[nodiscard]] auto plan() const noexcept -> plan_type const& { return plan_; }

  private:
    contraction_selector_type selector_;
    plan_type plan_;
    std::vector<detail::BoundRabcTerm<scalar_type>> bound_terms_;
    std::vector<std::size_t> term_group_;
    std::vector<detail::RightFirstRabcGroup> groups_;
    std::vector<ColumnMajorTensor<scalar_type, 2>> intermediates_;
    std::vector<std::size_t> output_offsets_;
    std::vector<std::size_t> output_order_;
};

template <class ContractionSelector, class Output, RabcPlan Plan, class A, class B, class C>
PreparedHostRightFirstRabcContraction(ContractionSelector, Output const&, Plan, A const&, B const&, C const&)
    -> PreparedHostRightFirstRabcContraction<std::remove_cvref_t<Plan>, std::remove_cvref_t<ContractionSelector>>;

/// \brief Report host right-first eligibility for immediate matrix-block operands.
template <class ContractionSelector, detail::HostWritableRabcTensor Output, RabcPlan Plan,
          detail::HostReadableRabcTensor A, detail::HostReadableRabcTensor B, detail::HostReadableRabcTensor C>
  requires detail::CompatibleHostRabcOperands<Output, Plan, A, B, C>
consteval auto kernel_accepts_types(HostRightFirstRabcBackend<ContractionSelector> const&, rabc_contract_op const&,
                                    Output&, Plan const&, A const&, B const&, C const&)
{
  using scalar_type = typename std::remove_cvref_t<Plan>::scalar_type;
  if constexpr (detail::host_right_first_types_compatible<ContractionSelector, Output, scalar_type, A, B, C>())
    return linalg::kernel_types_yes;
  else
    return linalg::kernel_types_no;
}

/// \brief Execute a host right-first R/A/B/C contraction.
template <class ContractionSelector, detail::HostWritableRabcTensor Output, RabcPlan Plan,
          detail::HostReadableRabcTensor A, detail::HostReadableRabcTensor B, detail::HostReadableRabcTensor C>
  requires detail::CompatibleHostRabcOperands<Output, Plan, A, B, C> &&
           (detail::host_right_first_types_compatible<ContractionSelector, Output,
                                                      typename std::remove_cvref_t<Plan>::scalar_type, A, B, C>())
auto try_kernel(HostRightFirstRabcBackend<ContractionSelector> const& backend, rabc_contract_op const&, Output& output,
                Plan const& plan, A const& a, B const& b, C const& c) -> linalg::KernelAttempt
{
  PreparedHostRightFirstRabcContraction prepared(backend.contraction_selector, output, plan, a, b, c);
  prepared(output, a, b, c);
  return linalg::KernelAttempt::success;
}

} // namespace uni20::tensor_network
