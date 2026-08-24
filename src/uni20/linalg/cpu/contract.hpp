#pragma once

/**
 * \file contract.hpp
 * \ingroup linalg
 * \brief Accessor-respecting reference CPU tensor contraction.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/contraction_axes.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/output.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20::linalg::cpu
{
namespace detail
{

template <class Span, class Scalar, std::size_t... Axis>
consteval bool contraction_input_expressions(std::index_sequence<Axis...>)
{
  using span_type = std::remove_cvref_t<Span>;
  return requires(Span& span, typename span_type::index_type index) {
    static_cast<Scalar>(span.operator[](((void)Axis, index)...));
  };
}

template <class Span, class Scalar, std::size_t... Axis>
consteval bool contraction_output_expressions(std::index_sequence<Axis...>)
{
  using span_type = std::remove_cvref_t<Span>;
  return requires(Span& span, typename span_type::index_type index, Scalar value) {
    static_cast<Scalar>(span.operator[](((void)Axis, index)...));
    span.operator[](((void)Axis, index)...) = value;
  };
}

template <class Span, std::size_t Rank, std::size_t... Axis>
constexpr decltype(auto) element_at(Span& span, std::array<uni20::index_type, Rank> const& indices,
                                    std::index_sequence<Axis...>)
{
  using index_type = typename std::remove_cvref_t<Span>::index_type;
  return span.operator[](static_cast<index_type>(indices[Axis])...);
}

template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan, std::size_t LhsRank, std::size_t RhsRank,
          std::size_t ContractedRank>
class ContractionLoop {
  public:
    using axes_type = ContractionAxes<LhsRank, RhsRank, ContractedRank>;
    static constexpr std::size_t output_rank = axes_type::output_rank;

    ContractionLoop(OutputMdspan& output, Scalar alpha, LhsMdspan& lhs, RhsMdspan& rhs, Scalar beta,
                    axes_type const& axes)
        : output_(output), alpha_(alpha), lhs_(lhs), rhs_(rhs), beta_(beta), axes_(axes)
    {}

    void run() { this->loop_output(0); }

  private:
    void loop_output(std::size_t axis)
    {
      if (axis == output_rank)
      {
        this->evaluate_output_element();
        return;
      }

      auto const extent = static_cast<uni20::index_type>(output_.extent(axis));
      for (uni20::index_type index = 0; index < extent; ++index)
      {
        output_indices_[axis] = index;
        if (axis < axes_type::lhs_surviving_rank)
          lhs_indices_[axes_.lhs_surviving[axis]] = index;
        else
          rhs_indices_[axes_.rhs_surviving[axis - axes_type::lhs_surviving_rank]] = index;
        this->loop_output(axis + 1);
      }
    }

    void evaluate_output_element()
    {
      Scalar product{};
      if (alpha_ != Scalar{}) this->accumulate_contracted(0, product);

      auto&& output_value = element_at(output_, output_indices_, std::make_index_sequence<output_rank>{});
      if (beta_ == Scalar{})
        output_value = alpha_ * product;
      else
        output_value = beta_ * static_cast<Scalar>(output_value) + alpha_ * product;
    }

    void accumulate_contracted(std::size_t axis, Scalar& product)
    {
      if (axis == ContractedRank)
      {
        product += static_cast<Scalar>(element_at(lhs_, lhs_indices_, std::make_index_sequence<LhsRank>{})) *
                   static_cast<Scalar>(element_at(rhs_, rhs_indices_, std::make_index_sequence<RhsRank>{}));
        return;
      }

      auto const lhs_axis = axes_.lhs_contracted[axis];
      auto const rhs_axis = axes_.rhs_contracted[axis];
      auto const extent = static_cast<uni20::index_type>(lhs_.extent(lhs_axis));
      for (uni20::index_type index = 0; index < extent; ++index)
      {
        lhs_indices_[lhs_axis] = index;
        rhs_indices_[rhs_axis] = index;
        this->accumulate_contracted(axis + 1, product);
      }
    }

    OutputMdspan& output_;
    Scalar alpha_;
    LhsMdspan& lhs_;
    RhsMdspan& rhs_;
    Scalar beta_;
    axes_type const& axes_;
    std::array<uni20::index_type, output_rank> output_indices_{};
    std::array<uni20::index_type, LhsRank> lhs_indices_{};
    std::array<uni20::index_type, RhsRank> rhs_indices_{};
};

} // namespace detail

/// \brief Report whether resolved mdspans support reference contraction expressions.
template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan, std::size_t LhsRank, std::size_t RhsRank,
          std::size_t ContractedRank>
concept ContractionCompatible =
    uni20::MutableRankedMdspanLike<OutputMdspan, ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank> &&
    uni20::Scalar<Scalar> && uni20::RankedMdspanLike<LhsMdspan, LhsRank> &&
    uni20::RankedMdspanLike<RhsMdspan, RhsRank> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<OutputMdspan>::element_type>, Scalar> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<LhsMdspan>::element_type>, Scalar> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<RhsMdspan>::element_type>, Scalar> &&
    detail::contraction_output_expressions<OutputMdspan, Scalar>(
        std::make_index_sequence<ContractionAxes<LhsRank, RhsRank, ContractedRank>::output_rank>{}) &&
    detail::contraction_input_expressions<LhsMdspan, Scalar>(std::make_index_sequence<LhsRank>{}) &&
    detail::contraction_input_expressions<RhsMdspan, Scalar>(std::make_index_sequence<RhsRank>{}) &&
    requires(Scalar value) {
      value += value * value;
      { value == Scalar{} } -> std::convertible_to<bool>;
      { value != Scalar{} } -> std::convertible_to<bool>;
    };

/// \brief Execute a fixed-output pairwise contraction over resolved host mdspans.
/// \details This correctness implementation evaluates every operand through
///          its accessor. It supports rank-zero operands, outer products, empty
///          contracted extents, and arbitrary mdspan mappings.
template <class OutputMdspan, uni20::Scalar Scalar, class LhsMdspan, class RhsMdspan, std::size_t LhsRank,
          std::size_t RhsRank, std::size_t ContractedRank>
  requires ContractionCompatible<OutputMdspan, Scalar, LhsMdspan, RhsMdspan, LhsRank, RhsRank, ContractedRank>
void contract(OutputMdspan& output, Scalar alpha, LhsMdspan& lhs, RhsMdspan& rhs, Scalar beta,
              ContractionAxes<LhsRank, RhsRank, ContractedRank> const& axes)
{
  CHECK(contraction_axes_are_valid(axes));
  auto const required_extents = contraction_output_extents(lhs, rhs, axes);
  CHECK(uni20::tensor_extents_equal(output.extents(), required_extents),
        "contraction output extents do not match the surviving input axes");

  detail::ContractionLoop<OutputMdspan, Scalar, LhsMdspan, RhsMdspan, LhsRank, RhsRank, ContractedRank> loop{
      output, alpha, lhs, rhs, beta, axes};
  loop.run();
}

} // namespace uni20::linalg::cpu
