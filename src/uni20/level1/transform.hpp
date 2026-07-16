#pragma once

/**
 * \file transform.hpp
 * \ingroup level1_ops
 * \brief Eager strided elementwise transform primitives.
 */

#include <uni20/mdspan/iteration_plan.hpp>

#include <concepts>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <bool ReadsOutput, class Output, class Op, class... Inputs>
consteval bool strided_transform_operation()
{
  using output_type = std::remove_cvref_t<Output>;
  if constexpr (!MutableStridedMdspan<Output> || (!(StridedMdspan<Inputs> && ...)) ||
                !((output_type::rank() == std::remove_cvref_t<Inputs>::rank()) && ...))
  {
    return false;
  }
  else if constexpr (ReadsOutput)
  {
    return std::invocable<Op&, typename output_type::reference,
                          typename std::remove_cvref_t<Inputs>::reference...> &&
           requires(typename output_type::reference output, Op& operation) {
             output = std::invoke(operation, output,
                                  std::declval<typename std::remove_cvref_t<Inputs>::reference>()...);
           };
  }
  else
  {
    return sizeof...(Inputs) >= 1 &&
           std::invocable<Op&, typename std::remove_cvref_t<Inputs>::reference...> &&
           requires(typename output_type::reference output, Op& operation) {
             output =
                 std::invoke(operation, std::declval<typename std::remove_cvref_t<Inputs>::reference>()...);
           };
  }
}

template <bool ReadsOutput, class Output, class Op, class... Inputs>
  requires(strided_transform_operation<ReadsOutput, Output, Op, Inputs...>())
void transform_strided(Output output, Op&& operation, Inputs const&... inputs)
{
  auto mappings = std::tuple{output.mapping(), inputs.mapping()...};
  auto [plan, offsets] = make_multi_iteration_plan_with_offset(mappings);

  TransformUnrollHelper<ReadsOutput, Op, Output, Inputs...> helper{
      std::forward<Op>(operation), output, inputs...};
  helper.run(plan, offsets);
}

} // namespace detail

/// \brief Overwrite a strided output with an elementwise unary transform.
/// \details Computes `output[i...] = operation(input[i...])` for every logical
///          index. Input and output may use different strided mappings.
/// \pre Output and input have equal extents.
/// \pre Output storage does not overlap input storage.
template <MutableStridedMdspan Output, StridedMdspan Input, class Op>
  requires(detail::strided_transform_operation<false, Output, Op, Input>())
void transform(Output output, Input const& input, Op&& operation)
{
  detail::transform_strided<false>(output, std::forward<Op>(operation), input);
}

/// \brief Overwrite a strided output with an elementwise binary transform.
/// \details Computes `output[i...] = operation(lhs[i...], rhs[i...])` for every
///          logical index. The three operands may use different strided mappings.
/// \pre Output, lhs, and rhs have equal extents.
/// \pre Output storage does not overlap either input.
template <MutableStridedMdspan Output, StridedMdspan Lhs, StridedMdspan Rhs, class Op>
  requires(detail::strided_transform_operation<false, Output, Op, Lhs, Rhs>())
void transform(Output output, Lhs const& lhs, Rhs const& rhs, Op&& operation)
{
  detail::transform_strided<false>(output, std::forward<Op>(operation), lhs, rhs);
}

/// \brief Apply an elementwise unary transform to a strided span in place.
/// \details Computes `output[i...] = operation(output[i...])`.
template <MutableStridedMdspan Output, class Op>
  requires(detail::strided_transform_operation<true, Output, Op>())
void transform_inplace(Output output, Op&& operation)
{
  detail::transform_strided<true>(output, std::forward<Op>(operation));
}

/// \brief Apply an elementwise binary update to a strided span in place.
/// \details Computes `output[i...] = operation(output[i...], rhs[i...])`.
/// \pre Output and rhs have equal extents.
/// \pre Rhs storage does not overlap output storage.
template <MutableStridedMdspan Output, StridedMdspan Rhs, class Op>
  requires(detail::strided_transform_operation<true, Output, Op, Rhs>())
void transform_inplace(Output output, Rhs const& rhs, Op&& operation)
{
  detail::transform_strided<true>(output, std::forward<Op>(operation), rhs);
}

} // namespace uni20
