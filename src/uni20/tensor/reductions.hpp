#pragma once

/**
 * \file reductions.hpp
 * \ingroup tensor
 * \brief Tensor sums, inner products, and Euclidean norms with explicit result residency.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/linalg/backends/cpu/reductions.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/storage/vectorstorage.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/tensor.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <class LhsSpan, class RhsSpan>
concept CompatibleInnerProductSpans = MdspanLike<LhsSpan> && MdspanLike<RhsSpan> &&
                                      (std::remove_cvref_t<LhsSpan>::rank() == std::remove_cvref_t<RhsSpan>::rank()) &&
                                      std::same_as<typename std::remove_cvref_t<LhsSpan>::value_type,
                                                   typename std::remove_cvref_t<RhsSpan>::value_type> &&
                                      RealOrComplex<typename std::remove_cvref_t<LhsSpan>::value_type>;

template <class LhsTensor, class RhsTensor>
concept CompatibleInnerProductTensors =
    DeviceTensorView<LhsTensor> && DeviceTensorView<RhsTensor> &&
    (device_tensor_mdspan_t<LhsTensor>::rank() == device_tensor_mdspan_t<RhsTensor>::rank()) &&
    std::same_as<tensor_element_t<LhsTensor>, tensor_element_t<RhsTensor>> &&
    RealOrComplex<tensor_element_t<LhsTensor>>;

template <class Tensor>
concept ReductionStorageTypedTensor =
    DeviceTensorView<Tensor> && requires { typename std::remove_cvref_t<Tensor>::storage_policy; };

template <class... Inputs> struct reduction_result_storage
{
    using selected_type = linalg::detail::first_backend_bound_storage_t<Inputs...>;
    using type = std::conditional_t<std::is_void_v<selected_type>, VectorStorage, selected_type>;
};

template <class... Inputs> using reduction_result_storage_t = typename reduction_result_storage<Inputs...>::type;

template <class Element, class... Inputs>
concept ReductionResultAvailable = (ReductionStorageTypedTensor<Inputs> && ...) && requires {
  typename reduction_result_storage_t<Inputs...>::template storage_t<Element>;
};

template <DeviceTensorView InputTensor, std::size_t ReducedRank>
using sum_reduction_layout_t =
    std::conditional_t<device_tensor_mdspan_t<InputTensor>::rank() == ReducedRank, ColumnMajor,
                       materialized_layout_t<void, device_tensor_mdspan_t<InputTensor>>>;

template <DeviceTensorView InputTensor, std::size_t ReducedRank>
using sum_reduction_result_t =
    Tensor<tensor_element_t<InputTensor>, device_tensor_mdspan_t<InputTensor>::rank() - ReducedRank,
           reduction_result_storage_t<InputTensor>, sum_reduction_layout_t<InputTensor, ReducedRank>>;

template <DeviceTensorView InputTensor, std::size_t InputRank, std::size_t ReducedRank, std::size_t... OutputAxis>
[[nodiscard]] auto reduction_output_extents(InputTensor const& input,
                                            linalg::ReductionAxes<InputRank, ReducedRank> const& axes,
                                            std::index_sequence<OutputAxis...>)
{
  using extents_type = stdex::dextents<uni20::index_type, InputRank - ReducedRank>;
  return extents_type{static_cast<uni20::index_type>(input.extent(axes.surviving[OutputAxis]))...};
}

template <DeviceTensorView InputTensor, std::size_t InputRank, std::size_t ReducedRank>
[[nodiscard]] auto reduction_output_extents(InputTensor const& input,
                                            linalg::ReductionAxes<InputRank, ReducedRank> const& axes)
{
  return reduction_output_extents(input, axes, std::make_index_sequence<InputRank - ReducedRank>{});
}

template <class BackendSelector, class OutputSpan, class InputSpan, std::size_t InputRank, std::size_t ReducedRank>
void dispatch_sum(BackendSelector&& selector, OutputSpan&& output, InputSpan&& input,
                  linalg::ReductionAxes<InputRank, ReducedRank> axes)
{
  auto operation = linalg::sum_reduction_op<InputRank, ReducedRank>{.axes = std::move(axes)};
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), std::move(operation),
                          std::forward<OutputSpan>(output), std::forward<InputSpan>(input));
}

template <class Reference, class... Others>
void require_reduction_extents(Reference const& reference, Others const&... others)
{
  constexpr std::size_t rank = device_tensor_mdspan_t<Reference>::rank();
  if constexpr (rank > 0)
  {
    auto require_one = [&](auto const& other) {
      for (std::size_t axis = 0; axis < rank; ++axis)
        ERROR_IF(reference.extent(axis) != other.extent(axis), "reduction operands have different extents");
    };
    (require_one(others), ...);
  }
}

} // namespace detail

/// \brief Sum every element of an mdspan into a rank-zero output.
/// \details The output preserves the input element type. The CPU reference
///          backend uses compensated accumulation in that scalar field.
template <linalg::KernelBackendSelector BackendSelector, MutableRankedMdspanLike<0> OutputSpan, MdspanLike InputSpan>
  requires RealOrComplex<typename std::remove_cvref_t<InputSpan>::value_type> &&
           std::same_as<typename std::remove_cvref_t<OutputSpan>::value_type,
                        typename std::remove_cvref_t<InputSpan>::value_type>
void sum(BackendSelector&& selector, OutputSpan&& output, InputSpan&& input)
{
  using input_type = std::remove_cvref_t<InputSpan>;
  detail::dispatch_sum(std::forward<BackendSelector>(selector), std::forward<OutputSpan>(output),
                       std::forward<InputSpan>(input), linalg::all_reduction_axes<input_type::rank()>());
}

/// \brief Sum selected mdspan axes into an existing lower-rank output.
/// \details Surviving axes retain their original logical order. Negative axes
///          count backward from the input rank.
template <linalg::KernelBackendSelector BackendSelector, MutableMdspanLike OutputSpan, MdspanLike InputSpan,
          linalg::ReductionAxis FirstAxis, linalg::ReductionAxis... RestAxes>
  requires RealOrComplex<typename std::remove_cvref_t<InputSpan>::value_type> &&
           std::same_as<typename std::remove_cvref_t<OutputSpan>::value_type,
                        typename std::remove_cvref_t<InputSpan>::value_type> &&
           (1 + sizeof...(RestAxes) <= std::remove_cvref_t<InputSpan>::rank()) &&
           (std::remove_cvref_t<OutputSpan>::rank() + 1 + sizeof...(RestAxes) == std::remove_cvref_t<InputSpan>::rank())
void sum(BackendSelector&& selector, OutputSpan&& output, InputSpan&& input, FirstAxis first_axis,
         RestAxes... rest_axes)
{
  using input_type = std::remove_cvref_t<InputSpan>;
  auto axes = linalg::make_reduction_axes<input_type::rank()>(first_axis, rest_axes...);
  detail::dispatch_sum(std::forward<BackendSelector>(selector), std::forward<OutputSpan>(output),
                       std::forward<InputSpan>(input), std::move(axes));
}

/// \brief Return the full mdspan sum as a host C++ scalar.
template <linalg::KernelBackendSelector BackendSelector, MdspanLike InputSpan>
  requires RealOrComplex<typename std::remove_cvref_t<InputSpan>::value_type>
[[nodiscard]] auto sum_host(BackendSelector&& selector, InputSpan&& input) ->
    typename std::remove_cvref_t<InputSpan>::value_type
{
  using input_type = std::remove_cvref_t<InputSpan>;
  using result_type = typename input_type::value_type;
  result_type result{};
  detail::dispatch_sum(std::forward<BackendSelector>(selector), result, std::forward<InputSpan>(input),
                       linalg::all_reduction_axes<input_type::rank()>());
  return result;
}

/// \brief Sum every element of a device tensor view into an existing scalar tensor.
template <linalg::KernelBackendSelector BackendSelector, MutableScalarDeviceTensorView OutputTensor,
          DeviceTensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<InputTensor>>
void sum(BackendSelector&& selector, OutputTensor&& output, InputTensor const& input)
{
  constexpr std::size_t input_rank = device_tensor_mdspan_t<InputTensor>::rank();
  detail::dispatch_sum(std::forward<BackendSelector>(selector), output, input,
                       linalg::all_reduction_axes<input_rank>());
}

/// \brief Sum every device tensor view element into an existing scalar tensor using storage policy.
template <MutableScalarDeviceTensorView OutputTensor, DeviceTensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<InputTensor>>
void sum(OutputTensor&& output, InputTensor const& input)
{
  constexpr std::size_t input_rank = device_tensor_mdspan_t<InputTensor>::rank();
  auto operation = linalg::sum_reduction_op<input_rank, input_rank>{.axes = linalg::all_reduction_axes<input_rank>()};
  auto selector = linalg::select_backend(operation, output, input);
  detail::dispatch_sum(selector, output, input, operation.axes);
}

/// \brief Sum selected device tensor axes into an existing lower-rank tensor.
template <linalg::KernelBackendSelector BackendSelector, MutableDeviceTensorView OutputTensor,
          DeviceTensorView InputTensor, linalg::ReductionAxis FirstAxis, linalg::ReductionAxis... RestAxes>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<InputTensor>> &&
           (1 + sizeof...(RestAxes) <= device_tensor_mdspan_t<InputTensor>::rank()) &&
           (device_tensor_mdspan_t<OutputTensor>::rank() + 1 + sizeof...(RestAxes) ==
            device_tensor_mdspan_t<InputTensor>::rank())
void sum(BackendSelector&& selector, OutputTensor&& output, InputTensor const& input, FirstAxis first_axis,
         RestAxes... rest_axes)
{
  constexpr std::size_t input_rank = device_tensor_mdspan_t<InputTensor>::rank();
  auto axes = linalg::make_reduction_axes<input_rank>(first_axis, rest_axes...);
  prepare_output(output, detail::reduction_output_extents(input, axes));
  detail::dispatch_sum(std::forward<BackendSelector>(selector), output, input, std::move(axes));
}

/// \brief Sum selected device tensor axes into an existing output using storage policy.
template <MutableDeviceTensorView OutputTensor, DeviceTensorView InputTensor, linalg::ReductionAxis FirstAxis,
          linalg::ReductionAxis... RestAxes>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<InputTensor>> &&
           (1 + sizeof...(RestAxes) <= device_tensor_mdspan_t<InputTensor>::rank()) &&
           (device_tensor_mdspan_t<OutputTensor>::rank() + 1 + sizeof...(RestAxes) ==
            device_tensor_mdspan_t<InputTensor>::rank())
void sum(OutputTensor&& output, InputTensor const& input, FirstAxis first_axis, RestAxes... rest_axes)
{
  constexpr std::size_t input_rank = device_tensor_mdspan_t<InputTensor>::rank();
  auto axes = linalg::make_reduction_axes<input_rank>(first_axis, rest_axes...);
  auto operation = linalg::sum_reduction_op<input_rank, 1 + sizeof...(RestAxes)>{.axes = axes};
  auto selector = linalg::select_backend(operation, output, input);
  prepare_output(output, detail::reduction_output_extents(input, axes));
  detail::dispatch_sum(selector, output, input, std::move(axes));
}

/// \brief Return a storage-preserving full Tensor sum.
template <linalg::KernelBackendSelector BackendSelector, DeviceTensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           detail::ReductionResultAvailable<tensor_element_t<InputTensor>, InputTensor>
[[nodiscard]] auto sum(BackendSelector&& selector, InputTensor const& input)
{
  using result_type = ScalarTensor<tensor_element_t<InputTensor>, detail::reduction_result_storage_t<InputTensor>>;
  result_type result;
  sum(std::forward<BackendSelector>(selector), result, input);
  return result;
}

/// \brief Return a storage-preserving full Tensor sum using storage policy.
template <DeviceTensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           detail::ReductionResultAvailable<tensor_element_t<InputTensor>, InputTensor>
[[nodiscard]] auto sum(InputTensor const& input)
{
  constexpr std::size_t input_rank = device_tensor_mdspan_t<InputTensor>::rank();
  auto operation = linalg::sum_reduction_op<input_rank, input_rank>{.axes = linalg::all_reduction_axes<input_rank>()};
  auto selector = linalg::select_backend(operation, input);
  return sum(selector, input);
}

/// \brief Return a storage-preserving Tensor sum over selected axes.
template <linalg::KernelBackendSelector BackendSelector, DeviceTensorView InputTensor, linalg::ReductionAxis FirstAxis,
          linalg::ReductionAxis... RestAxes>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           (1 + sizeof...(RestAxes) <= device_tensor_mdspan_t<InputTensor>::rank()) &&
           detail::ReductionResultAvailable<tensor_element_t<InputTensor>, InputTensor>
[[nodiscard]] auto sum(BackendSelector&& selector, InputTensor const& input, FirstAxis first_axis,
                       RestAxes... rest_axes)
{
  constexpr std::size_t input_rank = device_tensor_mdspan_t<InputTensor>::rank();
  constexpr std::size_t reduced_rank = 1 + sizeof...(RestAxes);
  auto axes = linalg::make_reduction_axes<input_rank>(first_axis, rest_axes...);
  using result_type = detail::sum_reduction_result_t<InputTensor, reduced_rank>;
  result_type result(detail::reduction_output_extents(input, axes));
  detail::dispatch_sum(std::forward<BackendSelector>(selector), result, input, std::move(axes));
  return result;
}

/// \brief Return a storage-preserving Tensor sum over selected axes using storage policy.
template <DeviceTensorView InputTensor, linalg::ReductionAxis FirstAxis, linalg::ReductionAxis... RestAxes>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           (1 + sizeof...(RestAxes) <= device_tensor_mdspan_t<InputTensor>::rank()) &&
           detail::ReductionResultAvailable<tensor_element_t<InputTensor>, InputTensor>
[[nodiscard]] auto sum(InputTensor const& input, FirstAxis first_axis, RestAxes... rest_axes)
{
  constexpr std::size_t input_rank = device_tensor_mdspan_t<InputTensor>::rank();
  auto axes = linalg::make_reduction_axes<input_rank>(first_axis, rest_axes...);
  auto operation = linalg::sum_reduction_op<input_rank, 1 + sizeof...(RestAxes)>{.axes = axes};
  auto selector = linalg::select_backend(operation, input);
  return sum(selector, input, first_axis, rest_axes...);
}

/// \brief Return a full Tensor sum as a host C++ scalar through an explicit selector.
template <linalg::KernelBackendSelector BackendSelector, DeviceTensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>>
[[nodiscard]] auto sum_host(BackendSelector&& selector, InputTensor const& input) -> tensor_element_t<InputTensor>
{
  using result_type = tensor_element_t<InputTensor>;
  constexpr std::size_t input_rank = device_tensor_mdspan_t<InputTensor>::rank();
  result_type result{};
  detail::dispatch_sum(std::forward<BackendSelector>(selector), result, input,
                       linalg::all_reduction_axes<input_rank>());
  return result;
}

/// \brief Return a full Tensor sum as a host C++ scalar using storage policy.
template <DeviceTensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>>
[[nodiscard]] auto sum_host(InputTensor const& input) -> tensor_element_t<InputTensor>
{
  constexpr std::size_t input_rank = device_tensor_mdspan_t<InputTensor>::rank();
  auto operation = linalg::sum_reduction_op<input_rank, input_rank>{.axes = linalg::all_reduction_axes<input_rank>()};
  auto selector = linalg::select_backend(operation, input);
  return sum_host(selector, input);
}

/// \brief Compute an mdspan inner product into a rank-zero output.
/// \details The operation is conjugate-linear in `lhs` and linear in `rhs`.
/// \pre All input extents agree and output does not overlap an input.
template <class BackendSelector, MutableRankedMdspanLike<0> OutputSpan, class LhsSpan, class RhsSpan>
  requires detail::CompatibleInnerProductSpans<LhsSpan, RhsSpan> &&
           std::same_as<typename std::remove_cvref_t<OutputSpan>::value_type,
                        typename std::remove_cvref_t<LhsSpan>::value_type>
void inner_product(BackendSelector&& selector, OutputSpan&& output, LhsSpan&& lhs, RhsSpan&& rhs)
{
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::inner_product_op{},
                          std::forward<OutputSpan>(output), std::forward<LhsSpan>(lhs), std::forward<RhsSpan>(rhs));
}

/// \brief Return an mdspan inner product as a host C++ scalar.
/// \details The explicit selector determines how non-host inputs produce the
///          host-resident result.
template <class BackendSelector, class LhsSpan, class RhsSpan>
  requires detail::CompatibleInnerProductSpans<LhsSpan, RhsSpan>
[[nodiscard]] auto inner_product_host(BackendSelector&& selector, LhsSpan&& lhs, RhsSpan&& rhs) ->
    typename std::remove_cvref_t<LhsSpan>::value_type
{
  using result_type = typename std::remove_cvref_t<LhsSpan>::value_type;
  result_type result{};
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::inner_product_op{}, result,
                          std::forward<LhsSpan>(lhs), std::forward<RhsSpan>(rhs));
  return result;
}

/// \brief Compute a device tensor view inner product into an existing scalar tensor.
template <class BackendSelector, MutableScalarDeviceTensorView OutputTensor, class LhsTensor, class RhsTensor>
  requires detail::CompatibleInnerProductTensors<LhsTensor, RhsTensor> &&
           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<LhsTensor>>
void inner_product(BackendSelector&& selector, OutputTensor&& output, LhsTensor const& lhs, RhsTensor const& rhs)
{
  detail::require_reduction_extents(lhs, rhs);
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::inner_product_op{}, output, lhs, rhs);
}

/// \brief Compute a device tensor view inner product into an existing scalar tensor using storage policy.
template <MutableScalarDeviceTensorView OutputTensor, class LhsTensor, class RhsTensor>
  requires detail::CompatibleInnerProductTensors<LhsTensor, RhsTensor> &&
           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<LhsTensor>>
void inner_product(OutputTensor&& output, LhsTensor const& lhs, RhsTensor const& rhs)
{
  detail::require_reduction_extents(lhs, rhs);
  auto selector = linalg::select_backend(linalg::inner_product_op{}, output, lhs, rhs);
  inner_product(selector, output, lhs, rhs);
}

/// \brief Return a storage-preserving rank-zero Tensor inner product.
template <class BackendSelector, class LhsTensor, class RhsTensor>
  requires(!DeviceTensorView<BackendSelector>) && detail::CompatibleInnerProductTensors<LhsTensor, RhsTensor> &&
          detail::ReductionResultAvailable<tensor_element_t<LhsTensor>, LhsTensor, RhsTensor>
[[nodiscard]] auto inner_product(BackendSelector&& selector, LhsTensor const& lhs, RhsTensor const& rhs)
{
  using result_type =
      ScalarTensor<tensor_element_t<LhsTensor>, detail::reduction_result_storage_t<LhsTensor, RhsTensor>>;
  result_type result;
  inner_product(std::forward<BackendSelector>(selector), result, lhs, rhs);
  return result;
}

/// \brief Return a storage-preserving rank-zero Tensor inner product using storage policy.
template <class LhsTensor, class RhsTensor>
  requires detail::CompatibleInnerProductTensors<LhsTensor, RhsTensor> &&
           detail::ReductionResultAvailable<tensor_element_t<LhsTensor>, LhsTensor, RhsTensor>
[[nodiscard]] auto inner_product(LhsTensor const& lhs, RhsTensor const& rhs)
{
  detail::require_reduction_extents(lhs, rhs);
  auto selector = linalg::select_backend(linalg::inner_product_op{}, lhs, rhs);
  return inner_product(selector, lhs, rhs);
}

/// \brief Return a Tensor inner product as a host C++ scalar through an explicit selector.
template <class BackendSelector, class LhsTensor, class RhsTensor>
  requires detail::CompatibleInnerProductTensors<LhsTensor, RhsTensor>
[[nodiscard]] auto inner_product_host(BackendSelector&& selector, LhsTensor const& lhs,
                                      RhsTensor const& rhs) -> tensor_element_t<LhsTensor>
{
  detail::require_reduction_extents(lhs, rhs);
  using result_type = tensor_element_t<LhsTensor>;
  result_type result{};
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::inner_product_op{}, result, lhs, rhs);
  return result;
}

/// \brief Return a Tensor inner product as a host C++ scalar using storage policy.
template <class LhsTensor, class RhsTensor>
  requires detail::CompatibleInnerProductTensors<LhsTensor, RhsTensor>
[[nodiscard]] auto inner_product_host(LhsTensor const& lhs, RhsTensor const& rhs) -> tensor_element_t<LhsTensor>
{
  detail::require_reduction_extents(lhs, rhs);
  auto selector = linalg::select_backend(linalg::inner_product_op{}, lhs, rhs);
  return inner_product_host(selector, lhs, rhs);
}

/// \brief Compute an mdspan Euclidean norm into a real rank-zero output.
template <class BackendSelector, MutableRankedMdspanLike<0> OutputSpan, MdspanLike InputSpan>
  requires RealOrComplex<typename std::remove_cvref_t<InputSpan>::value_type> &&
           std::same_as<typename std::remove_cvref_t<OutputSpan>::value_type,
                        make_real_t<typename std::remove_cvref_t<InputSpan>::value_type>>
void norm(BackendSelector&& selector, OutputSpan&& output, InputSpan&& input)
{
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::norm_op{}, std::forward<OutputSpan>(output),
                          std::forward<InputSpan>(input));
}

/// \brief Return an mdspan Euclidean norm as a host C++ scalar.
template <class BackendSelector, MdspanLike InputSpan>
  requires RealOrComplex<typename std::remove_cvref_t<InputSpan>::value_type>
[[nodiscard]] auto norm_host(BackendSelector&& selector,
                             InputSpan&& input) -> make_real_t<typename std::remove_cvref_t<InputSpan>::value_type>
{
  using result_type = make_real_t<typename std::remove_cvref_t<InputSpan>::value_type>;
  result_type result{};
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::norm_op{}, result,
                          std::forward<InputSpan>(input));
  return result;
}

/// \brief Compute a device tensor view Euclidean norm into an existing real scalar tensor.
template <class BackendSelector, MutableScalarDeviceTensorView OutputTensor, DeviceTensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           std::same_as<tensor_element_t<OutputTensor>, make_real_t<tensor_element_t<InputTensor>>>
void norm(BackendSelector&& selector, OutputTensor&& output, InputTensor const& input)
{
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::norm_op{}, output, input);
}

/// \brief Compute a device tensor view Euclidean norm into an existing real scalar tensor using storage policy.
template <MutableScalarDeviceTensorView OutputTensor, DeviceTensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           std::same_as<tensor_element_t<OutputTensor>, make_real_t<tensor_element_t<InputTensor>>>
void norm(OutputTensor&& output, InputTensor const& input)
{
  auto selector = linalg::select_backend(linalg::norm_op{}, output, input);
  norm(selector, output, input);
}

/// \brief Return a storage-preserving real rank-zero Tensor Euclidean norm.
template <class BackendSelector, DeviceTensorView InputTensor>
  requires(!DeviceTensorView<BackendSelector>) && RealOrComplex<tensor_element_t<InputTensor>> &&
          detail::ReductionResultAvailable<make_real_t<tensor_element_t<InputTensor>>, InputTensor>
[[nodiscard]] auto norm(BackendSelector&& selector, InputTensor const& input)
{
  using result_type =
      ScalarTensor<make_real_t<tensor_element_t<InputTensor>>, detail::reduction_result_storage_t<InputTensor>>;
  result_type result;
  norm(std::forward<BackendSelector>(selector), result, input);
  return result;
}

/// \brief Return a storage-preserving real rank-zero Tensor Euclidean norm using storage policy.
template <DeviceTensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           detail::ReductionResultAvailable<make_real_t<tensor_element_t<InputTensor>>, InputTensor>
[[nodiscard]] auto norm(InputTensor const& input)
{
  auto selector = linalg::select_backend(linalg::norm_op{}, input);
  return norm(selector, input);
}

/// \brief Return a Tensor Euclidean norm as a host C++ scalar through an explicit selector.
template <class BackendSelector, DeviceTensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>>
[[nodiscard]] auto norm_host(BackendSelector&& selector,
                             InputTensor const& input) -> make_real_t<tensor_element_t<InputTensor>>
{
  using result_type = make_real_t<tensor_element_t<InputTensor>>;
  result_type result{};
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::norm_op{}, result, input);
  return result;
}

/// \brief Return a Tensor Euclidean norm as a host C++ scalar using storage policy.
template <DeviceTensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>>
[[nodiscard]] auto norm_host(InputTensor const& input) -> make_real_t<tensor_element_t<InputTensor>>
{
  auto selector = linalg::select_backend(linalg::norm_op{}, input);
  return norm_host(selector, input);
}

} // namespace uni20
