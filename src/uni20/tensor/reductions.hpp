#pragma once

/**
 * \file reductions.hpp
 * \ingroup tensor
 * \brief Tensor inner products and Euclidean norms with explicit result residency.
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
concept CompatibleInnerProductSpans = SpanLike<LhsSpan> && SpanLike<RhsSpan> &&
                                      (std::remove_cvref_t<LhsSpan>::rank() == std::remove_cvref_t<RhsSpan>::rank()) &&
                                      std::same_as<typename std::remove_cvref_t<LhsSpan>::value_type,
                                                   typename std::remove_cvref_t<RhsSpan>::value_type> &&
                                      RealOrComplex<typename std::remove_cvref_t<LhsSpan>::value_type>;

template <class LhsTensor, class RhsTensor>
concept CompatibleInnerProductTensors = TensorView<LhsTensor> && TensorView<RhsTensor> &&
                                        (tensor_mdspan_t<LhsTensor>::rank() == tensor_mdspan_t<RhsTensor>::rank()) &&
                                        std::same_as<tensor_element_t<LhsTensor>, tensor_element_t<RhsTensor>> &&
                                        RealOrComplex<tensor_element_t<LhsTensor>>;

template <class Tensor>
concept ReductionStorageTypedTensor =
    TensorView<Tensor> && requires { typename std::remove_cvref_t<Tensor>::storage_policy; };

template <class... Inputs> struct reduction_result_storage
{
    using selected_type = linalg::detail::first_backend_bound_storage_t<Inputs...>;
    using type = std::conditional_t<std::is_void_v<selected_type>, VectorStorage, selected_type>;
};

template <class... Inputs> using reduction_result_storage_t = typename reduction_result_storage<Inputs...>::type;

template <class Element, class... Inputs>
concept ScalarReductionResultAvailable = (ReductionStorageTypedTensor<Inputs> && ...) && requires {
  typename reduction_result_storage_t<Inputs...>::template storage_t<Element>;
};

template <class Reference, class... Others>
void require_reduction_extents(Reference const& reference, Others const&... others)
{
  constexpr std::size_t rank = tensor_mdspan_t<Reference>::rank();
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

/// \brief Compute an mdspan inner product into a rank-zero output.
/// \details The operation is conjugate-linear in `lhs` and linear in `rhs`.
/// \pre All input extents agree and output does not overlap an input.
template <class BackendSelector, MutableRankedSpanLike<0> OutputSpan, class LhsSpan, class RhsSpan>
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

/// \brief Compute a Tensor inner product into an existing scalar tensor.
template <class BackendSelector, MutableScalarTensorView OutputTensor, class LhsTensor, class RhsTensor>
  requires detail::CompatibleInnerProductTensors<LhsTensor, RhsTensor> &&
           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<LhsTensor>>
void inner_product(BackendSelector&& selector, OutputTensor&& output, LhsTensor const& lhs, RhsTensor const& rhs)
{
  detail::require_reduction_extents(lhs, rhs);
  inner_product(std::forward<BackendSelector>(selector), output.mdspan(), lhs.mdspan(), rhs.mdspan());
}

/// \brief Compute a Tensor inner product into an existing scalar tensor using storage policy.
template <MutableScalarTensorView OutputTensor, class LhsTensor, class RhsTensor>
  requires detail::CompatibleInnerProductTensors<LhsTensor, RhsTensor> &&
           std::same_as<tensor_element_t<OutputTensor>, tensor_element_t<LhsTensor>>
void inner_product(OutputTensor&& output, LhsTensor const& lhs, RhsTensor const& rhs)
{
  detail::require_reduction_extents(lhs, rhs);
  auto selector = linalg::select_backend(linalg::inner_product_op{}, output, lhs, rhs);
  inner_product(selector, output.mdspan(), lhs.mdspan(), rhs.mdspan());
}

/// \brief Return a storage-preserving rank-zero Tensor inner product.
template <class BackendSelector, class LhsTensor, class RhsTensor>
  requires(!TensorView<BackendSelector>) && detail::CompatibleInnerProductTensors<LhsTensor, RhsTensor> &&
          detail::ScalarReductionResultAvailable<tensor_element_t<LhsTensor>, LhsTensor, RhsTensor>
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
           detail::ScalarReductionResultAvailable<tensor_element_t<LhsTensor>, LhsTensor, RhsTensor>
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
  return inner_product_host(std::forward<BackendSelector>(selector), lhs.mdspan(), rhs.mdspan());
}

/// \brief Return a Tensor inner product as a host C++ scalar using storage policy.
template <class LhsTensor, class RhsTensor>
  requires detail::CompatibleInnerProductTensors<LhsTensor, RhsTensor>
[[nodiscard]] auto inner_product_host(LhsTensor const& lhs, RhsTensor const& rhs) -> tensor_element_t<LhsTensor>
{
  detail::require_reduction_extents(lhs, rhs);
  auto selector = linalg::select_backend(linalg::inner_product_op{}, lhs, rhs);
  return inner_product_host(selector, lhs.mdspan(), rhs.mdspan());
}

/// \brief Compute an mdspan Euclidean norm into a real rank-zero output.
template <class BackendSelector, MutableRankedSpanLike<0> OutputSpan, SpanLike InputSpan>
  requires RealOrComplex<typename std::remove_cvref_t<InputSpan>::value_type> &&
           std::same_as<typename std::remove_cvref_t<OutputSpan>::value_type,
                        make_real_t<typename std::remove_cvref_t<InputSpan>::value_type>>
void norm(BackendSelector&& selector, OutputSpan&& output, InputSpan&& input)
{
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::norm_op{}, std::forward<OutputSpan>(output),
                          std::forward<InputSpan>(input));
}

/// \brief Return an mdspan Euclidean norm as a host C++ scalar.
template <class BackendSelector, SpanLike InputSpan>
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

/// \brief Compute a Tensor Euclidean norm into an existing real scalar tensor.
template <class BackendSelector, MutableScalarTensorView OutputTensor, TensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           std::same_as<tensor_element_t<OutputTensor>, make_real_t<tensor_element_t<InputTensor>>>
void norm(BackendSelector&& selector, OutputTensor&& output, InputTensor const& input)
{
  norm(std::forward<BackendSelector>(selector), output.mdspan(), input.mdspan());
}

/// \brief Compute a Tensor Euclidean norm into an existing real scalar tensor using storage policy.
template <MutableScalarTensorView OutputTensor, TensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           std::same_as<tensor_element_t<OutputTensor>, make_real_t<tensor_element_t<InputTensor>>>
void norm(OutputTensor&& output, InputTensor const& input)
{
  auto selector = linalg::select_backend(linalg::norm_op{}, output, input);
  norm(selector, output.mdspan(), input.mdspan());
}

/// \brief Return a storage-preserving real rank-zero Tensor Euclidean norm.
template <class BackendSelector, TensorView InputTensor>
  requires(!TensorView<BackendSelector>) && RealOrComplex<tensor_element_t<InputTensor>> &&
          detail::ScalarReductionResultAvailable<make_real_t<tensor_element_t<InputTensor>>, InputTensor>
[[nodiscard]] auto norm(BackendSelector&& selector, InputTensor const& input)
{
  using result_type =
      ScalarTensor<make_real_t<tensor_element_t<InputTensor>>, detail::reduction_result_storage_t<InputTensor>>;
  result_type result;
  norm(std::forward<BackendSelector>(selector), result, input);
  return result;
}

/// \brief Return a storage-preserving real rank-zero Tensor Euclidean norm using storage policy.
template <TensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>> &&
           detail::ScalarReductionResultAvailable<make_real_t<tensor_element_t<InputTensor>>, InputTensor>
[[nodiscard]] auto norm(InputTensor const& input)
{
  auto selector = linalg::select_backend(linalg::norm_op{}, input);
  return norm(selector, input);
}

/// \brief Return a Tensor Euclidean norm as a host C++ scalar through an explicit selector.
template <class BackendSelector, TensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>>
[[nodiscard]] auto norm_host(BackendSelector&& selector,
                             InputTensor const& input) -> make_real_t<tensor_element_t<InputTensor>>
{
  return norm_host(std::forward<BackendSelector>(selector), input.mdspan());
}

/// \brief Return a Tensor Euclidean norm as a host C++ scalar using storage policy.
template <TensorView InputTensor>
  requires RealOrComplex<tensor_element_t<InputTensor>>
[[nodiscard]] auto norm_host(InputTensor const& input) -> make_real_t<tensor_element_t<InputTensor>>
{
  auto selector = linalg::select_backend(linalg::norm_op{}, input);
  return norm_host(selector, input.mdspan());
}

} // namespace uni20
