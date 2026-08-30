#pragma once

/**
 * \file matrix_norm.hpp
 * \ingroup linalg
 * \brief Dense matrix norm front ends over operation-tag dispatch.
 */

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_traits.hpp>
#include <uni20/linalg/backends/cpu/matrix_norm.hpp>
#include <uni20/linalg/backends/lapack/matrix_norm.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/reductions.hpp>

#include <concepts>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{

/// \brief Compute a bare mdspan matrix norm into a rank-zero output.
template <KernelBackendSelector BackendSelector, uni20::MutableRankedMdspanLike<0> OutputMdspan,
          uni20::RankedMdspanLike<2> MatrixMdspan>
  requires uni20::RealOrComplex<typename std::remove_cvref_t<MatrixMdspan>::value_type> &&
           std::same_as<typename std::remove_cvref_t<OutputMdspan>::value_type,
                        uni20::make_real_t<typename std::remove_cvref_t<MatrixMdspan>::value_type>>
void matrix_norm(BackendSelector&& selector, OutputMdspan&& output, MatrixMdspan&& matrix, MatrixNorm kind)
{
  auto output_descriptor = std::forward<OutputMdspan>(output);
  auto matrix_descriptor = uni20::make_const_mdspan(matrix);
  dispatch_kernel(std::forward<BackendSelector>(selector), matrix_norm_op{.kind = kind}, output_descriptor,
                  matrix_descriptor);
}

/// \brief Return a bare mdspan matrix norm as a host C++ scalar.
template <KernelBackendSelector BackendSelector, uni20::RankedMdspanLike<2> MatrixMdspan>
  requires uni20::RealOrComplex<typename std::remove_cvref_t<MatrixMdspan>::value_type>
[[nodiscard]] auto matrix_norm_host(BackendSelector&& selector, MatrixMdspan&& matrix, MatrixNorm kind)
    -> uni20::make_real_t<typename std::remove_cvref_t<MatrixMdspan>::value_type>
{
  using result_type = uni20::make_real_t<typename std::remove_cvref_t<MatrixMdspan>::value_type>;
  result_type result{};
  auto matrix_descriptor = uni20::make_const_mdspan(matrix);
  dispatch_kernel(std::forward<BackendSelector>(selector), matrix_norm_op{.kind = kind}, result, matrix_descriptor);
  return result;
}

/// \brief Compute a Tensor matrix norm into an existing real scalar Tensor.
template <KernelBackendSelector BackendSelector, uni20::MutableScalarTensorView OutputTensor,
          uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<MatrixTensor>> &&
           std::same_as<uni20::tensor_element_t<OutputTensor>,
                        uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>>
void matrix_norm(BackendSelector&& selector, OutputTensor&& output, MatrixTensor const& matrix, MatrixNorm kind)
{
  auto output_descriptor = uni20::mdspec_of(output);
  auto matrix_descriptor = uni20::mdspec_of(matrix);
  dispatch_kernel(std::forward<BackendSelector>(selector), matrix_norm_op{.kind = kind}, output_descriptor,
                  matrix_descriptor);
}

/// \brief Compute a Tensor matrix norm using the operands' storage policy.
template <uni20::MutableScalarTensorView OutputTensor, uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<MatrixTensor>> &&
           std::same_as<uni20::tensor_element_t<OutputTensor>,
                        uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>>
void matrix_norm(OutputTensor&& output, MatrixTensor const& matrix, MatrixNorm kind)
{
  auto operation = matrix_norm_op{.kind = kind};
  auto selector = select_backend(operation, output, matrix);
  matrix_norm(selector, std::forward<OutputTensor>(output), matrix, kind);
}

/// \brief Return a storage-preserving rank-zero Tensor matrix norm.
template <KernelBackendSelector BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<MatrixTensor>> &&
           requires(MatrixTensor const& matrix) { uni20::norm(matrix); }
[[nodiscard]] auto matrix_norm(BackendSelector&& selector, MatrixTensor const& matrix, MatrixNorm kind)
{
  using result_type = decltype(uni20::norm(matrix));
  result_type result;
  matrix_norm(std::forward<BackendSelector>(selector), result, matrix, kind);
  return result;
}

/// \brief Return a storage-preserving rank-zero Tensor matrix norm using storage policy.
template <uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<MatrixTensor>> &&
           requires(MatrixTensor const& matrix) { uni20::norm(matrix); }
[[nodiscard]] auto matrix_norm(MatrixTensor const& matrix, MatrixNorm kind)
{
  auto selector = select_backend(matrix_norm_op{.kind = kind}, matrix);
  return matrix_norm(selector, matrix, kind);
}

/// \brief Return a Tensor matrix norm as a host C++ scalar through an explicit selector.
template <KernelBackendSelector BackendSelector, uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto matrix_norm_host(BackendSelector&& selector, MatrixTensor const& matrix,
                                    MatrixNorm kind) -> uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>
{
  using result_type = uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>;
  result_type result{};
  auto matrix_descriptor = uni20::mdspec_of(matrix);
  dispatch_kernel(std::forward<BackendSelector>(selector), matrix_norm_op{.kind = kind}, result, matrix_descriptor);
  return result;
}

/// \brief Return a Tensor matrix norm as a host C++ scalar using storage policy.
template <uni20::RankedTensorView<2> MatrixTensor>
  requires uni20::RealOrComplex<uni20::tensor_element_t<MatrixTensor>>
[[nodiscard]] auto matrix_norm_host(MatrixTensor const& matrix,
                                    MatrixNorm kind) -> uni20::make_real_t<uni20::tensor_element_t<MatrixTensor>>
{
  auto selector = select_backend(matrix_norm_op{.kind = kind}, matrix);
  return matrix_norm_host(selector, matrix, kind);
}

} // namespace uni20::linalg
