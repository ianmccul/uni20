#pragma once

/**
 * \file conjugate_inplace.hpp
 * \ingroup tensor
 * \brief Backend-dispatched in-place conjugation of tensor elements.
 */

#include <uni20/linalg/backend_selector.hpp>
#include <uni20/linalg/backends/cpu/conjugate_inplace.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/concepts.hpp>

#include <utility>

namespace uni20
{

/// \brief Conjugate a mutable mdspan-like object through an explicit backend selector.
template <class BackendSelector, MutableMdspanLike Mdspan>
void conjugate_inplace(BackendSelector&& selector, Mdspan&& span)
{
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::conjugate_inplace_op{},
                          std::forward<Mdspan>(span));
}

/// \brief Conjugate a mutable tensor through an explicit backend selector.
template <class BackendSelector, MutableTensorView Tensor>
void conjugate_inplace(BackendSelector&& selector, Tensor&& tensor)
{
  auto span = tensor.mdspan();
  conjugate_inplace(std::forward<BackendSelector>(selector), span);
}

/// \brief Conjugate a mutable tensor using its storage policy's backend selector.
template <MutableTensorView Tensor> void conjugate_inplace(Tensor&& tensor)
{
  auto operation = linalg::conjugate_inplace_op{};
  auto selector = linalg::select_backend(operation, tensor);
  auto span = tensor.mdspan();
  conjugate_inplace(selector, span);
}

} // namespace uni20
