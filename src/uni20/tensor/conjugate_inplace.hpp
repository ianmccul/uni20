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
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#if UNI20_BACKEND_CUDA
#include <uni20/linalg/backends/cuda/conjugate_inplace.hpp>
#endif

#include <utility>

namespace uni20
{

/// \brief Conjugate a mutable mdspec-like object through an explicit backend selector.
template <class BackendSelector, MutableMdspecLike Mdspec>
void conjugate_inplace(BackendSelector&& selector, Mdspec&& span)
{
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::conjugate_inplace_op{},
                          std::forward<Mdspec>(span));
}

/// \brief Conjugate a mutable tensor view through an explicit backend selector.
template <class BackendSelector, MutableTensorView Tensor>
void conjugate_inplace(BackendSelector&& selector, Tensor&& tensor)
{
  auto descriptor = mdspec_of(tensor);
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::conjugate_inplace_op{}, descriptor);
}

/// \brief Conjugate a mutable tensor view using its storage policy's backend selector.
template <MutableTensorView Tensor> void conjugate_inplace(Tensor&& tensor)
{
  auto operation = linalg::conjugate_inplace_op{};
  auto selector = linalg::select_backend(operation, tensor);
  conjugate_inplace(selector, std::forward<Tensor>(tensor));
}

} // namespace uni20
