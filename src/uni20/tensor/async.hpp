#pragma once

/**
 * \file async.hpp
 * \ingroup tensor
 * \brief Async aliases for lazy tensor-level views.
 */

#include <uni20/async/async.hpp>
#include <uni20/tensor/conjugate.hpp>
#include <uni20/tensor/copy.hpp>

namespace uni20::async
{

/// \brief Return a lazy conjugating alias of an async complex tensor.
/// \details The result retains the parent tensor storage and shares its exact
///          epoch queue. Awaiting the alias therefore observes the same causal
///          timeline as awaiting the parent.
template <uni20::TensorView Tensor>
  requires uni20::Complex<uni20::tensor_element_t<Tensor>>
[[nodiscard]] auto conj(Async<Tensor> const& tensor)
{
  using view_type = uni20::ConjugatedTensorView<Tensor>;
  return make_async_alias<view_type>(tensor, tensor.storage().storage_address());
}

/// \brief Return a lazy read-only identity alias of an async real tensor.
template <uni20::TensorView Tensor>
  requires(!uni20::Complex<uni20::tensor_element_t<Tensor>>)
[[nodiscard]] auto conj(Async<Tensor> const& tensor)
{
  using view_type = uni20::ConstTensorView<Tensor>;
  return make_async_alias<view_type>(tensor, tensor.storage().storage_address());
}

} // namespace uni20::async
