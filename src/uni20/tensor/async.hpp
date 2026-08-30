#pragma once

/**
 * \file async.hpp
 * \ingroup tensor
 * \brief Async aliases for lazy tensor-level views.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/tensor/conjugate.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/reshape.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

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

/// \brief Return a mutable structural reshape alias of an async tensor.
/// \details The descriptor retains the parent storage and shares its exact
///          epoch queue. Shape and layout validation occurs when the shared
///          parent epoch first becomes readable.
template <uni20::MutableStridedTensorView Tensor, std::integral... Extents>
  requires uni20::detail::CanonicalReshapeLayout<typename uni20::tensor_mdspec_t<Tensor>::layout_type>
[[nodiscard]] auto reshape_view(Async<Tensor>& tensor, Extents... requested_extents)
{
  using layout_type = typename uni20::tensor_mdspec_t<Tensor>::layout_type;
  using view_type = uni20::IndirectReshapedTensorView<Tensor, layout_type, std::remove_cvref_t<Extents>...>;
  return make_async_alias_from_parent<view_type>(tensor, requested_extents...);
}

/// \brief Return a read-only structural reshape alias of an async tensor.
template <uni20::StridedTensorView Tensor, std::integral... Extents>
  requires uni20::detail::CanonicalReshapeLayout<typename uni20::tensor_mdspec_t<Tensor>::layout_type>
[[nodiscard]] auto reshape_view(Async<Tensor> const& tensor, Extents... requested_extents)
{
  using layout_type = typename uni20::tensor_mdspec_t<Tensor>::layout_type;
  using view_type = uni20::IndirectReshapedTensorView<Tensor const, layout_type, std::remove_cvref_t<Extents>...>;
  return make_async_alias_from_parent<view_type>(tensor, requested_extents...);
}

/// \brief Return a mutable column-major structural reshape alias.
template <uni20::MutableStridedTensorView Tensor, std::integral... Extents>
[[nodiscard]] auto reshape_view_left(Async<Tensor>& tensor, Extents... requested_extents)
{
  using view_type = uni20::IndirectReshapedTensorView<Tensor, stdex::layout_left, std::remove_cvref_t<Extents>...>;
  return make_async_alias_from_parent<view_type>(tensor, requested_extents...);
}

/// \brief Return a read-only column-major structural reshape alias.
template <uni20::StridedTensorView Tensor, std::integral... Extents>
[[nodiscard]] auto reshape_view_left(Async<Tensor> const& tensor, Extents... requested_extents)
{
  using view_type =
      uni20::IndirectReshapedTensorView<Tensor const, stdex::layout_left, std::remove_cvref_t<Extents>...>;
  return make_async_alias_from_parent<view_type>(tensor, requested_extents...);
}

/// \brief Return a mutable row-major structural reshape alias.
template <uni20::MutableStridedTensorView Tensor, std::integral... Extents>
[[nodiscard]] auto reshape_view_right(Async<Tensor>& tensor, Extents... requested_extents)
{
  using view_type = uni20::IndirectReshapedTensorView<Tensor, stdex::layout_right, std::remove_cvref_t<Extents>...>;
  return make_async_alias_from_parent<view_type>(tensor, requested_extents...);
}

/// \brief Return a read-only row-major structural reshape alias.
template <uni20::StridedTensorView Tensor, std::integral... Extents>
[[nodiscard]] auto reshape_view_right(Async<Tensor> const& tensor, Extents... requested_extents)
{
  using view_type =
      uni20::IndirectReshapedTensorView<Tensor const, stdex::layout_right, std::remove_cvref_t<Extents>...>;
  return make_async_alias_from_parent<view_type>(tensor, requested_extents...);
}

} // namespace uni20::async

namespace uni20
{
namespace detail
{
template <class Tensor, class... Extents>
using preserving_async_reshape_result_t =
    std::remove_cvref_t<decltype(uni20::reshape(std::declval<Tensor const&>(), std::declval<Extents>()...))>;

template <class Tensor, class... Extents>
using consuming_async_reshape_result_t =
    std::remove_cvref_t<decltype(uni20::reshape(std::declval<Tensor&&>(), std::declval<Extents>()...))>;

template <class ResultTensor, TensorView InputTensor, class... Extents>
async::AsyncTask co_preserving_reshape(async::WriteBuffer<ResultTensor> output, async::ReadBuffer<InputTensor> input,
                                       Extents... requested_extents)
{
  auto output_storage = output.storage();
  auto awaited = co_await async::all(output_storage, input);
  auto& storage = std::get<0>(awaited);
  auto const& input_value = std::get<1>(awaited);
  storage.emplace(uni20::reshape(input_value, requested_extents...));
  co_return;
}

template <class ResultTensor, class InputTensor, class... Extents>
  requires OwningTensor<InputTensor> && MutableTensorView<InputTensor>
async::AsyncTask co_consuming_reshape(async::WriteBuffer<ResultTensor> output, async::WriteBuffer<InputTensor> input,
                                      Extents... requested_extents)
{
  auto output_storage = output.storage();
  auto& storage = co_await output_storage;
  auto input_value = co_await input.take();
  storage.emplace(uni20::reshape(std::move(input_value), requested_extents...));
  co_return;
}

template <class Tensor, class... Extents>
async::AsyncTask co_reshape_inplace(async::WriteBuffer<Tensor> input, Extents... requested_extents)
{
  auto storage_awaiter = input.storage();
  auto& storage = co_await storage_awaiter;
  if (!storage.constructed()) throw async::buffer_write_uninitialized{};
  uni20::reshape_inplace(*storage, requested_extents...);
  co_return;
}
} // namespace detail

/// \brief Schedule a preserving owning reshape of an async Tensor.
/// \details The result has an independent epoch and matches synchronous
///          `reshape(input, extents...)`, including materialization when needed.
template <TensorView InputTensor, std::integral... Extents>
  requires requires(InputTensor const& input, Extents... extents) { uni20::reshape(input, extents...); }
[[nodiscard]] auto reshape(async::Async<InputTensor> const& input, Extents... requested_extents)
{
  using result_type = detail::preserving_async_reshape_result_t<InputTensor, Extents...>;
  async::Async<result_type> output;
  output.debug_name("reshape.result");
  auto task = detail::co_preserving_reshape(output.write(), input.read(), requested_extents...);
  task.debug_name("reshape");
  async::schedule(std::move(task));
  return output;
}

/// \brief Schedule an owning reshape that consumes an async Tensor value.
/// \details Passing the async input as an rvalue grants permission to remove
///          and transfer its stored allocation when the synchronous reshape can do so.
template <class InputTensor, std::integral... Extents>
  requires OwningTensor<InputTensor> && MutableTensorView<InputTensor> &&
           requires(InputTensor&& input, Extents... extents) { uni20::reshape(std::move(input), extents...); }
[[nodiscard]] auto reshape(async::Async<InputTensor>&& input, Extents... requested_extents)
{
  using result_type = detail::consuming_async_reshape_result_t<InputTensor, Extents...>;
  async::Async<result_type> output;
  output.debug_name("reshape.result");
  auto task = detail::co_consuming_reshape(output.write(), input.write(), requested_extents...);
  task.debug_name("reshape");
  async::schedule(std::move(task));
  return output;
}

/// \brief Schedule an in-place mapping change on an async owning Tensor.
/// \details The operation retains the allocation and publishes metadata mutation
///          through one writer epoch.
template <class Tensor, std::integral... Extents>
  requires requires(Tensor& tensor, Extents... extents) { uni20::reshape_inplace(tensor, extents...); }
void reshape_inplace(async::Async<Tensor>& input, Extents... requested_extents)
{
  auto task = detail::co_reshape_inplace(input.write(), requested_extents...);
  task.debug_name("reshape_inplace");
  async::schedule(std::move(task));
}

} // namespace uni20
