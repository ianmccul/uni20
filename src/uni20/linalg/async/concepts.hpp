#pragma once

/**
 * \file concepts.hpp
 * \ingroup linalg
 * \brief Capabilities shared by asynchronous Tensor operation wrappers.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/tensor/concepts.hpp>

#include <concepts>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Mutable Tensor output supported by async operation wrappers.
/// \details Async aliases must be copyable because their writer exposes a
///          const bound descriptor that the coroutine copies before dispatch.
template <class Tensor>
concept AsyncTensorOutput =
    MutableTensorView<Tensor> && (!async::is_async_alias_v<Tensor> || std::copy_constructible<Tensor>);

} // namespace uni20

namespace uni20::linalg
{
namespace detail
{

template <class Scalar>
using async_operation_scalar_awaiter_t = std::remove_cvref_t<decltype(async::read(std::declval<Scalar>()))>;

} // namespace detail

/// \brief Scalar argument that an async wrapper can await and convert.
template <class Value, class Scalar>
concept AsyncOperationScalar = async::TaskAwaitable<detail::async_operation_scalar_awaiter_t<Value>> &&
                               requires(detail::async_operation_scalar_awaiter_t<Value>& awaiter) {
                                 { awaiter.await_resume() } -> std::convertible_to<Scalar>;
                               };

} // namespace uni20::linalg
