// async_ops.hpp
#pragma once

#include "async.hpp"
#include "async_task.hpp"
#include "awaiters.hpp"
#include "debug_scheduler.hpp"
#include <functional>
#include <type_traits>

namespace uni20::async
{

/// \defgroup async_api Asynchronous Expression System
/// @{

/// \brief Trait: true if any argument is an Async<T>
/// \ingroup async_api
template <typename T> struct is_async : std::false_type
{};

template <typename T> struct is_async<Async<T>> : std::true_type
{};

template <typename T> constexpr bool is_async_v = is_async<std::remove_cvref_t<T>>::value;

/// \brief Awaitable wrapper for scalar values used in async expressions.
///
/// Allows scalar values to participate in co_await expressions used by async kernels.
/// This awaitable never suspends, and simply returns the value when resumed.
/// We store a value here, which means that we necessarily copy (or move).
/// This is fine, since we have have an expression involving an Async<T> x, and a concrete value y,
/// we must copy (or move) y into the coroutine / scheduler anyway.
///
/// \tparam T Scalar type
template <typename T> struct ValueAwaiter
{
    T value;

    /// \brief Always ready — never suspends.
    bool await_ready() const noexcept { return true; }

    /// \brief No-op suspension hook.
    void await_suspend(AsyncTask) const noexcept {}

    /// \brief Return the value.
    T const& await_resume() const& noexcept { return value; }
    T await_resume() && noexcept { return std::move(value); }

    // no-op, simulating a ReadBuffer
    void release() const noexcept {};
};

/// \brief Read adapter for scalar values.
///
/// Wraps a scalar value into an awaitable that can be co_awaited like an Async buffer.
/// \ingroup async_api
template <typename T>
  requires(!is_async_v<T>)
ValueAwaiter<std::remove_cvref_t<T>> read(T&& x)
{
  return ValueAwaiter<std::remove_cvref_t<T>>{std::forward<T>(x)};
}

template <typename... Args>
constexpr bool contains_an_async = (false || ... || is_async<std::remove_cvref_t<Args>>::value);

/// \brief Strip Async<T> to T for type deduction
/// \ingroup async_api
template <typename T> struct async_value_type
{
    using type = T;
};

template <typename T> struct async_value_type<Async<T>>
{
    using type = T;
};

/// \brief Extract the underlying value type from scalar or Async<T>
/// \ingroup async_api
template <typename T> using async_value_t = typename async_value_type<std::remove_cvref_t<T>>::type;

/// \brief Result type of applying binary op Op to async_value_t<T>, async_value_t<U>
/// \ingroup async_api
template <typename T, typename U, typename Op>
using async_binary_result_t =
    decltype(std::declval<Op>()(std::declval<async_value_t<T>>(), std::declval<async_value_t<U>>()));

template <typename T, typename U, typename Op> struct is_async_binary_applicable : std::false_type
{};

template <typename T, typename U, typename Op>
  requires contains_an_async<T, U> && requires(Op op, async_value_t<T> t, async_value_t<U> u) {
                                        {
                                          op(t, u)
                                        };
                                      }
struct is_async_binary_applicable<T, U, Op> : std::true_type
{};

template <typename T, typename U, typename Op>
constexpr bool is_async_binary_applicable_v = is_async_binary_applicable<T, U, Op>::value;

// Example using move semantics
// template <typename A, typename B, typename R, typename Op> void async_binary_op(A&& a, B&& b, Async<R>& out, Op op)
// {
//   schedule([](auto a_, auto b_, WriteBuffer<R> out_, Op op_) -> AsyncTask {
//     auto [va, vb] = co_await all(std::move(a_), std::move(b_));
//     a_.release();
//     b_.release();
//     R tmp = op_(std::move(va), std::move(vb)); // move values into op
//     auto& outval = co_await out_;
//     outval = std::move(tmp);
//     co_return;
//   }(read(std::forward<A>(a)), read(std::forward<B>(b)), out.write(), std::move(op)));
// }

/// \brief Launch a coroutine computing result = op(a, b)
///
/// Constructs read/write buffers in dependency order and schedules the operation.
///
/// \tparam A First operand (scalar or Async<T>)
/// \tparam B Second operand (scalar or Async<U>)
/// \tparam R Result value type
/// \tparam Op Callable with signature R(op(A, B))
/// \ingroup async_api

template <typename A, typename B, typename R, typename Op> void async_binary_op(A&& a, B&& b, Async<R>& out, Op op)
{
  auto a_buf = read(std::forward<A>(a));
  auto b_buf = read(std::forward<B>(b));
  auto out_buf = out.write();

  schedule([](auto a_, auto b_, WriteBuffer<R> out_, Op op_) -> AsyncTask {
    // auto& [va, vb] = make_ref(co_await all(a_, b_));
    // TRACE(&va, &vb);
    // auto tmp = op_(va, vb);
    auto ab = co_await all(a_, b_);
    auto tmp = op_(std::get<0>(ab), std::get<1>(ab));
    a_.release();
    b_.release();
    co_await out_ = std::move(tmp); // Suspend *after* releasing readers
    co_return;
  }(std::move(a_buf), std::move(b_buf), std::move(out_buf), std::move(op)));
}

// This version is correct, but does not use all() awaiter
// template <typename A, typename B, typename R, typename Op> void async_binary_op(A&& a, B&& b, Async<R>& out, Op op)
// {
//   auto a_buf = read(std::forward<A>(a));
//   auto b_buf = read(std::forward<B>(b));
//   auto out_buf = out.write();
//
//   schedule([](auto a_, auto b_, WriteBuffer<R> out_, Op op_) -> AsyncTask {
//     auto const& va = co_await a_;
//     auto const& vb = co_await b_;
//     auto tmp = op_(va, vb);
//     a_.release();
//     b_.release();
//     co_await out_ = std::move(tmp); // Suspend *after* releasing readers
//     co_return;
//   }(read(std::forward<A>(a)), read(std::forward<B>(b)), out.write(), std::move(op)));
// }

/// \brief Assigns an Async or scalar value into an Async destination.
/// \ingroup async_api
///
/// This schedules a coroutine that reads the value `rhs` (whether it's an
/// Async or a scalar), and writes it into `lhs`.
///
/// \tparam T The value type of the destination.
/// \tparam U The source type, either a value or Async<U>.
template <typename U, typename T>
void async_assign(U&& rhs, Async<T>& lhs)
  requires requires { T{std::declval<async_value_t<U>>()}; }
{
  schedule([](auto in_, WriteBuffer<T> out_) -> AsyncTask {
    auto const& val = co_await in_;
    auto& out = co_await out_;
    out = static_cast<T>(val); // conversion if needed
    co_return;
  }(read(std::forward<U>(rhs)), lhs.write()));
}

// +, -, *, /
#define UNI20_DEFINE_BINARY_OP(OPNAME, OP)                                                                             \
  template <typename T, typename U>                                                                                    \
    requires is_async_binary_applicable_v<T, U, OP> && contains_an_async<T, U>                                         \
  auto operator OPNAME(T&& a, U&& b)                                                                                   \
  {                                                                                                                    \
    using R = async_binary_result_t<T, U, OP>;                                                                         \
    Async<R> result;                                                                                                   \
    async_binary_op(std::forward<T>(a), std::forward<U>(b), result, OP{});                                             \
    return result;                                                                                                     \
  }

UNI20_DEFINE_BINARY_OP(+, std::plus<>)
UNI20_DEFINE_BINARY_OP(-, std::minus<>)
UNI20_DEFINE_BINARY_OP(*, std::multiplies<>)
UNI20_DEFINE_BINARY_OP(/, std::divides<>)

#undef UNI20_DEFINE_BINARY_OP

// +=, -=, *=, /=
#define UNI20_DEFINE_COMPOUND_OP(OPNAME, OP)                                                                           \
  template <typename T, typename U>                                                                                    \
    requires is_async_binary_applicable_v<T, U, OP> && contains_an_async<T, U>                                         \
  Async<T>& operator OPNAME##=(Async<T>& lhs, U&& rhs)                                                                 \
  {                                                                                                                    \
    async_binary_op(lhs, std::forward<U>(rhs), lhs, OP{});                                                             \
    return lhs;                                                                                                        \
  }

UNI20_DEFINE_COMPOUND_OP(+, std::plus<>)
UNI20_DEFINE_COMPOUND_OP(-, std::minus<>)
UNI20_DEFINE_COMPOUND_OP(*, std::multiplies<>)
UNI20_DEFINE_COMPOUND_OP(/, std::divides<>)

#undef UNI20_DEFINE_COMPOUND_OP
/// @}

} // namespace uni20::async
