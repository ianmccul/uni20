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

// /// \brief concept for a valid type that behaves as ar read buffer awaitable.
// ///        This requires that `T x = co_await a` is valid.
// template <typename Awaitable, typename T>
// concept read_buffer_awaitable =
//     requires(Awaitable a) { requires std::constructible_from<T, decltype(a.await_resume())>; };
//
// /// \brief concept for a valid type that behaves as a write buffer awaitable
// ///        This requires that `T x; co_await a = x` is valid.
// template <typename Awaitable, typename T>
// concept write_buffer_awaitable =
//     requires(Awaitable a, T x) { requires std::assignable_from<decltype(a.await_resume()), T>; };

template <typename Awaitable>
concept read_buffer_awaitable =
    requires(Awaitable a) {
      typename std::remove_cvref_t<decltype(get_awaiter(a))>::value_type;
      requires std::constructible_from<typename std::remove_cvref_t<decltype(get_awaiter(a))>::value_type,
                                       decltype(get_awaiter(a).await_resume())>;
    };

// template <typename Awaitable>
// concept read_buffer_awaitable =
//     requires(Awaitable a) {
//       typename std::remove_cvref_t<decltype(get_awaiter(a))>::value_type;
//       requires std::constructible_from<typename remove_cvref_t<decltype(get_awaiter(a))>::value_type,
//                                        decltype(get_awaiter(a).await_resume())>;
//     };

template <typename Awaitable>
concept write_buffer_awaitable =
    requires(Awaitable a, typename Awaitable::value_type v) {
      typename std::remove_cvref_t<decltype(get_awaiter(a))>::value_type;
      requires std::assignable_from<decltype(get_awaiter(a).await_resume()),
                                    typename std::remove_cvref_t<decltype(get_awaiter(a))>::value_type>;
    };

template <typename Awaitable, typename T>
concept read_buffer_awaitable_of =
    requires(Awaitable a) { requires std::constructible_from<T, decltype(a.await_resume())>; };

template <typename Awaitable, typename T>
concept write_buffer_awaitable_of =
    requires(Awaitable a) { requires std::assignable_from<decltype(a.await_resume()), T>; };

/// \brief concept for a valid type that behaves as a write buffer that is also readable
///        This requires that `T x = co_await a; co_await a = x` is valid
// template <typename Awaitable, typename T>
// concept read_write_buffer_awaitable = read_buffer_awaitable<Awaitable, T> && write_buffer_awaitable<Awaitable, T>;

template <typename Awaitable>
concept read_write_buffer_awaitable = read_buffer_awaitable<Awaitable> && write_buffer_awaitable<Awaitable>;

template <typename Awaitable, typename T>
concept read_write_buffer_awaitable_of =
    read_buffer_awaitable_of<Awaitable, T> && write_buffer_awaitable_of<Awaitable, T>;

/// \Brief an async_reader is satisfied if the .read() member returns an async read buffer
template <typename T>
concept async_reader = requires(T t) { requires read_buffer_awaitable<decltype(t.read())>; };

/// \Brief an async_reader is satisfied if the .write() member returns an async write buffer
template <typename T>
concept async_writer = requires(T t) { requires write_buffer_awaitable<decltype(t.write())>; };

// FIXME: I think the std::remove_cvref_t is wrong here
template <typename Awaitable, typename T>
concept async_reader_of =
    requires(std::remove_cvref_t<Awaitable> a) { requires read_buffer_awaitable_of<decltype(a.read()), T>; };

// FIXME: I think the std::remove_cvref_t is wrong here
template <typename Awaitable, typename T>
concept async_writer_to =
    requires(std::remove_cvref_t<Awaitable> a) { requires write_buffer_awaitable_of<decltype(a.write()), T>; };

template <typename Awaitable, typename T>
concept async_movable_to =
    async_writer<Awaitable> &&
    requires(Awaitable a) {
      // Write buffer yields a movable type...
      typename std::remove_cvref_t<decltype(get_awaiter(a.write()))>::element_type;

      // ...that can be moved into T
      requires std::constructible_from<T, decltype(std::move(get_awaiter(a.write()).await_resume()))>;
    };

// template <typename T, typename Expected>
// concept async_writer_to = requires(std::remove_cvref_t<T> t) {
//                             {
//                               t.write()
//                               } -> write_buffer_awaitable<Expected>;
//                           };
//
// template <typename T, typename U>
// concept async_reader_of = requires(std::remove_cvref_t<T> t) {
//   { t.read() } -> read_buffer_awaitable<U>;
// };
//
// template <typename T, typename U>
// concept async_writer_to = requires(std::remove_cvref_t<T> t) {
//   { t.writer() } -> read_buffer_awaitable<U>;
// };

//
// template <typename T>
// concept async_reader =
//   requires(std::remove_cvref_t<T> t) {
//     typename decltype(t.read())::value_type;
//   };
//
// template <typename T>
// concept async_writer =
//   requires(std::remove_cvref_t<T> t) {
//     typename decltype(t.write())::value_type;
//   };
//

// /// \brief concept for a type that behaves like an asyncronous reader: it has a read() function that returns an
// ///        awaiter that is convertible to the nested value_type
// template <typename T>
// concept async_reader =
//     requires(std::remove_cvref_t<T> t) {
//       typename std::remove_cvref_t<T>::value_type;
//       requires read_buffer_awaitable<decltype(t.read()), typename std::remove_cvref_t<T>::value_type>;
//     };
//
// /// \brief concept for a type that behaves like an asyncronous writer: it has a write() function that returns an
// ///        awaiter that can be assigned from the nested value_type
// template <typename T>
// concept async_writer =
//     requires(std::remove_cvref_t<T> t) {
//       typename std::remove_cvref_t<T>::value_type;
//       requires write_buffer_awaitable<decltype(t.write()), typename std::remove_cvref_t<T>::value_type>;
//     };

/// \brief concept for a type that behaves like an asyncronous writer that is also readable:
///        it has a write() function that returns an awaiter that can be assigned from the nested value_type,
///        and also read from
template <typename T>
concept async_read_writer = requires(T t) {
                              requires read_buffer_awaitable<decltype(t.read())>;
                              requires write_buffer_awaitable<decltype(t.write())>;
                            };

// template <typename T>
// concept async_read_writer = requires(T t) {
//                               typename T::value_type;
//                               requires read_write_buffer_awaitable<decltype(t.write())>;
//                             };

/// \brief concept for a type that behaves like an asyncronous reader and writer (like Async<T>)
template <typename T>
concept async_like = async_reader<T> && async_writer<T>;

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
  requires(!async_reader<std::remove_cvref_t<T>>)
ValueAwaiter<std::remove_cvref_t<T>> read(T&& x)
{
  return ValueAwaiter<std::remove_cvref_t<T>>{std::forward<T>(x)};
}

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

/// \brief Concept to determine if Op(T,U) is a candidate for forwarding for Async
template <typename T, typename U, typename Op>
concept async_binary_applicable = (async_reader<std::remove_cvref_t<T>> || async_reader<std::remove_cvref_t<U>>) &&
                                  requires(Op op, async_value_t<T> t, async_value_t<U> u) {
                                    {
                                      op(t, u)
                                    };
                                  };

/// \brief concept for validating a binary operation
///        true if co_await out.write() = op(a,b) is valid
template <typename A, typename B, typename Writer, typename Op>
concept async_binary_assignable =
    async_writer<Writer> && requires(Op op, async_value_t<A> a, async_value_t<B> b, Writer& out) {
                              requires std::assignable_from<decltype(out.write().await_resume()), decltype(op(a, b))>;
                            };

template <typename T, typename U, typename Op>
concept is_async_compound_applicable =
    async_read_writer<std::remove_reference_t<T>> &&
    requires(async_value_t<T>& lhs, async_value_t<U>&& rhs, Op op) { op(lhs, std::forward<async_value_t<U>>(rhs)); };

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
template <typename A, typename B, async_writer Writer, typename Op>
  requires async_binary_applicable<A, B, Op>
void async_binary_op(A&& a, B&& b, Writer& out, Op op)
{
  auto a_buf = read(std::forward<A>(a));
  auto b_buf = read(std::forward<B>(b));
  auto out_buf = out.write();

  schedule([](auto a_, auto b_, auto out_, Op op_) -> AsyncTask {
    auto ab = co_await all(a_, b_);
    auto tmp = op_(std::get<0>(ab), std::get<1>(ab));
    a_.release();
    b_.release();
    co_await out_ = std::move(tmp); // Suspend *after* releasing readers
    co_return;
  }(std::move(a_buf), std::move(b_buf), std::move(out_buf), std::move(op)));
}

/// \brief Launch a coroutine applying an in-place operation op(lhs, rhs) on an Async<T>,
///        where lhs is written, and rhs is read.
///
/// Schedules a coroutine that obtains the value in lhs as a reference, rhs as a const-reference,
/// and executes `op(lhs, rhs)`. The coroutine suspends once to await both operands and
/// then performs the in-place operation on lhs.
///
/// \tparam U Right-hand operand type (scalar or Async<U>)
/// \tparam T Value type of the Async<T> being modified
/// \tparam Op Callable with signature void(T&, U const&)
/// \param rhs Right-hand operand (either U or Async<U>)
/// \param lhs Left-hand operand of the form Async<T>& — this is the value being modified
/// \param op  In-place operation to apply (e.g., a lambda using operator+=)
/// \ingroup async_api
template <typename U, typename T, typename Op> void async_compound_op(U&& rhs, Async<T>& lhs, Op op)
{
  schedule([](auto rhs_, WriteBuffer<T> out_, Op op_) -> AsyncTask {
    auto tmp = co_await all(rhs_, out_);
    auto& rhs_val = std::get<0>(tmp);
    auto& lhs_ref = std::get<1>(tmp);
    op_(lhs_ref, rhs_val); // in-place mutation
    co_return;
  }(read(std::forward<U>(rhs)), lhs.write(), std::move(op)));
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
void async_assign(U&& rhs, WriteBuffer<T> lhs)
  requires requires { T{std::declval<async_value_t<U>>()}; }
{
  schedule([](auto in_, WriteBuffer<T> out_) -> AsyncTask {
    auto const& val = co_await in_;
    auto& out = co_await out_;
    out = val;
    co_return;
  }(read(std::forward<U>(rhs)), std::move(lhs)));
}

template <typename U, typename T>
void async_assign(U&& rhs, Async<T>& lhs)
  requires requires { T{std::declval<async_value_t<U>>()}; }
{
  async_assign(rhs, lhs.write());
}

template <typename U, typename T>
  requires async_movable_to<U, T>
void async_move(U& rhs, WriteBuffer<T> lhs)
{
  schedule([](auto src, WriteBuffer<T> dst) -> AsyncTask {
    auto&& movable = co_await src;
    auto& out = co_await dst;
    out = std::move(movable);
    co_return;
  }(rhs.write(), std::move(lhs)));
}

template <typename U, typename T>
  requires(!async_writer<U>)
void async_move(U&& rhs, Async<T>& lhs)
{
  schedule([](T val, WriteBuffer<T> dst) -> AsyncTask {
    auto& out = co_await dst;
    out = std::move(val);
    co_return;
  }(std::move(rhs), lhs.write()));
}

template <typename U, typename T> void async_move(U&& rhs, Async<T>& lhs)
{
  async_move(std::forward<U>(rhs), lhs.write());
}

// +, -, *, /
#define UNI20_DEFINE_BINARY_OP(OPNAME, OP)                                                                             \
  template <typename T, typename U>                                                                                    \
    requires async_binary_applicable<T, U, OP>                                                                         \
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

#define UNI20_DEFINE_ASSIGN_OP(NAME, OP)                                                                               \
  struct NAME                                                                                                          \
  {                                                                                                                    \
      template <typename LHS, typename RHS> void operator()(LHS& lhs, RHS&& rhs) const                                 \
      {                                                                                                                \
        lhs OP std::forward<RHS>(rhs);                                                                                 \
      }                                                                                                                \
  };

UNI20_DEFINE_ASSIGN_OP(plus_assign, +=)
UNI20_DEFINE_ASSIGN_OP(minus_assign, -=)
UNI20_DEFINE_ASSIGN_OP(multiplies_assign, *=)
UNI20_DEFINE_ASSIGN_OP(divides_assign, /=)

#undef UNI20_DEFINE_ASSIGN_OP

#define UNI20_DEFINE_ASYNC_COMPOUND_OPERATOR(OPSYM, FUNCTOR)                                                           \
  template <typename T, typename U>                                                                                    \
    requires uni20::async::is_async_compound_applicable<Async<T>                                                       \
                                                        &,                                                             \
                                                        U, uni20::async::FUNCTOR>                                      \
        Async<T>& operator OPSYM(Async<T>& lhs, U&& rhs)                                                               \
  {                                                                                                                    \
    uni20::async::async_compound_op(std::forward<U>(rhs), lhs, uni20::async::FUNCTOR{});                               \
    return lhs;                                                                                                        \
  }

UNI20_DEFINE_ASYNC_COMPOUND_OPERATOR(+=, plus_assign)
UNI20_DEFINE_ASYNC_COMPOUND_OPERATOR(-=, minus_assign)
UNI20_DEFINE_ASYNC_COMPOUND_OPERATOR(*=, multiplies_assign)
UNI20_DEFINE_ASYNC_COMPOUND_OPERATOR(/=, divides_assign)

#undef UNI20_DEFINE_ASYNC_COMPOUND_OPERATOR

/// @}

} // namespace uni20::async
