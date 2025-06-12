#pragma once
#include "async.hpp"
#include "buffers.hpp"

namespace uni20::async
{

template <typename T, typename U> AsyncTask accumulate(ReadBuffer<T> a_, ReadBuffer<U> b_, WriteBuffer<T> out_)
{
  auto a = co_await a_.maybe();
  if (a)
  {
    auto tmp = std::move(*a);
    a_.release();
    auto b = co_await b_.maybe();
    if (b) tmp += *b;
    b_.release();
    co_await out_ = std::move(tmp);
  }
  else
  {
    a_.release();
    auto b = co_await b_.or_cancel();
    b_.release();
    co_await out_ = std::move(b);
  }
  co_return;
}

template <typename T, typename U> AsyncTask accumulate_minus(ReadBuffer<T> a_, ReadBuffer<U> b_, WriteBuffer<T> out_)
{
  auto a = co_await a_.maybe();
  if (a)
  {
    auto tmp = std::move(*a);
    a_.release();
    auto b = co_await b_.maybe();
    if (b) tmp -= *b;
    b_.release();
    co_await out_ = std::move(tmp);
  }
  else
  {
    a_.release();
    auto b = co_await b_.or_cancel();
    b_.release();
    co_await out_ = -std::move(b);
  }
  co_return;
}

/// ReverseValue<T> owns a write-capable handle (WriteBuffer<T>) to an Async<T> value.
template <typename T> class ReverseValue {
  public:
    using value_type = T;

    /// Construct a new, uninitialized ReverseValue.
    ReverseValue() : async_{}, buffers_(async_.prepend_epoch()) {}

    // move ctor and move-assign should "just work"
    ReverseValue(ReverseValue&&) noexcept = default;
    ReverseValue& operator=(ReverseValue&&) noexcept = default;

    // copy-asssignment doesn't make sense
    ReverseValue(ReverseValue const&) = delete;

    /// Access as an Async<T> (read-only use).
    // Async<T> const& async() const noexcept { return async_; }

    /// \brief Get the final graident value; also finalizes the computation chain
    // ReadBuffer<T> final()
    // {
    //   this->finalize();
    //   return async_.read();
    // }

    Async<T> final()
    {
      this->finalize();
      return async_;
    }

    T final_wait()
    {
      this->finalize();
      return async_.get_wait();
    }

    /// Get a ReadBuffer<T> from the earliest epoch - this is the 'input gradient' to be fed into the next stage
    ReadBuffer<T> input() const { return buffers_.reader; }

    ReadBuffer<T> read() const { return buffers_.reader; }

    // Get a WriteBuffer<T> to the earliest epoch - this is the 'output gradient' fed from the next stage
    [[nodiscard]] WriteBuffer<T> output()
    {
      WriteBuffer<T> w{std::move(buffers_.writer)};
      buffers_ = async_.prepend_epoch();
      return w;
    }

    void finalize()
    {
      buffers_.reader.release();
      buffers_.writer.release();
    }

    /// Since we are guaranteed that the write is immediate, we don't need to wait
    template <typename U>
    ReverseValue& operator=(U&& v)
      requires std::assignable_from<T&, U&&>
    {
      buffers_.writer.data() = std::forward<U>(v);
      buffers_.writer.release();
      buffers_.reader.release();
      return *this;
    }

    // Assigning from an Async<T> is possible; this launches a coroutine do to the copy
    // This is 'final', and cannot assign a second time, nor can we access the input gradient
    template <typename U> ReverseValue& operator=(Async<U> const& v)
    {
      buffers_.reader.release();
      async_assign(v.read(), WriteBuffer<T>(std::move(buffers_.writer)));
      return *this;
    }

    // Moving from an Async<T> is possible; this launches a coroutine do to the copy
    // This is 'final', and cannot assign a second time
    template <typename U> ReverseValue& operator=(Async<U>&& v)
    {
      buffers_.reader.release();
      async_move(std::move(v), WriteBuffer<T>(std::move(buffers_.writer)));
      return *this;
    }

    template <typename U> ReverseValue& operator+=(Async<U> const& v)
    {
      WriteBuffer<T> w{std::move(buffers_.writer)};
      buffers_ = async_.prepend_epoch();

      schedule(accumulate(ReadBuffer<T>(buffers_.reader), v.read(), std::move(w)));
      return *this;
    }

    template <typename U> ReverseValue& operator+=(ReadBuffer<U> v)
    {
      WriteBuffer<T> w{std::move(buffers_.writer)};
      buffers_ = async_.prepend_epoch();

      schedule(accumulate(ReadBuffer<T>(buffers_.reader), std::move(v), std::move(w)));
      return *this;
    }

    template <typename U> ReverseValue& operator-=(Async<U> const& v)
    {
      WriteBuffer<T> w{std::move(buffers_.writer)};
      buffers_ = async_.prepend_epoch();

      schedule(accumulate_minus(ReadBuffer<T>(buffers_.reader), v.read(), std::move(w)));
      return *this;
    }

    template <typename U> ReverseValue& operator-=(ReadBuffer<U> v)
    {
      WriteBuffer<T> w{std::move(buffers_.writer)};
      buffers_ = async_.prepend_epoch();

      schedule(accumulate_minus(ReadBuffer<T>(buffers_.reader), std::move(v), std::move(w)));
      return *this;
    }

    Async<T>& value() & { return async_; }
    Async<T> const& value() const& { return async_; }
    Async<T> value() && { return std::move(async_); }

  private:
    Async<T> async_;
    EpochQueue::EpochPair<T> buffers_;
};

template <typename T> struct async_value_type<ReverseValue<T>>
{
    using type = T;
};

template <typename T> ReadBuffer<T> read(ReverseValue<T> const& x) { return x.input(); }

} // namespace uni20::async
