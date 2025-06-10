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
    ReverseValue() : async_{}, write_buf_(std::get<0>(async_.prepend_epoch())), read_buf_(async_.read()) {}

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
    ReadBuffer<T> input() const { return read_buf_; }

    ReadBuffer<T> read() const { return read_buf_; }

    // Get a WriteBuffer<T> to the earliest epoch - this is the 'output gradient' fed from the next stage
    [[nodiscard]] WriteBuffer<T> output()
    {
      WriteBuffer<T> w = std::move(write_buf_);
      std::tie(write_buf_, read_buf_) = async_.prepend_epoch();
      return w;
    }

    void finalize()
    {
      read_buf_.release();
      write_buf_.release();
    }

    /// Since we are guaranteed that the write is immediate, we don't need to wait
    template <typename U>
    ReverseValue& operator=(U&& v)
      requires std::assignable_from<T&, U&&>
    {
      write_buf_.write_assert(std::forward<U>(v));
      write_buf_.release();
      read_buf_.release();
      // async_ = Async<T>{};
      // std::tie(write_buf_, read_buf_) = async_.prepend_epoch();
      return *this;
    }

    // Assigning from an Async<T> is possible; this launches a coroutine do to the copy
    // This is 'final', and cannot assign a second time, nor can we access the input gradient
    template <typename U> ReverseValue& operator=(Async<U> const& v)
    {
      read_buf_.release();
      async_assign(v.read(), std::move(write_buf_));
      return *this;
    }

    // Moving from an Async<T> is possible; this launches a coroutine do to the copy
    // This is 'final', and cannot assign a second time
    template <typename U> ReverseValue& operator=(Async<U>&& v)
    {
      read_buf_.release();
      async_move(std::move(v), std::move(write_buf_));
      return *this;
    }

    template <typename U> ReverseValue& operator+=(Async<U> const& v)
    {
      WriteBuffer<T> w = std::move(write_buf_);
      std::tie(write_buf_, read_buf_) = async_.prepend_epoch();

      schedule(accumulate(read_buf_, v.read(), std::move(w)));
      return *this;
    }

    template <typename U> ReverseValue& operator+=(ReadBuffer<U> v)
    {
      WriteBuffer<T> w = std::move(write_buf_);
      std::tie(write_buf_, read_buf_) = async_.prepend_epoch();

      schedule(accumulate(read_buf_, std::move(v), std::move(w)));
      return *this;
    }

    template <typename U> ReverseValue& operator-=(Async<U> const& v)
    {
      WriteBuffer<T> w = std::move(write_buf_);
      std::tie(write_buf_, read_buf_) = async_.prepend_epoch();

      schedule(accumulate_minus(read_buf_, v.read(), std::move(w)));
      return *this;
    }

    template <typename U> ReverseValue& operator-=(ReadBuffer<U> v)
    {
      WriteBuffer<T> w = std::move(write_buf_);
      std::tie(write_buf_, read_buf_) = async_.prepend_epoch();

      schedule(accumulate_minus(read_buf_, std::move(v), std::move(w)));
      return *this;
    }

    Async<T>& value() & { return async_; }
    Async<T> const& value() const& { return async_; }
    Async<T> value() && { return std::move(async_); }

  private:
    Async<T> async_;
    WriteBuffer<T> write_buf_;
    ReadBuffer<T> read_buf_;
};

template <typename T> struct async_value_type<ReverseValue<T>>
{
    using type = T;
};

template <typename T> ReadBuffer<T> read(ReverseValue<T> const& x) { return x.input(); }

} // namespace uni20::async
