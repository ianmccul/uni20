#pragma once
#include "async.hpp"
#include "buffers.hpp"

namespace uni20::async
{

/// ReverseValue<T> owns a write-capable handle (WriteBuffer<T>) to an Async<T> value.
template <typename T> class ReverseValue {
  public:
    using value_type = T;

    /// Construct a new, uninitialized ReverseValue.
    ReverseValue() : async_{}, write_buf_{async_.write()}, read_buf_{async_.read()} {}

    // move ctor and move-assign should "just work"
    ReverseValue(ReverseValue&&) noexcept = default;
    ReverseValue& operator=(ReverseValue&&) noexcept = default;

    // copy-asssignment doesn't make sense
    ReverseValue(ReverseValue const&) = delete;

    /// Access as an Async<T> (read-only use).
    // Async<T> const& async() const noexcept { return async_; }

    /// \brief Get the final graident value
    ReadBuffer<T> final_grad() const { return async_.read(); }

    /// Get a ReadBuffer<T> from the earliest epoch - this is the 'output gradient' to be fed into the next stage
    ReadBuffer<T> grad_output() const { return read_buf_; }

    /// Since we are guaranteed that the write is immediate, we don't need to wait
    template <typename U>
    ReverseValue& operator=(U&& v)
      requires std::assignable_from<T&, U&&>
    {
      write_buf_.write_assert(std::forward<U>(v));
      write_buf_.release();
      async_ = Async<T>{};
      std::tie(write_buf_, read_buf_) = async_.prepend_epoch();
      return *this;
    }

    // Assigning from an Async<T> is possible; this launches a coroutine do to the copy
    // This is 'final', and cannot assign a second time
    template <typename U> ReverseValue& operator=(Async<U> const& v)
    {
      async_assign(v.read(), std::move(write_buf_));
      // async_ = Async<T>{};
      // std::tie(write_buf_, read_buf_) = async_.prepend_epoch();
      return *this;
    }

    // Moving from an Async<T> is possible; this launches a coroutine do to the copy
    // This is 'final', and cannot assign a second time
    template <typename U> ReverseValue& operator=(Async<U>&& v)
    {
      async_move(std::move(v), std::move(write_buf_));
      // async_ = Async<T>{};
      // std::tie(write_buf_, read_buf_) = async_.prepend_epoch();
      return *this;
    }

    template <typename U> ReverseValue& operator+=(Async<U> const& v)
    {
      WriteBuffer<T> w = std::move(write_buf_);
      std::tie(write_buf_, read_buf_) = async_.prepend_epoch();

      schedule([](auto a_, auto b_, auto out_) -> AsyncTask {
        auto ab = co_await all(a_, b_);
        auto tmp = std::get<0>(ab);
        tmp += std::get<1>(ab);
        a_.release();
        b_.release();
        co_await out_ = std::move(tmp); // Suspend *after* releasing readers
        co_return;
      }(read_buf_, v.read(), std::move(w)));
      return *this;
    }

    Async<T>& value() { return async_; }
    Async<T> const& value() const { return async_; }

  private:
    Async<T> async_;
    WriteBuffer<T> write_buf_;
    ReadBuffer<T> read_buf_;
};

template <typename T> struct async_value_type<ReverseValue<T>>
{
    using type = T;
};

} // namespace uni20::async
