#pragma once
#include "async.hpp"
#include "buffers.hpp"
#include "epoch_queue.hpp"
#include <type_traits>

namespace uni20::async
{

// Computes out_ = a_ + b_
// in such a way that a_ and/or b_ can be cancelled
template <typename T, typename U> AsyncTask async_accumulate(ReadBuffer<T> a_, ReadBuffer<U> b_, WriteBuffer<T> out_)
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

// Computes out_ = a_ - b_
// in such a way that a_ and/or b_ can be cancelled
template <typename T, typename U>
AsyncTask async_accumulate_minus(ReadBuffer<T> a_, ReadBuffer<U> b_, WriteBuffer<T> out_)
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
    ReverseValue() : async_(async_do_not_start), rqueue_(async_.queue().latest()) {}

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
    ReadBuffer<T> input() const { return this->read_buffer(); }

    ReadBuffer<T> read() const { return this->read_buffer(); }

    // Get a WriteBuffer<T> to the earliest epoch - this is the 'output gradient' fed from the next stage
    [[nodiscard]] WriteBuffer<T> output() { return this->write_buffer(); }

    void finalize()
    {
      if (!started_)
      {
        rqueue_.start();
        started_ = true;
      }
    }

    /// Since we are guaranteed that the write is immediate, we don't need to wait
    template <typename U>
    ReverseValue& operator=(U&& v)
      requires std::assignable_from<T&, U&&>
    {
      TRACE("Assigning to ReverseValue", this);
      EmplaceBuffer<T> w(this->emplace_buffer());
      rqueue_.start();
      w.emplace_assert(std::forward<U>(v));
      return *this;
    }

    // Assigning from an Async<T> is possible; this launches a coroutine do to the copy
    // This is 'final', and cannot assign a second time, nor can we access the input gradient
    template <typename U> ReverseValue& operator=(Async<U> const& v)
    {
      async_assign(v.read(), this->write_buffer());
      rqueue_.start();
      return *this;
    }

    // Moving from an Async<T> is possible; this launches a coroutine do to the copy
    // This is 'final', and cannot assign a second time
    template <typename U> ReverseValue& operator=(Async<U>&& v)
    {
      async_move(std::move(v), this->write_buffer());
      rqueue_.start();
      return *this;
    }

    template <typename U> ReverseValue& operator+=(Async<U> const& v)
    {
      // It is important that we construct the buffer objects in the right order. Writer first,
      // then reader, so the reader is the earlier epoch in the ReverseEpochQueue
      WriteBuffer<T> w(this->write_buffer());
      schedule(async_accumulate(this->read_buffer(), v.read(), std::move(w)));
      return *this;
    }

    template <typename U> ReverseValue& operator+=(ReverseValue<U> const& v)
    {
      WriteBuffer<T> w(this->write_buffer());
      schedule(async_accumulate(this->read_buffer(), v.read(), std::move(w)));
      return *this;
    }

    template <typename U> ReverseValue& operator+=(ReadBuffer<U> v)
    {
      WriteBuffer<T> w(this->write_buffer());
      schedule(async_accumulate(this->read_buffer(), std::move(v), std::move(w)));
      return *this;
    }

    template <typename U> ReverseValue& operator-=(Async<U> const& v)
    {
      WriteBuffer<T> w(this->write_buffer());
      schedule(async_accumulate_minus(this->read_buffer(), v.read(), std::move(w)));
      return *this;
    }

    template <typename U> ReverseValue& operator-=(ReverseValue<U> const& v)
    {
      WriteBuffer<T> w(this->write_buffer());
      schedule(async_accumulate_minus(this->read_buffer(), v.read(), std::move(w)));
      return *this;
    }

    template <typename U> ReverseValue& operator-=(ReadBuffer<U> v)
    {
      WriteBuffer<T> w(this->write_buffer());
      schedule(async_accumulate_minus(this->read_buffer(), std::move(v), std::move(w)));
      return *this;
    }

    Async<T>& value() & { return async_; }
    Async<T> const& value() const& { return async_; }
    Async<T> value() && { return std::move(async_); }

  private:
    ReadBuffer<T> read_buffer() const { return ReadBuffer<T>(rqueue_.create_read_context(async_.storage())); }
    WriteBuffer<T> write_buffer() { return WriteBuffer<T>(rqueue_.create_write_context(async_.storage())); }
    EmplaceBuffer<T> emplace_buffer() { return EmplaceBuffer<T>(rqueue_.create_write_context(async_.storage())); }

    Async<T> async_;
    mutable ReverseEpochQueue rqueue_; // must be mutable if we want read access to be logically const
    bool started_{false};
};

template <typename T> struct async_value_type<ReverseValue<T>>
{
    using type = T;
};

template <typename T> ReadBuffer<T> read(ReverseValue<T> const& x) { return x.input(); }

} // namespace uni20::async
