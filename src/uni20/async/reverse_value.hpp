#pragma once
/// \file reverse_value.hpp
/// \brief Reverse-mode gradient accumulation channel built on reverse epoch queues.
#include "async.hpp"
#include "buffers.hpp"
#include "epoch_queue.hpp"
#include <type_traits>

namespace uni20::async
{

/// \brief Compute `out = a + b` while tolerating cancellation of either input.
/// \tparam T Left/output value type.
/// \tparam U Right input value type.
/// \param a_ Left input read buffer.
/// \param b_ Right input read buffer.
/// \param out_ Output write buffer.
template <typename T, typename U> AsyncTask async_accumulate(ReadBuffer<T> a_, ReadBuffer<U> b_, WriteBuffer<T> out_)
{
  auto a = co_await std::move(a_).maybe();
  if (a)
  {
    T value = std::move(*a).get_release();
    auto b = co_await std::move(b_).maybe();
    if (b) value += std::move(*b).get_release();
    co_await out_ = std::move(value);
  }
  else
  {
    // GCC 13 workaround for `(co_await ...).get()`: stage through a named proxy.
    // GCC 14+ supports the one-liner form:
    //   U value = (co_await std::move(b_).or_cancel()).get();
    auto b = co_await std::move(b_).or_cancel();
    U value = b.get();
    b.release();
    co_await out_ += std::move(value);
  }
  co_return;
}

/// \brief Compute `out = a - b` while tolerating cancellation of either input.
/// \tparam T Left/output value type.
/// \tparam U Right input value type.
/// \param a_ Left input read buffer.
/// \param b_ Right input read buffer.
/// \param out_ Output write buffer.
template <typename T, typename U>
AsyncTask async_accumulate_minus(ReadBuffer<T> a_, ReadBuffer<U> b_, WriteBuffer<T> out_)
{
  auto a = co_await std::move(a_).maybe();
  if (a)
  {
    T value = std::move(*a).get_release();
    auto b = co_await std::move(b_).maybe();
    if (b) value -= std::move(*b).get_release();
    co_await out_ = std::move(value);
  }
  else
  {
    auto b = co_await std::move(b_).or_cancel();
    U value = b.get();
    b.release();
    co_await out_ -= std::move(value);
  }
  co_return;
}

/// \brief Reverse-mode accumulation endpoint with staged input/output gradient channels.
/// \tparam T Gradient value type.
template <typename T> class ReverseValue {
  public:
    using value_type = T;

    /// \brief Construct a new, uninitialized ReverseValue.
    ReverseValue() : async_(async_do_not_start), rqueue_(async_.queue().latest()) {}

    /// \brief Move constructor.
    ReverseValue(ReverseValue&&) noexcept = default;

    /// \brief Move assignment finalizes any pending chain before transfer.
    /// \param other Source reverse value.
    /// \return Reference to `*this`.
    ReverseValue& operator=(ReverseValue&& other) noexcept
    {
      if (this != &other)
      {
        this->finalize();
        async_ = std::move(other.async_);
        rqueue_ = std::move(other.rqueue_);
        started_ = std::exchange(other.started_, false);
      }
      return *this;
    }

    /// \brief Non-copyable.
    ReverseValue(ReverseValue const&) = delete;

    /// Access as an Async<T> (read-only use).
    // Async<T> const& async() const noexcept { return async_; }

    /// \brief Get the final gradient value; also finalizes the computation chain.
    // ReadBuffer<T> final()
    // {
    //   this->finalize();
    //   return async_.read();
    // }

    /// \brief Destructor finalizes the reverse queue if needed.
    ~ReverseValue() { this->finalize(); }

    /// \brief Finalize and return the underlying async gradient channel.
    /// \return Finalized async gradient channel.
    Async<T> final()
    {
      this->finalize();
      return async_;
    }

    /// \brief Finalize and synchronously wait for the terminal gradient value.
    /// \return Final gradient value.
    T final_wait()
    {
      this->finalize();
      return async_.get_wait();
    }

    /// \brief Wait for and return the final gradient value.
    /// \return Final gradient value copied from the finalized reverse channel.
    T get_wait() { return this->final_wait(); }

    /// \brief Get the input-gradient read buffer from the earliest reverse epoch.
    /// \return Read buffer used as input to upstream reverse operations.
    ReadBuffer<T> input() const { return this->read_buffer(); }

    /// \brief Alias for `input()`.
    /// \return Read buffer used as input to upstream reverse operations.
    ReadBuffer<T> read() const { return this->read_buffer(); }

    /// \brief Get the output-gradient write buffer for downstream accumulation.
    /// \return Write buffer for feeding gradient contributions.
    [[nodiscard]] WriteBuffer<T> output() { return this->write_buffer(); }

    /// \brief Ensure the reverse queue has been started exactly once.
    void finalize() const
    {
      if (!started_)
      {
        if (!rqueue_.is_started())
        {
          rqueue_.start();
        }
        started_ = true;
      }
    }

    /// \brief Immediately assign a terminal gradient value.
    /// \tparam U Source type constructible as `T`.
    /// \param v Source value.
    /// \return Reference to `*this`.
    template <typename U>
    ReverseValue& operator=(U&& v)
      requires std::constructible_from<T, U&&>
    {
      WriteBuffer<T> w(this->write_buffer());
      rqueue_.start();
      w.emplace_assert(std::forward<U>(v));
      return *this;
    }

    /// \brief Assign from an async source by scheduling an async copy.
    /// \tparam U Source value type.
    /// \param v Source async value.
    /// \return Reference to `*this`.
    template <typename U> ReverseValue& operator=(Async<U> const& v)
    {
      async_assign(v.read(), this->write_buffer());
      rqueue_.start();
      return *this;
    }

    /// \brief Move-assign from an async source by scheduling an async move.
    /// \tparam U Source value type.
    /// \param v Source async value.
    /// \return Reference to `*this`.
    template <typename U> ReverseValue& operator=(Async<U>&& v)
    {
      async_move(std::move(v), this->write_buffer());
      rqueue_.start();
      return *this;
    }

    /// \brief Accumulate an async source into this reverse channel.
    /// \tparam U Source value type.
    /// \param v Source async value.
    /// \return Reference to `*this`.
    template <typename U> ReverseValue& operator+=(Async<U> const& v)
    {
      // It is important that we construct the buffer objects in the right order. Writer first,
      // then reader, so the reader is the earlier epoch in the ReverseEpochQueue
      WriteBuffer<T> w(this->write_buffer());
      schedule(async_accumulate(this->read_buffer(), v.read(), std::move(w)));
      return *this;
    }

    /// \brief Accumulate another reverse channel into this channel.
    /// \tparam U Source value type.
    /// \param v Source reverse value.
    /// \return Reference to `*this`.
    template <typename U> ReverseValue& operator+=(ReverseValue<U> const& v)
    {
      WriteBuffer<T> w(this->write_buffer());
      schedule(async_accumulate(this->read_buffer(), v.read(), std::move(w)));
      return *this;
    }

    /// \brief Accumulate a read buffer source into this channel.
    /// \tparam U Source value type.
    /// \param v Source read buffer.
    /// \return Reference to `*this`.
    template <typename U> ReverseValue& operator+=(ReadBuffer<U> v)
    {
      WriteBuffer<T> w(this->write_buffer());
      schedule(async_accumulate(this->read_buffer(), std::move(v), std::move(w)));
      return *this;
    }

    /// \brief Subtract an async source from this reverse channel.
    /// \tparam U Source value type.
    /// \param v Source async value.
    /// \return Reference to `*this`.
    template <typename U> ReverseValue& operator-=(Async<U> const& v)
    {
      WriteBuffer<T> w(this->write_buffer());
      schedule(async_accumulate_minus(this->read_buffer(), v.read(), std::move(w)));
      return *this;
    }

    /// \brief Subtract another reverse channel from this channel.
    /// \tparam U Source value type.
    /// \param v Source reverse value.
    /// \return Reference to `*this`.
    template <typename U> ReverseValue& operator-=(ReverseValue<U> const& v)
    {
      WriteBuffer<T> w(this->write_buffer());
      schedule(async_accumulate_minus(this->read_buffer(), v.read(), std::move(w)));
      return *this;
    }

    /// \brief Subtract a read-buffer source from this channel.
    /// \tparam U Source value type.
    /// \param v Source read buffer.
    /// \return Reference to `*this`.
    template <typename U> ReverseValue& operator-=(ReadBuffer<U> v)
    {
      WriteBuffer<T> w(this->write_buffer());
      schedule(async_accumulate_minus(this->read_buffer(), std::move(v), std::move(w)));
      return *this;
    }

    /// \brief Finalize and return mutable access to the backprop async channel.
    /// \return Mutable reference to finalized async gradient channel.
    Async<T>& backprop() &
    {
      this->finalize();
      return async_;
    }
    /// \brief Finalize and return const access to the backprop async channel.
    /// \return Const reference to finalized async gradient channel.
    Async<T> const& backprop() const&
    {
      this->finalize();
      return async_;
    }
    /// \brief Finalize and move out the backprop async channel.
    /// \return Finalized async gradient channel.
    Async<T> backprop() &&
    {
      this->finalize();
      return std::move(async_);
    }

    /// \brief Access the latest internal async value without forcing finalization.
    /// \return Mutable reference to the underlying async channel.
    Async<T>& last_value() & { return async_; }
    /// \brief Access the latest internal async value without forcing finalization.
    /// \return Const reference to the underlying async channel.
    Async<T> const& last_value() const& { return async_; }
    /// \brief Move out the latest internal async value without forcing finalization.
    /// \return Underlying async channel.
    Async<T> last_value() && { return std::move(async_); }

    /// \brief Explicitly start the reverse queue.
    void start() { rqueue_.start(); }

  private:
    /// \brief Create the read buffer for the current earliest reverse epoch.
    /// \return Read buffer bound to reverse queue ordering.
    ReadBuffer<T> read_buffer() const { return ReadBuffer<T>(rqueue_.create_read_context(async_.storage())); }
    /// \brief Create the write buffer for the current earliest reverse epoch.
    /// \return Write buffer bound to reverse queue ordering.
    WriteBuffer<T> write_buffer() { return WriteBuffer<T>(rqueue_.create_write_context(async_.storage())); }

    Async<T> async_;
    mutable ReverseEpochQueue rqueue_; // must be mutable if we want read access to be logically const
    mutable bool started_{false};
};

template <typename T> struct async_value_type<ReverseValue<T>>
{
    using type = T;
};

/// \brief Free-function read adapter for ReverseValue.
/// \tparam T Stored value type.
/// \param x Reverse value.
/// \return Input read buffer for the reverse channel.
template <typename T> ReadBuffer<T> read(ReverseValue<T> const& x) { return x.input(); }

} // namespace uni20::async
