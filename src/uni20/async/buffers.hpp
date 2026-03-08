/// \file buffers.hpp
/// \brief Awaitable gates for Async<T>: snapshot‐reads and in‐place writes.
/// \ingroup async_api

#pragma once

#include "async_task.hpp"
#include "async_task_promise.hpp"
#include <uni20/common/trace.hpp>
#include "epoch_context.hpp"
#include "shared_storage.hpp"

#include <atomic>
#include <concepts>
#include <coroutine>
#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::async
{

template <typename T> class Async;

template <typename T> class ReadMaybeAwaiter;
template <typename T> class ReadOrCancelAwaiter;
template <typename T> class OwningReadAccessProxy;
template <typename T> class OwningReadAwaiter;

/// \brief RAII handle for reading the value from an Async container at a given epoch.
///
/// A ReadBuffer<T> represents a read-only access to the value of an Async<T>
/// at a specific epoch. It is awaitable, and yields either a borrowed reference
/// or an owning proxy depending on value category:
///
/// - `co_await buf` yields `T const&`: shared read access.
/// - `co_await std::move(buf)` yields an owning proxy that keeps the read epoch alive.
///
/// \note The `std::move(buf)` form is recommended when the buffer will be consumed
///       immediately, such as when assigning to a local variable.
///
/// \note A `ReadBuffer<T>` can be co_awaited multiple times, but `std::move(buf)`
///       transfers ownership semantics and should only be used once. After moving,
///       further use is undefined.
///
/// \tparam T The underlying value type.
template <typename T> class ReadBuffer { //}: public AsyncAwaiter {
  public:
    using value_type = T;

    /// \brief Construct a read buffer tied to a reader context.
    /// \param reader The RAII epoch reader handle for this operation.
    ReadBuffer(EpochContextReader<T> reader) : reader_(std::move(reader)) {}

    ReadBuffer(ReadBuffer const& other) : reader_(other.reader_) { this->copy_exception_sink_from(other); }

    // No copy ctor here, although we could add one
    ReadBuffer& operator=(ReadBuffer const&) = delete;

    ReadBuffer(ReadBuffer&& other) noexcept : reader_(std::move(other.reader_))
    {
      this->move_exception_sink_from(other);
    }

    ReadBuffer& operator=(ReadBuffer&& other) noexcept
    {
      if (this != &other)
      {
        this->unregister_exception_sink(false);
        reader_ = std::move(other.reader_);
        this->move_exception_sink_from(other);
      }
      return *this;
    }

    ~ReadBuffer() noexcept { this->unregister_exception_sink(true); }

#if UNI20_DEBUG_DAG
    /// \brief Get the debug node pointer of the object
    NodeInfo const* node() const { return reader_.node(); }
#endif

    /// \brief Returns a `ReadMaybeAwaiter`.
    /// \details Lvalue reads return `T const*`; moved reads return `std::optional<OwningReadAccessProxy<T>>`.
    ReadMaybeAwaiter<T const&> maybe() &;

    ReadMaybeAwaiter<T> maybe() &&;

    ReadOrCancelAwaiter<T const&> or_cancel() &;

    ReadOrCancelAwaiter<T> or_cancel() &&;

    /// \brief Check if the value is already ready to be read.
    /// \return True if the epoch is ready and no suspension is needed.
    bool await_ready() const noexcept { return reader_.ready(); }

    /// \brief Suspend this coroutine and enqueue for resumption.
    /// \param t Coroutine task to enqueue.
    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "ReadBuffer::await_suspend()", this, t.h_);
      reader_.suspend(std::move(t), false);
    }

    /// \brief Resume execution and return the stored value.
    /// \return Reference to the stored T inside Async<T>.
    T const& await_resume() const& { return reader_.data(); }

    /// \brief Resume execution and return an owning read proxy.
    OwningReadAccessProxy<T> await_resume() &&;

    /// \brief Manually release the epoch reader before awaitable destruction.
    ///
    /// This allows the coroutine to relinquish its reader role earlier than
    /// its full lifetime.
    ///
    /// \post The ReadBuffer becomes inert and idempotent; calling `release()`
    ///       more than once has no effect.
    void release() noexcept { reader_.release(); }

    // T get_wait() && { return T(reader_.get_wait()); } // TODO: can this use move semantics?
    T const& get_wait() const { return reader_.get_wait(); }

    T const& get_wait(IScheduler& sched) const { return reader_.get_wait(sched); }

    /// \brief Enable co_await on lvalue ReadBuffer and return a borrowed reference.
    auto operator co_await() & noexcept -> ReadBuffer& { return *this; }
    auto operator co_await() const& noexcept -> ReadBuffer const& { return *this; }

    /// \brief Enable co_await on rvalue ReadBuffer and transfer ownership to an owning read proxy.
    auto operator co_await() && noexcept -> OwningReadAwaiter<T> { return OwningReadAwaiter<T>(std::move(reader_)); }

    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return reader_.epoch_context_shared(); }

    void register_exception_sink(BasicAsyncTaskPromise& promise, bool explicit_sink) const
    {
      promise.register_exception_sink(exception_sink_, this->epoch_context_shared(), explicit_sink);
    }

  private:
    void copy_exception_sink_from(ReadBuffer const& other)
    {
      if (other.exception_sink_.owner)
      {
        other.exception_sink_.owner->register_exception_sink(exception_sink_, other.exception_sink_.epoch,
                                                             other.exception_sink_.explicit_sink);
      }
    }

    void move_exception_sink_from(ReadBuffer& other) noexcept
    {
      if (!other.exception_sink_.owner) return;
      auto* owner = other.exception_sink_.owner;
      auto epoch = std::move(other.exception_sink_.epoch);
      bool const explicit_sink = other.exception_sink_.explicit_sink;
      owner->unregister_exception_sink(other.exception_sink_, false);
      owner->register_exception_sink(exception_sink_, std::move(epoch), explicit_sink);
    }

    void unregister_exception_sink(bool from_destructor) noexcept
    {
      if (!exception_sink_.owner) return;
      exception_sink_.owner->unregister_exception_sink(exception_sink_, from_destructor);
    }

    EpochContextReader<T> reader_; ///< RAII object managing epoch state.
    mutable BasicAsyncTaskPromise::ExceptionSinkNode exception_sink_{};
};

/// \brief Adaptor for transferring ReadBuffer ownership into an await expression.
/// \details This is a synonym for `std::move`.
template <typename T> ReadBuffer<T>&& release(ReadBuffer<T>& in) { return std::move(in); }

template <typename T> ReadBuffer<T>&& release(ReadBuffer<T>&& in) { return std::move(in); }

/// \brief Owning proxy returned by `co_await std::move(read_buffer)`.
/// \details This keeps the read epoch alive until the proxy is released or destroyed.
template <typename T> class OwningReadAccessProxy {
  public:
    using value_type = T;

    OwningReadAccessProxy() = delete;
    OwningReadAccessProxy(OwningReadAccessProxy const&) = delete;
    OwningReadAccessProxy& operator=(OwningReadAccessProxy const&) = delete;
    OwningReadAccessProxy(OwningReadAccessProxy&&) noexcept = default;
    OwningReadAccessProxy& operator=(OwningReadAccessProxy&&) noexcept = delete;

    T const& get() const { return this->reader_.data(); }

    operator T const&() const { return this->get(); }

    T const* operator->() const { return std::addressof(this->get()); }

    /// \brief Copy the value and release the read epoch in one call.
    /// \return A copy of the buffered value.
    T get_release()
    {
      T value = this->get();
      this->release();
      return value;
    }

    void release() noexcept { this->reader_.release(); }

  private:
    explicit OwningReadAccessProxy(EpochContextReader<T>&& reader) : reader_(std::move(reader)) {}

    friend class ReadBuffer<T>;
    friend class OwningReadAwaiter<T>;
    friend class ReadMaybeAwaiter<T>;
    friend class ReadOrCancelAwaiter<T>;

    EpochContextReader<T> reader_;
};

/// \brief Awaiter that transfers a read handle into an owning read proxy.
template <typename T> class OwningReadAwaiter {
  public:
    using value_type = T;

    explicit OwningReadAwaiter(EpochContextReader<T>&& reader) : reader_(std::move(reader)) {}

    bool await_ready() const noexcept { return this->reader_.ready(); }

    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "OwningReadAwaiter::await_suspend()", this, t.h_);
      this->reader_.suspend(std::move(t), false);
    }

    OwningReadAccessProxy<T> await_resume() { return OwningReadAccessProxy<T>(std::move(this->reader_)); }

  private:
    EpochContextReader<T> reader_;
};

template <typename T> OwningReadAccessProxy<T> ReadBuffer<T>::await_resume() &&
{
  return OwningReadAccessProxy<T>(std::move(reader_));
}

template <typename T> class ReadMaybeAwaiter {
  public:
    using value_type = std::optional<OwningReadAccessProxy<T>>;

    ReadMaybeAwaiter(ReadMaybeAwaiter&&) = default; // movable

    /// \brief Check if the value is already ready to be read.
    /// \return True if the epoch is ready and no suspension is needed.
    bool await_ready() const noexcept { return reader_.ready(); }

    /// \brief Suspend this coroutine and enqueue for resumption.
    /// \param t Coroutine task to enqueue.
    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "ReadBuffer::await_suspend()", this, t.h_);
      this->reader_.suspend(std::move(t), false);
    }

    /// \brief Resume execution and return an optional owning read proxy.
    value_type await_resume()
    {
      if (this->reader_.data_maybe() == nullptr)
      {
        this->reader_.release();
        return std::nullopt;
      }
      return value_type(OwningReadAccessProxy<T>(std::move(this->reader_)));
    }

  private:
    ReadMaybeAwaiter() = delete;
    ReadMaybeAwaiter(ReadMaybeAwaiter const&) = delete;
    ReadMaybeAwaiter& operator=(ReadMaybeAwaiter const&) = delete;
    ReadMaybeAwaiter& operator=(ReadMaybeAwaiter&&) = delete;

    ReadMaybeAwaiter(EpochContextReader<T>&& reader) : reader_(std::move(reader)) {}

    friend ReadBuffer<T>;

    EpochContextReader<T> reader_; ///< RAII object managing epoch state.
};

template <typename T> class ReadMaybeAwaiter<T const&> {
  public:
    using value_type = T const*;

    ReadMaybeAwaiter(ReadMaybeAwaiter&&) = default; // movable

    /// \brief Check if the value is already ready to be read.
    /// \return True if the epoch is ready and no suspension is needed.
    bool await_ready() const noexcept { return reader_.ready(); }

    /// \brief Suspend this coroutine and enqueue for resumption.
    /// \param t Coroutine task to enqueue.
    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "ReadBuffer::await_suspend()", this, t.h_);
      reader_.suspend(std::move(t), false);
    }

    /// \brief Resume execution and return a pointer to stored value, or nullptr
    value_type await_resume() { return reader_.data_maybe(); }

  private:
    ReadMaybeAwaiter() = delete;
    ReadMaybeAwaiter(ReadMaybeAwaiter const&) = delete;
    ReadMaybeAwaiter& operator=(ReadMaybeAwaiter const&) = delete;
    ReadMaybeAwaiter& operator=(ReadMaybeAwaiter&&) = delete;

    ReadMaybeAwaiter(EpochContextReader<T>& reader) : reader_(reader) {}

    friend ReadBuffer<T>;

    EpochContextReader<T>& reader_; ///< RAII object managing epoch state.
};

template <typename T> class ReadOrCancelAwaiter {
  public:
    using value_type = OwningReadAccessProxy<T>;

    ReadOrCancelAwaiter(ReadOrCancelAwaiter&&) = default; // movable

    /// \brief Check if the value is already ready to be read.
    /// \return true if the epoch is ready and no suspension is needed.
    bool await_ready() const noexcept
    {
      // we must suspend here, because it is a possible cancellation point
      return false;
    }

    /// \brief Suspend this coroutine and enqueue for resumption.
    /// \param t Coroutine task to enqueue.
    void await_suspend(AsyncTask&& t) noexcept { this->reader_.suspend(std::move(t), true); }

    /// \brief Resume execution and return an owning read proxy.
    /// \throws task_cancelled when the read source was cancelled.
    value_type await_resume()
    {
      T const* ptr = this->reader_.data_maybe();
      if (ptr == nullptr)
      {
        this->reader_.release();
        throw task_cancelled();
      }
      return value_type(std::move(this->reader_));
    }

  private:
    ReadOrCancelAwaiter() = delete;
    ReadOrCancelAwaiter(ReadOrCancelAwaiter const&) = delete;
    ReadOrCancelAwaiter& operator=(ReadOrCancelAwaiter const&) = delete;
    ReadOrCancelAwaiter& operator=(ReadOrCancelAwaiter&&) = delete;

    ReadOrCancelAwaiter(EpochContextReader<T>&& reader) : reader_(std::move(reader)) {}

    friend ReadBuffer<T>;

    EpochContextReader<T> reader_; ///< RAII object managing epoch state.
};

template <typename T> class ReadOrCancelAwaiter<T const&> {
  public:
    using value_type = T;

    ReadOrCancelAwaiter(ReadOrCancelAwaiter&&) = default; // movable

    /// \brief Check if the value is already ready to be read.
    /// \return true if the epoch is ready and no suspension is needed.
    bool await_ready() const noexcept
    {
      // we must suspend here, because it is a possible cancellation point
      return false;
    }

    /// \brief Suspend this coroutine and enqueue for resumption.
    /// \param t Coroutine task to enqueue.
    void await_suspend(AsyncTask&& t) noexcept { reader_.suspend(std::move(t), true); }

    /// \brief Resume execution and return a borrowed reference to the stored value.
    T const& await_resume()
    {
      T const* ptr = reader_.data_maybe();
      if (!ptr) throw task_cancelled();
      return *ptr;
    }

  private:
    ReadOrCancelAwaiter() = delete;
    ReadOrCancelAwaiter(ReadOrCancelAwaiter const&) = delete;
    ReadOrCancelAwaiter& operator=(ReadOrCancelAwaiter const&) = delete;
    ReadOrCancelAwaiter& operator=(ReadOrCancelAwaiter&&) = delete;

    ReadOrCancelAwaiter(EpochContextReader<T>& reader) : reader_(reader) {}

    friend ReadBuffer<T>;

    EpochContextReader<T>& reader_; ///< RAII object managing epoch state.
};

template <typename T> ReadMaybeAwaiter<T const&> ReadBuffer<T>::maybe() &
{
  return ReadMaybeAwaiter<T const&>(reader_);
}

template <typename T> ReadMaybeAwaiter<T> ReadBuffer<T>::maybe() && { return ReadMaybeAwaiter<T>(std::move(reader_)); }

template <typename T> ReadOrCancelAwaiter<T const&> ReadBuffer<T>::or_cancel() &
{
  return ReadOrCancelAwaiter<T const&>(reader_);
}

template <typename T> ReadOrCancelAwaiter<T> ReadBuffer<T>::or_cancel() && { return ReadOrCancelAwaiter<T>(std::move(reader_)); }

// Forward declaration of the proxy used for deferred writes
template <typename T> class WriteAccessProxy;
template <typename T> class OwningWriteAccessProxy;
template <typename T> class OwningWriteAwaiter;
template <typename T> class OwningStorageAccessProxy;
template <typename T> class OwningStorageAwaiter;
template <typename T> class WriteAssignProxy;

#if UNI20_DEBUG_ASYNC_TASKS
struct WriteProxyLifetimeState
{
    std::atomic<bool> alive{true};
};
#endif

template <typename T> class StorageAwaiter {
  public:
    StorageAwaiter(EpochContextWriter<T>* writer) : writer_(writer) {}

    bool await_ready() const noexcept { return writer_->ready(); }

    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "StorageAwaiter::await_suspend()", this, t.h_);
      writer_->suspend(std::move(t), false);
    }

    shared_storage<T>& await_resume()
    {
      writer_->resume();
      return writer_->storage();
    }

    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return writer_->epoch_context_shared(); }

  private:
    EpochContextWriter<T>* writer_; // by pointer, since we don't want to take ownership
};

template <typename T> class TakeAwaiter {
  public:
    TakeAwaiter(EpochContextWriter<T>* writer) : writer_(writer) {}

    bool await_ready() const noexcept { return writer_->ready(); }

    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "TakeAwaiter::await_suspend()", this, t.h_);
      writer_->suspend(std::move(t), false);
    }

    T await_resume()
    {
      writer_->resume();
      return writer_->storage().take();
    }

    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return writer_->epoch_context_shared(); }

  private:
    EpochContextWriter<T>* writer_; // by pointer, since we don't want to take ownership
};

template <typename T> class OwningStorageAccessProxy {
  public:
    using value_type = shared_storage<T>;

    OwningStorageAccessProxy() = delete;
    OwningStorageAccessProxy(OwningStorageAccessProxy const&) = delete;
    OwningStorageAccessProxy& operator=(OwningStorageAccessProxy const&) = delete;
    OwningStorageAccessProxy(OwningStorageAccessProxy&&) noexcept = default;
    OwningStorageAccessProxy& operator=(OwningStorageAccessProxy&&) noexcept = delete;

    shared_storage<T>& get() { return writer_.storage(); }
    shared_storage<T> const& get() const { return writer_.storage(); }

    operator shared_storage<T>&() { return this->get(); }
    operator shared_storage<T> const&() const { return this->get(); }

    shared_storage<T>* operator->() { return std::addressof(this->get()); }
    shared_storage<T> const* operator->() const { return std::addressof(this->get()); }

    void release() noexcept { writer_.release(); }

  private:
    explicit OwningStorageAccessProxy(EpochContextWriter<T>&& writer) : writer_(std::move(writer)) {}

    friend class OwningStorageAwaiter<T>;

    EpochContextWriter<T> writer_;
};

template <typename T> class OwningStorageAwaiter {
  public:
    using value_type = shared_storage<T>;

    explicit OwningStorageAwaiter(EpochContextWriter<T>&& writer) : writer_(std::move(writer)) {}

    bool await_ready() const noexcept { return writer_.ready(); }

    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "OwningStorageAwaiter::await_suspend()", this, t.h_);
      writer_.suspend(std::move(t), false);
    }

    OwningStorageAccessProxy<T> await_resume()
    {
      writer_.resume();
      return OwningStorageAccessProxy<T>(std::move(writer_));
    }

    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return writer_.epoch_context_shared(); }

  private:
    EpochContextWriter<T> writer_;
};

/// \brief Awaitable write buffer for initializing uninitialized storage.
///
/// A WriteBuffer enforces that the underlying value is treated as uninitialized
/// until a write occurs. The constructor marks the epoch as requiring a write,
/// and consumers are expected to assign a value before releasing the buffer.
template <typename T> class WriteBuffer {
  public:
    using value_type = T;
    using element_type = T&;

    explicit WriteBuffer(EpochContextWriter<T> writer) : writer_(std::move(writer)) {}

    ~WriteBuffer() noexcept
    {
      this->invalidate_proxy_state();
      this->unregister_exception_sink(true);
    }

    WriteBuffer(WriteBuffer const&) = delete;
    WriteBuffer& operator=(WriteBuffer const&) = delete;

    WriteBuffer(WriteBuffer&& other) noexcept : writer_(std::move(other.writer_))
    {
      this->move_exception_sink_from(other);
      other.invalidate_proxy_state();
    }

    WriteBuffer& operator=(WriteBuffer&& other) noexcept
    {
      if (this != &other)
      {
        this->invalidate_proxy_state();
        this->unregister_exception_sink(false);
        writer_ = std::move(other.writer_);
        this->move_exception_sink_from(other);
        this->reset_proxy_state();
        other.invalidate_proxy_state();
      }
      return *this;
    }

#if UNI20_DEBUG_DAG
    NodeInfo const* node() const { return writer_.node(); }
#endif

    bool await_ready() const noexcept { return writer_.ready(); }

    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "WriteBuffer::await_suspend()", this, t.h_);
      writer_.suspend(std::move(t), false);
    }

    WriteAccessProxy<T> await_resume() &
    {
      writer_.resume();
      return WriteAccessProxy<T>(
          &writer_
#if UNI20_DEBUG_ASYNC_TASKS
          ,
          proxy_state_
#endif
      );
    }

    WriteAccessProxy<T> await_resume() const&
    {
      auto* writer = const_cast<EpochContextWriter<T>*>(&writer_);
      writer->resume();
      return WriteAccessProxy<T>(
          writer
#if UNI20_DEBUG_ASYNC_TASKS
          ,
          proxy_state_
#endif
      );
    }

    OwningWriteAccessProxy<T> await_resume() &&
    {
      writer_.resume();
      return OwningWriteAccessProxy<T>(std::move(writer_));
    }

    void release() noexcept
    {
      this->invalidate_proxy_state();
      writer_.release();
    }

    template <typename... Args>
    requires std::constructible_from<T, Args...> T& emplace_assert(Args&&... args)
    {
      DEBUG_CHECK(writer_.ready(), "WriteBuffer must be immediately writable");
      writer_.emplace(std::forward<Args>(args)...);
      return writer_.data();
    }

    auto take() & { return TakeAwaiter<T>(&writer_); }

    auto storage() & { return StorageAwaiter<T>(&writer_); }

    auto storage() && -> OwningStorageAwaiter<T>
    {
      this->invalidate_proxy_state();
      return OwningStorageAwaiter<T>(std::move(writer_));
    }

    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return writer_.epoch_context_shared(); }

    void register_exception_sink(BasicAsyncTaskPromise& promise, bool explicit_sink) const
    {
      promise.register_exception_sink(exception_sink_, this->epoch_context_shared(), explicit_sink);
    }

    T move_from_wait() { return writer_.move_from_wait(); }

    template <typename U> void write(U&& val) { async_assign(std::forward<U>(val), std::move(*this)); }

    template <typename U> void write_assert(U&& val) requires std::assignable_from<T&, U&&>
    {
      DEBUG_CHECK(writer_.ready(), "WriteBuffer must be immediately writable");
      writer_.data() = std::forward<U>(val);
    }

    template <typename U> void write_move_assert(U&& val) requires std::assignable_from<T&, U&&>
    {
      DEBUG_CHECK(writer_.ready(), "WriteBuffer must be immediately writable");
      writer_.data() = std::move(val);
    }

    [[nodiscard]] WriteAssignProxy<T> write();

    template <typename U> void write_move(U&& val) { async_move(std::move(val), std::move(*this)); }

    auto operator co_await() & noexcept -> WriteBuffer& { return *this; }
    auto operator co_await() const& noexcept -> WriteBuffer const& { return *this; }
    auto operator co_await() && noexcept -> OwningWriteAwaiter<T>
    {
      this->invalidate_proxy_state();
      return OwningWriteAwaiter<T>(std::move(writer_));
    }

  private:
    void move_exception_sink_from(WriteBuffer& other) noexcept
    {
      if (!other.exception_sink_.owner) return;
      auto* owner = other.exception_sink_.owner;
      auto epoch = std::move(other.exception_sink_.epoch);
      bool const explicit_sink = other.exception_sink_.explicit_sink;
      owner->unregister_exception_sink(other.exception_sink_, false);
      owner->register_exception_sink(exception_sink_, std::move(epoch), explicit_sink);
    }

    void unregister_exception_sink(bool from_destructor) noexcept
    {
      if (!exception_sink_.owner) return;
      exception_sink_.owner->unregister_exception_sink(exception_sink_, from_destructor);
    }

    void invalidate_proxy_state() noexcept
    {
#if UNI20_DEBUG_ASYNC_TASKS
      if (proxy_state_) proxy_state_->alive.store(false, std::memory_order_release);
#endif
    }

    void reset_proxy_state() noexcept
    {
#if UNI20_DEBUG_ASYNC_TASKS
      proxy_state_ = std::make_shared<WriteProxyLifetimeState>();
#endif
    }

    EpochContextWriter<T> writer_;
    mutable BasicAsyncTaskPromise::ExceptionSinkNode exception_sink_{};
#if UNI20_DEBUG_ASYNC_TASKS
    std::shared_ptr<WriteProxyLifetimeState> proxy_state_{std::make_shared<WriteProxyLifetimeState>()};
#endif

    // Duplicating a WriteBuffer is no longer possible with the refactored EpochContext.
    // To re-enable this function we would need to make the
    // friend WriteBuffer dup(WriteBuffer& wb) { return WriteBuffer(wb.writer_); }
};

// For a ReadBuffer, we add the node to the ReadDependencies
template <typename T> void ProcessCoroutineArgument(BasicAsyncTaskPromise* promise, ReadBuffer<T> const& x)
{
#if UNI20_DEBUG_DAG
  promise->ReadDependencies.push_back(x.node());
#endif
}

// For a WriteBuffer, we add the node to the WriteDependencies
template <typename T> void ProcessCoroutineArgument(BasicAsyncTaskPromise* promise, WriteBuffer<T> const& x)
{
#if UNI20_DEBUG_DAG
  promise->WriteDependencies.push_back(x.node());
#endif
  x.register_exception_sink(*promise, false);
}

template <typename T> class Defer;

template <typename B>
concept exception_sink_buffer = requires(B& buffer, BasicAsyncTaskPromise& promise)
{
  buffer.register_exception_sink(promise, true);
};

template <exception_sink_buffer... Buffers> class PropagateExceptionsAwaiter {
  public:
    explicit PropagateExceptionsAwaiter(Buffers&... buffers) : buffers_(std::addressof(buffers)...) {}

    bool await_ready() const noexcept { return true; }

    AsyncTask await_suspend(AsyncTask&& task) noexcept { return std::move(task); }

    void await_resume() noexcept {}

    void register_exception_sinks(BasicAsyncTaskPromise& promise) const
    {
      std::apply([&](auto*... buffers) { (buffers->register_exception_sink(promise, true), ...); }, buffers_);
    }

  private:
    std::tuple<Buffers*...> buffers_;
};

/// \brief Register buffers that should receive unhandled coroutine exceptions.
/// \details The registration remains active for as long as each buffer object remains alive.
/// \note If a registered explicit sink is destroyed during stack unwinding, the runtime aborts.
template <exception_sink_buffer... Buffers> auto propagate_exceptions_to(Buffers&... buffers)
{
  return PropagateExceptionsAwaiter<Buffers...>(buffers...);
}

/// \brief Non-owning proxy returned by `co_await` on an lvalue `WriteBuffer<T>`.
/// \details This proxy references the underlying writer held by the buffer.
template <typename T> class WriteAccessProxy {
  public:
    using value_type = T;

    WriteAccessProxy() = delete;
    WriteAccessProxy(WriteAccessProxy const&) noexcept = default;
    WriteAccessProxy& operator=(WriteAccessProxy const&) = delete;
    WriteAccessProxy(WriteAccessProxy&&) noexcept = default;
    WriteAccessProxy& operator=(WriteAccessProxy&&) = delete;

    template <typename U>
    requires(!std::same_as<std::remove_cvref_t<U>, WriteAccessProxy>) &&
        std::constructible_from<T, U&&> WriteAccessProxy& operator=(U&& u)
    {
      this->emplace(std::forward<U>(u));
      return *this;
    }

    template <typename... Args>
    requires std::constructible_from<T, Args...> T& emplace(Args&&... args)
    {
      return this->writer().emplace(std::forward<Args>(args)...);
    }

    template <typename U>
    requires requires(T& value, U&& x)
    {
      value += std::forward<U>(x);
    } WriteAccessProxy& operator+=(U&& x)
    {
      auto& storage = this->writer().storage();
      if (storage.constructed())
      {
        this->get() += std::forward<U>(x);
      }
      else if constexpr (std::constructible_from<T, U&&>)
      {
        storage.emplace(std::forward<U>(x));
      }
      else
      {
        this->get() += std::forward<U>(x);
      }
      return *this;
    }

    template <typename U>
    requires requires(T& value, U&& x)
    {
      value -= std::forward<U>(x);
    } WriteAccessProxy& operator-=(U&& x)
    {
      auto& storage = this->writer().storage();
      if (storage.constructed())
      {
        this->get() -= std::forward<U>(x);
      }
      else if constexpr (requires { -std::declval<U&&>(); } &&
                         std::constructible_from<T, decltype(-std::declval<U&&>())>)
      {
        auto neg = -std::forward<U>(x);
        storage.emplace(std::move(neg));
      }
      else
      {
        this->get() -= std::forward<U>(x);
      }
      return *this;
    }

    template <typename U>
    requires requires(T& value, U&& x)
    {
      value *= std::forward<U>(x);
    } WriteAccessProxy& operator*=(U&& x)
    {
      this->get() *= std::forward<U>(x);
      return *this;
    }

    template <typename U>
    requires requires(T& value, U&& x)
    {
      value /= std::forward<U>(x);
    } WriteAccessProxy& operator/=(U&& x)
    {
      this->get() /= std::forward<U>(x);
      return *this;
    }

    T& get() const { return this->writer().data(); }

    operator T&() const { return this->get(); }

    T* operator->() const { return std::addressof(this->get()); }

  private:
    explicit WriteAccessProxy(
        EpochContextWriter<T>* writer
#if UNI20_DEBUG_ASYNC_TASKS
        ,
        std::shared_ptr<WriteProxyLifetimeState> proxy_state
#endif
        ) noexcept
        : writer_(writer)
#if UNI20_DEBUG_ASYNC_TASKS
        , proxy_state_(std::move(proxy_state))
#endif
    {}

    EpochContextWriter<T>& writer() const
    {
#if UNI20_DEBUG_ASYNC_TASKS
      auto state = proxy_state_.lock();
      DEBUG_CHECK(state && state->alive.load(std::memory_order_acquire),
                  "WriteAccessProxy used after WriteBuffer moved, released, or destroyed");
#endif
      DEBUG_CHECK(writer_, "WriteAccessProxy has no writer");
      return *writer_;
    }

    friend class WriteBuffer<T>;

    EpochContextWriter<T>* writer_;
#if UNI20_DEBUG_ASYNC_TASKS
    std::weak_ptr<WriteProxyLifetimeState> proxy_state_;
#endif
};

/// \brief Owning proxy returned by `co_await std::move(write_buffer)`.
/// \details Ownership of the writer handle is transferred into this proxy.
template <typename T> class OwningWriteAccessProxy {
  public:
    using value_type = T;

    OwningWriteAccessProxy() = delete;
    OwningWriteAccessProxy(OwningWriteAccessProxy const&) = delete;
    OwningWriteAccessProxy& operator=(OwningWriteAccessProxy const&) = delete;
    OwningWriteAccessProxy(OwningWriteAccessProxy&&) noexcept = default;
    OwningWriteAccessProxy& operator=(OwningWriteAccessProxy&&) noexcept = delete;

    template <typename U>
    requires(!std::same_as<std::remove_cvref_t<U>, OwningWriteAccessProxy>) &&
        std::constructible_from<T, U&&> OwningWriteAccessProxy& operator=(U&& u)
    {
      this->emplace(std::forward<U>(u));
      return *this;
    }

    template <typename... Args>
    requires std::constructible_from<T, Args...> T& emplace(Args&&... args)
    {
      return writer_.emplace(std::forward<Args>(args)...);
    }

    template <typename U>
    requires requires(T& value, U&& x)
    {
      value += std::forward<U>(x);
    } OwningWriteAccessProxy& operator+=(U&& x)
    {
      auto& storage = writer_.storage();
      if (storage.constructed())
      {
        this->get() += std::forward<U>(x);
      }
      else if constexpr (std::constructible_from<T, U&&>)
      {
        storage.emplace(std::forward<U>(x));
      }
      else
      {
        this->get() += std::forward<U>(x);
      }
      return *this;
    }

    template <typename U>
    requires requires(T& value, U&& x)
    {
      value -= std::forward<U>(x);
    } OwningWriteAccessProxy& operator-=(U&& x)
    {
      auto& storage = writer_.storage();
      if (storage.constructed())
      {
        this->get() -= std::forward<U>(x);
      }
      else if constexpr (requires { -std::declval<U&&>(); } &&
                         std::constructible_from<T, decltype(-std::declval<U&&>())>)
      {
        auto neg = -std::forward<U>(x);
        storage.emplace(std::move(neg));
      }
      else
      {
        this->get() -= std::forward<U>(x);
      }
      return *this;
    }

    template <typename U>
    requires requires(T& value, U&& x)
    {
      value *= std::forward<U>(x);
    } OwningWriteAccessProxy& operator*=(U&& x)
    {
      this->get() *= std::forward<U>(x);
      return *this;
    }

    template <typename U>
    requires requires(T& value, U&& x)
    {
      value /= std::forward<U>(x);
    } OwningWriteAccessProxy& operator/=(U&& x)
    {
      this->get() /= std::forward<U>(x);
      return *this;
    }

    T& get() { return writer_.data(); }
    T const& get() const { return writer_.data(); }

    operator T&() { return this->get(); }
    operator T const&() const { return this->get(); }

    T* operator->() { return std::addressof(this->get()); }
    T const* operator->() const { return std::addressof(this->get()); }

  private:
    explicit OwningWriteAccessProxy(EpochContextWriter<T>&& writer) : writer_(std::move(writer)) {}

    friend class WriteBuffer<T>;
    friend class OwningWriteAwaiter<T>;

    EpochContextWriter<T> writer_;
};

/// \brief Awaiter that transfers a writer handle into an owning write proxy.
template <typename T> class OwningWriteAwaiter {
  public:
    explicit OwningWriteAwaiter(EpochContextWriter<T>&& writer) : writer_(std::move(writer)) {}

    bool await_ready() const noexcept { return writer_.ready(); }

    void await_suspend(AsyncTask&& task) noexcept
    {
      TRACE_MODULE(ASYNC, "OwningWriteAwaiter::await_suspend()", this, task.h_);
      writer_.suspend(std::move(task), false);
    }

    OwningWriteAccessProxy<T> await_resume()
    {
      writer_.resume();
      return OwningWriteAccessProxy<T>(std::move(writer_));
    }

    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return writer_.epoch_context_shared(); }

  private:
    EpochContextWriter<T> writer_;
};

template <typename T> using WriteAwaitProxy = WriteAccessProxy<T>;
template <typename T> using OwningWriteProxy = OwningWriteAccessProxy<T>;

/// \brief Assignment-only proxy returned by `WriteBuffer::write()`.
template <typename T> class WriteAssignProxy {
  public:
    using value_type = T;

    WriteAssignProxy() = delete;
    WriteAssignProxy(WriteAssignProxy const&) = delete;
    WriteAssignProxy& operator=(WriteAssignProxy const&) = delete;
    WriteAssignProxy(WriteAssignProxy&&) noexcept = default;
    WriteAssignProxy& operator=(WriteAssignProxy&&) noexcept = delete;

    template <typename U>
    requires(!std::same_as<std::remove_cvref_t<U>, WriteAssignProxy>) void operator=(U&& u)
    {
      async_assign(std::forward<U>(u), WriteBuffer<T>(std::move(writer_)));
    }

  private:
    explicit WriteAssignProxy(EpochContextWriter<T>&& writer) : writer_(std::move(writer)) {}

    friend class WriteBuffer<T>;

    EpochContextWriter<T> writer_;
};

template <typename T> using WriteProxy = WriteAssignProxy<T>;

template <typename T> WriteAssignProxy<T> WriteBuffer<T>::write()
{
  return WriteAssignProxy<T>(std::move(writer_));
}

} // namespace uni20::async
