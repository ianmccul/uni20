/// \file buffers.hpp
/// \brief Awaitable gates for Async<T>: snapshot‐reads and in‐place writes.

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

    /// \brief Copy constructor; duplicates read handle and exception sink registration.
    /// \param other Source read buffer.
    ReadBuffer(ReadBuffer const& other) : reader_(other.reader_) { this->copy_exception_sink_from(other); }

    // No copy ctor here, although we could add one
    ReadBuffer& operator=(ReadBuffer const&) = delete;

    /// \brief Move constructor.
    /// \param other Source read buffer.
    ReadBuffer(ReadBuffer&& other) noexcept : reader_(std::move(other.reader_))
    {
      this->move_exception_sink_from(other);
    }

    /// \brief Move assignment.
    /// \param other Source read buffer.
    /// \return Reference to `*this`.
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

    /// \brief Destructor unregisters any attached exception sink.
    ~ReadBuffer() noexcept { this->unregister_exception_sink(true); }

#if UNI20_DEBUG_DAG
    /// \brief Get the debug node pointer of the object
    NodeInfo const* node() const { return reader_.node(); }
#endif

    /// \brief Returns a `ReadMaybeAwaiter`.
    /// \details Lvalue reads return `T const*`; moved reads return `std::optional<OwningReadAccessProxy<T>>`.
    ReadMaybeAwaiter<T const&> maybe() &;

    /// \brief Returns a `ReadMaybeAwaiter` consuming this read buffer.
    /// \details Moved reads return `std::optional<OwningReadAccessProxy<T>>`.
    ReadMaybeAwaiter<T> maybe() &&;

    /// \brief Returns a `ReadOrCancelAwaiter`.
    /// \details Lvalue reads return `T const&` and throw `task_cancelled` on cancellation.
    ReadOrCancelAwaiter<T const&> or_cancel() &;

    /// \brief Returns a `ReadOrCancelAwaiter` consuming this read buffer.
    /// \details Moved reads return `OwningReadAccessProxy<T>` and throw `task_cancelled` on cancellation.
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
    /// \brief Block until the read value is available.
    /// \return Const reference to the available value.
    T const& get_wait() const { return reader_.get_wait(); }

    /// \brief Block using an explicit scheduler until the read value is available.
    /// \param sched Scheduler used to drive progress.
    /// \return Const reference to the available value.
    T const& get_wait(IScheduler& sched) const { return reader_.get_wait(sched); }

    /// \brief Enable co_await on lvalue ReadBuffer and return a borrowed reference.
    auto operator co_await() & noexcept -> ReadBuffer& { return *this; }
    auto operator co_await() const& noexcept -> ReadBuffer const& { return *this; }

    /// \brief Enable co_await on rvalue ReadBuffer and transfer ownership to an owning read proxy.
    auto operator co_await() && noexcept -> OwningReadAwaiter<T> { return OwningReadAwaiter<T>(std::move(reader_)); }

    /// \brief Returns the epoch context used for exception propagation.
    /// \return Shared pointer to the epoch context.
    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return reader_.epoch_context_shared(); }

    /// \brief Register this buffer as an exception sink with a promise.
    /// \param promise Promise that owns the sink list.
    /// \param explicit_sink Whether this sink came from `propagate_exceptions_to`.
    void register_exception_sink(BasicAsyncTaskPromise& promise, bool explicit_sink) const
    {
      promise.register_exception_sink(exception_sink_, this->epoch_context_shared(), explicit_sink);
    }

  private:
    /// \brief Copy exception sink registration from another read buffer.
    /// \param other Source buffer.
    void copy_exception_sink_from(ReadBuffer const& other)
    {
      if (other.exception_sink_.owner)
      {
        other.exception_sink_.owner->register_exception_sink(exception_sink_, other.exception_sink_.epoch,
                                                             other.exception_sink_.explicit_sink);
      }
    }

    /// \brief Move exception sink registration from another read buffer.
    /// \param other Source buffer.
    void move_exception_sink_from(ReadBuffer& other) noexcept
    {
      if (!other.exception_sink_.owner) return;
      auto* owner = other.exception_sink_.owner;
      auto epoch = std::move(other.exception_sink_.epoch);
      bool const explicit_sink = other.exception_sink_.explicit_sink;
      owner->unregister_exception_sink(other.exception_sink_, false);
      owner->register_exception_sink(exception_sink_, std::move(epoch), explicit_sink);
    }

    /// \brief Unregister this buffer's exception sink from its owning promise.
    /// \param from_destructor Whether called from destructor context.
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

/// \brief Adaptor for forwarding an rvalue ReadBuffer into an await expression.
/// \details This is a synonym for `std::move`.
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

    /// \brief Access the referenced value.
    /// \return Const reference to the buffered value.
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

    /// \brief Release the read epoch held by this proxy.
    void release() noexcept { this->reader_.release(); }

  private:
    /// \brief Construct from an owning epoch reader.
    /// \param reader Reader handle transferred into this proxy.
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

    /// \brief Construct from an owning epoch reader.
    /// \param reader Reader handle transferred into this awaiter.
    explicit OwningReadAwaiter(EpochContextReader<T>&& reader) : reader_(std::move(reader)) {}

    /// \brief Reports whether the read epoch is immediately ready.
    /// \return `true` when no suspension is needed.
    bool await_ready() const noexcept { return this->reader_.ready(); }

    /// \brief Suspend until the read epoch becomes available.
    /// \param t Owning task to suspend and enqueue.
    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "OwningReadAwaiter::await_suspend()", this, t.h_);
      this->reader_.suspend(std::move(t), false);
    }

    /// \brief Resume and transfer reader ownership to an owning read proxy.
    /// \return Owning read proxy.
    OwningReadAccessProxy<T> await_resume() { return OwningReadAccessProxy<T>(std::move(this->reader_)); }

  private:
    EpochContextReader<T> reader_;
};

template <typename T> OwningReadAccessProxy<T> ReadBuffer<T>::await_resume() &&
{
  return OwningReadAccessProxy<T>(std::move(reader_));
}

/// \brief Awaiter returning an optional owning read proxy.
/// \tparam T Stored value type.
template <typename T> class ReadMaybeAwaiter {
  public:
    using value_type = std::optional<OwningReadAccessProxy<T>>;

    /// \brief Move constructor.
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

/// \brief Awaiter returning pointer-or-null read access for lvalue buffers.
/// \tparam T Stored value type.
template <typename T> class ReadMaybeAwaiter<T const&> {
  public:
    using value_type = T const*;

    /// \brief Move constructor.
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

/// \brief Awaiter returning owning read access or throwing `task_cancelled`.
/// \tparam T Stored value type.
template <typename T> class ReadOrCancelAwaiter {
  public:
    using value_type = OwningReadAccessProxy<T>;

    /// \brief Move constructor.
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

/// \brief Awaiter returning borrowed read access or throwing `task_cancelled`.
/// \tparam T Stored value type.
template <typename T> class ReadOrCancelAwaiter<T const&> {
  public:
    using value_type = T;

    /// \brief Move constructor.
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

/// \brief Build a maybe-awaiter from an rvalue read buffer.
/// \tparam T Stored value type.
/// \return Awaiter yielding optional owning read access.
template <typename T> ReadMaybeAwaiter<T> ReadBuffer<T>::maybe() && { return ReadMaybeAwaiter<T>(std::move(reader_)); }

template <typename T> ReadOrCancelAwaiter<T const&> ReadBuffer<T>::or_cancel() &
{
  return ReadOrCancelAwaiter<T const&>(reader_);
}

/// \brief Build an or-cancel awaiter from an rvalue read buffer.
/// \tparam T Stored value type.
/// \return Awaiter yielding owning read access or throwing on cancellation.
template <typename T> ReadOrCancelAwaiter<T> ReadBuffer<T>::or_cancel() && { return ReadOrCancelAwaiter<T>(std::move(reader_)); }

// Forward declaration of the proxy used for deferred writes
template <typename T> class WriteAccessProxy;
template <typename T> class OwningWriteAccessProxy;
template <typename T> class OwningWriteAwaiter;
template <typename T> class OwningStorageAccessProxy;
template <typename T> class OwningStorageAwaiter;
template <typename T> class OwningTakeAwaiter;
template <typename T> class WriteAssignProxy;

#if UNI20_DEBUG_ASYNC_TASKS
/// \brief Debug lifetime token shared by `WriteAccessProxy` instances.
struct WriteProxyLifetimeState
{
    std::atomic<bool> alive{true};
};
#endif

/// \brief Awaiter returning direct access to a writer's shared storage object.
/// \tparam T Stored value type.
template <typename T> class StorageAwaiter {
  public:
    /// \brief Construct from a non-owning writer pointer.
    /// \param writer Writer handle owned by the parent `WriteBuffer`.
    StorageAwaiter(EpochContextWriter<T>* writer) : writer_(writer) {}

    /// \brief Reports whether the writer epoch is immediately writable.
    /// \return `true` when no suspension is needed.
    bool await_ready() const noexcept { return writer_->ready(); }

    /// \brief Suspend until the writer epoch becomes writable.
    /// \param t Owning task to suspend and enqueue.
    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "StorageAwaiter::await_suspend()", this, t.h_);
      writer_->suspend(std::move(t), false);
    }

    /// \brief Resume and return mutable access to shared storage.
    /// \return Reference to the writer storage.
    shared_storage<T>& await_resume()
    {
      writer_->resume();
      return writer_->storage();
    }

    /// \brief Returns the epoch context used for exception propagation.
    /// \return Shared pointer to the epoch context.
    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return writer_->epoch_context_shared(); }

  private:
    EpochContextWriter<T>* writer_; // by pointer, since we don't want to take ownership
};

/// \brief Awaiter that takes the writer value without releasing the writer epoch.
/// \tparam T Stored value type.
template <typename T> class TakeAwaiter {
  public:
    /// \brief Construct from a non-owning writer pointer.
    /// \param writer Writer handle owned by the parent `WriteBuffer`.
    TakeAwaiter(EpochContextWriter<T>* writer) : writer_(writer) {}

    /// \brief Reports whether the writer epoch is immediately writable.
    /// \return `true` when no suspension is needed.
    bool await_ready() const noexcept { return writer_->ready(); }

    /// \brief Suspend until the writer epoch becomes writable.
    /// \param t Owning task to suspend and enqueue.
    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "TakeAwaiter::await_suspend()", this, t.h_);
      writer_->suspend(std::move(t), false);
    }

    /// \brief Resume and move the stored value out of writer storage.
    /// \return Moved value.
    T await_resume()
    {
      writer_->resume();
      return writer_->storage().take();
    }

    /// \brief Returns the epoch context used for exception propagation.
    /// \return Shared pointer to the epoch context.
    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return writer_->epoch_context_shared(); }

  private:
    EpochContextWriter<T>* writer_; // by pointer, since we don't want to take ownership
};

/// \brief Awaiter that takes a value and immediately releases the writer epoch.
/// \tparam T Stored value type.
template <typename T> class TakeReleaseAwaiter {
  public:
    /// \brief Construct from a non-owning writer pointer.
    /// \param writer Writer handle owned by the parent `WriteBuffer`.
    TakeReleaseAwaiter(EpochContextWriter<T>* writer) : writer_(writer) {}

    /// \brief Reports whether the writer epoch is immediately writable.
    /// \return `true` when no suspension is needed.
    bool await_ready() const noexcept { return writer_->ready(); }

    /// \brief Suspend until the writer epoch becomes writable.
    /// \param t Owning task to suspend and enqueue.
    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "TakeReleaseAwaiter::await_suspend()", this, t.h_);
      writer_->suspend(std::move(t), false);
    }

    /// \brief Resume, move out the value, and release the writer epoch.
    /// \return Moved value.
    T await_resume()
    {
      writer_->resume();
      T value = writer_->storage().take();
      writer_->release();
      return value;
    }

    /// \brief Returns the epoch context used for exception propagation.
    /// \return Shared pointer to the epoch context.
    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return writer_->epoch_context_shared(); }

  private:
    EpochContextWriter<T>* writer_; // by pointer, since we don't want to take ownership
};

/// \brief Owning proxy providing access to writer storage.
/// \tparam T Stored value type.
template <typename T> class OwningStorageAccessProxy {
  public:
    using value_type = shared_storage<T>;

    OwningStorageAccessProxy() = delete;
    OwningStorageAccessProxy(OwningStorageAccessProxy const&) = delete;
    OwningStorageAccessProxy& operator=(OwningStorageAccessProxy const&) = delete;
    OwningStorageAccessProxy(OwningStorageAccessProxy&&) noexcept = default;
    OwningStorageAccessProxy& operator=(OwningStorageAccessProxy&&) noexcept = delete;

    /// \brief Returns mutable access to the underlying shared storage.
    /// \return Reference to writer storage.
    shared_storage<T>& get() { return writer_.storage(); }
    /// \brief Returns const access to the underlying shared storage.
    /// \return Const reference to writer storage.
    shared_storage<T> const& get() const { return writer_.storage(); }

    operator shared_storage<T>&() { return this->get(); }
    operator shared_storage<T> const&() const { return this->get(); }

    shared_storage<T>* operator->() { return std::addressof(this->get()); }
    shared_storage<T> const* operator->() const { return std::addressof(this->get()); }

    /// \brief Move the stored value out of writer storage.
    /// \return Moved value.
    T take() { return writer_.storage().take(); }

    /// \brief Move the stored value out and release the writer epoch.
    /// \return Moved value.
    T take_release()
    {
      T value = this->take();
      this->release();
      return value;
    }

    /// \brief Release the writer epoch held by this proxy.
    void release() noexcept { writer_.release(); }

  private:
    /// \brief Construct from an owning writer.
    /// \param writer Writer handle transferred into this proxy.
    explicit OwningStorageAccessProxy(EpochContextWriter<T>&& writer) : writer_(std::move(writer)) {}

    friend class OwningStorageAwaiter<T>;

    EpochContextWriter<T> writer_;
};

/// \brief Awaiter that yields an owning storage-access proxy.
/// \tparam T Stored value type.
template <typename T> class OwningStorageAwaiter {
  public:
    using value_type = shared_storage<T>;

    /// \brief Construct from an owning writer.
    /// \param writer Writer handle transferred into this awaiter.
    explicit OwningStorageAwaiter(EpochContextWriter<T>&& writer) : writer_(std::move(writer)) {}

    /// \brief Reports whether the writer epoch is immediately writable.
    /// \return `true` when no suspension is needed.
    bool await_ready() const noexcept { return writer_.ready(); }

    /// \brief Suspend until the writer epoch becomes writable.
    /// \param t Owning task to suspend and enqueue.
    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "OwningStorageAwaiter::await_suspend()", this, t.h_);
      writer_.suspend(std::move(t), false);
    }

    /// \brief Resume and transfer writer ownership to a storage proxy.
    /// \return Owning storage access proxy.
    OwningStorageAccessProxy<T> await_resume()
    {
      writer_.resume();
      return OwningStorageAccessProxy<T>(std::move(writer_));
    }

    /// \brief Returns the epoch context used for exception propagation.
    /// \return Shared pointer to the epoch context.
    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return writer_.epoch_context_shared(); }

  private:
    EpochContextWriter<T> writer_;
};

/// \brief Awaiter that takes and releases a value from an owning writer.
/// \tparam T Stored value type.
template <typename T> class OwningTakeAwaiter {
  public:
    /// \brief Construct from an owning writer.
    /// \param writer Writer handle transferred into this awaiter.
    explicit OwningTakeAwaiter(EpochContextWriter<T>&& writer) : writer_(std::move(writer)) {}

    /// \brief Reports whether the writer epoch is immediately writable.
    /// \return `true` when no suspension is needed.
    bool await_ready() const noexcept { return writer_.ready(); }

    /// \brief Suspend until the writer epoch becomes writable.
    /// \param t Owning task to suspend and enqueue.
    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "OwningTakeAwaiter::await_suspend()", this, t.h_);
      writer_.suspend(std::move(t), false);
    }

    /// \brief Resume, move out the value, and release the writer epoch.
    /// \return Moved value.
    T await_resume()
    {
      writer_.resume();
      T value = writer_.storage().take();
      writer_.release();
      return value;
    }

    /// \brief Returns the epoch context used for exception propagation.
    /// \return Shared pointer to the epoch context.
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

    /// \brief Construct a write buffer tied to a writer context.
    /// \param writer RAII writer handle for this operation.
    explicit WriteBuffer(EpochContextWriter<T> writer) : writer_(std::move(writer)) {}

    /// \brief Destructor unregisters exception sink and invalidates debug proxy state.
    ~WriteBuffer() noexcept
    {
      this->invalidate_proxy_state();
      this->unregister_exception_sink(true);
    }

    WriteBuffer(WriteBuffer const&) = delete;
    WriteBuffer& operator=(WriteBuffer const&) = delete;

    /// \brief Move constructor.
    /// \param other Source write buffer.
    WriteBuffer(WriteBuffer&& other) noexcept : writer_(std::move(other.writer_))
    {
      this->move_exception_sink_from(other);
      other.invalidate_proxy_state();
    }

    /// \brief Move assignment.
    /// \param other Source write buffer.
    /// \return Reference to `*this`.
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

    /// \brief Reports whether the writer epoch is immediately writable.
    /// \return `true` when no suspension is needed.
    bool await_ready() const noexcept { return writer_.ready(); }

    /// \brief Suspend until the writer epoch becomes writable.
    /// \param t Owning task to suspend and enqueue.
    void await_suspend(AsyncTask&& t) noexcept
    {
      TRACE_MODULE(ASYNC, "WriteBuffer::await_suspend()", this, t.h_);
      writer_.suspend(std::move(t), false);
    }

    /// \brief Resume and return a non-owning mutable write proxy.
    /// \return Non-owning write proxy.
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

    /// \brief Resume and return a non-owning mutable write proxy.
    /// \return Non-owning write proxy.
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

    /// \brief Resume and return an owning mutable write proxy.
    /// \return Owning write proxy.
    OwningWriteAccessProxy<T> await_resume() &&
    {
      writer_.resume();
      return OwningWriteAccessProxy<T>(std::move(writer_));
    }

    /// \brief Release the writer epoch held by this buffer.
    void release() noexcept
    {
      this->invalidate_proxy_state();
      writer_.release();
    }

    /// \brief Construct the destination value in-place when immediately writable.
    /// \tparam Args Constructor argument types.
    /// \param args Constructor arguments.
    /// \return Reference to the constructed value.
    template <typename... Args>
    requires std::constructible_from<T, Args...> T& emplace_assert(Args&&... args)
    {
      DEBUG_CHECK(writer_.ready(), "WriteBuffer must be immediately writable");
      writer_.emplace(std::forward<Args>(args)...);
      return writer_.data();
    }

    /// \brief Awaiter that takes the stored value.
    /// \return Awaiter yielding a moved value.
    auto take() & { return TakeAwaiter<T>(&writer_); }

    /// \brief Awaiter that takes the stored value from an rvalue buffer.
    /// \return Owning take awaiter.
    auto take() && -> OwningTakeAwaiter<T>
    {
      this->invalidate_proxy_state();
      return OwningTakeAwaiter<T>(std::move(writer_));
    }

    /// \brief Awaiter that takes the value and releases the writer epoch.
    /// \return Awaiter yielding a moved value.
    auto take_release() &
    {
      this->invalidate_proxy_state();
      return TakeReleaseAwaiter<T>(&writer_);
    }

    /// \brief Awaiter that takes the value and releases the writer epoch from an rvalue buffer.
    /// \return Owning take awaiter.
    auto take_release() && -> OwningTakeAwaiter<T>
    {
      this->invalidate_proxy_state();
      return OwningTakeAwaiter<T>(std::move(writer_));
    }

    /// \brief Awaiter yielding direct access to writer shared storage.
    /// \return Storage awaiter.
    auto storage() & { return StorageAwaiter<T>(&writer_); }

    /// \brief Awaiter yielding owning access to writer shared storage.
    /// \return Owning storage awaiter.
    auto storage() && -> OwningStorageAwaiter<T>
    {
      this->invalidate_proxy_state();
      return OwningStorageAwaiter<T>(std::move(writer_));
    }

    /// \brief Returns the epoch context used for exception propagation.
    /// \return Shared pointer to the epoch context.
    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return writer_.epoch_context_shared(); }

    /// \brief Register this buffer as an exception sink with a promise.
    /// \param promise Promise that owns the sink list.
    /// \param explicit_sink Whether this sink came from `propagate_exceptions_to`.
    void register_exception_sink(BasicAsyncTaskPromise& promise, bool explicit_sink) const
    {
      promise.register_exception_sink(exception_sink_, this->epoch_context_shared(), explicit_sink);
    }

    /// \brief Wait for writability and move out the contained value.
    /// \return Moved value.
    T move_from_wait() { return writer_.move_from_wait(); }

    /// \brief Schedule assignment into this write buffer.
    /// \tparam U Source type.
    /// \param val Source expression.
    template <typename U> void write(U&& val) { async_assign(std::forward<U>(val), std::move(*this)); }

    /// \brief Assign immediately when the writer is known to be ready.
    /// \tparam U Source type assignable to `T&`.
    /// \param val Source value.
    template <typename U> void write_assert(U&& val) requires std::assignable_from<T&, U&&>
    {
      DEBUG_CHECK(writer_.ready(), "WriteBuffer must be immediately writable");
      writer_.data() = std::forward<U>(val);
    }

    /// \brief Move-assign immediately when the writer is known to be ready.
    /// \tparam U Source type assignable to `T&`.
    /// \param val Source value.
    template <typename U> void write_move_assert(U&& val) requires std::assignable_from<T&, U&&>
    {
      DEBUG_CHECK(writer_.ready(), "WriteBuffer must be immediately writable");
      writer_.data() = std::move(val);
    }

    /// \brief Build an assignment-only proxy that consumes this writer.
    /// \return Assignment-only proxy.
    [[nodiscard]] WriteAssignProxy<T> write();

    /// \brief Schedule move-assignment into this write buffer.
    /// \tparam U Source type.
    /// \param val Source expression.
    template <typename U> void write_move(U&& val) { async_move(std::move(val), std::move(*this)); }

    /// \brief Enable `co_await` for lvalue write buffers.
    /// \return Reference to this buffer.
    auto operator co_await() & noexcept -> WriteBuffer& { return *this; }
    /// \brief Enable `co_await` for const lvalue write buffers.
    /// \return Const reference to this buffer.
    auto operator co_await() const& noexcept -> WriteBuffer const& { return *this; }
    /// \brief Enable `co_await` for rvalue write buffers and transfer ownership.
    /// \return Owning write awaiter.
    auto operator co_await() && noexcept -> OwningWriteAwaiter<T>
    {
      this->invalidate_proxy_state();
      return OwningWriteAwaiter<T>(std::move(writer_));
    }

  private:
    /// \brief Move registered exception sink linkage from another buffer.
    /// \param other Source buffer.
    void move_exception_sink_from(WriteBuffer& other) noexcept
    {
      if (!other.exception_sink_.owner) return;
      auto* owner = other.exception_sink_.owner;
      auto epoch = std::move(other.exception_sink_.epoch);
      bool const explicit_sink = other.exception_sink_.explicit_sink;
      owner->unregister_exception_sink(other.exception_sink_, false);
      owner->register_exception_sink(exception_sink_, std::move(epoch), explicit_sink);
    }

    /// \brief Unregister this buffer's exception sink from its owning promise.
    /// \param from_destructor Whether called from destructor context.
    void unregister_exception_sink(bool from_destructor) noexcept
    {
      if (!exception_sink_.owner) return;
      exception_sink_.owner->unregister_exception_sink(exception_sink_, from_destructor);
    }

    /// \brief Mark debug proxy lifetime state as invalid.
    void invalidate_proxy_state() noexcept
    {
#if UNI20_DEBUG_ASYNC_TASKS
      if (proxy_state_) proxy_state_->alive.store(false, std::memory_order_release);
#endif
    }

    /// \brief Reset debug proxy lifetime state for this writer.
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
/// \brief Register coroutine debug dependencies for a read-buffer argument.
/// \tparam T Stored value type.
/// \param promise Promise collecting dependency metadata.
/// \param x Read buffer argument.
template <typename T> void ProcessCoroutineArgument(BasicAsyncTaskPromise* promise, ReadBuffer<T> const& x)
{
#if UNI20_DEBUG_DAG
  promise->ReadDependencies.push_back(x.node());
#endif
}

// For a WriteBuffer, we add the node to the WriteDependencies
/// \brief Register coroutine debug dependencies and exception sink for a write-buffer argument.
/// \tparam T Stored value type.
/// \param promise Promise collecting dependency metadata.
/// \param x Write buffer argument.
template <typename T> void ProcessCoroutineArgument(BasicAsyncTaskPromise* promise, WriteBuffer<T> const& x)
{
#if UNI20_DEBUG_DAG
  promise->WriteDependencies.push_back(x.node());
#endif
  x.register_exception_sink(*promise, false);
}

template <typename T> class Defer;

/// \brief Concept for buffers that can register explicit exception sinks.
template <typename B>
concept exception_sink_buffer = requires(B& buffer, BasicAsyncTaskPromise& promise)
{
  buffer.register_exception_sink(promise, true);
};

/// \brief Awaiter that registers explicit exception sinks and then resumes immediately.
/// \tparam Buffers Buffer types that implement `register_exception_sink`.
template <exception_sink_buffer... Buffers> class PropagateExceptionsAwaiter {
  public:
    /// \brief Construct from references to sink buffers.
    /// \param buffers Buffers that should receive unhandled exceptions.
    explicit PropagateExceptionsAwaiter(Buffers&... buffers) : buffers_(std::addressof(buffers)...) {}

    /// \brief Always ready; no suspension required.
    /// \return Always `true`.
    bool await_ready() const noexcept { return true; }

    /// \brief Pass through the currently-running task unchanged.
    /// \param task Current owning task.
    /// \return The same task, preserving ownership transfer semantics.
    AsyncTask await_suspend(AsyncTask&& task) noexcept { return std::move(task); }

    /// \brief No-op resume hook.
    void await_resume() noexcept {}

    /// \brief Register all configured buffers as explicit exception sinks.
    /// \param promise Promise that owns the sink list.
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

    /// \brief Assign by constructing/replacing the underlying value.
    /// \tparam U Source type constructible as `T`.
    /// \param u Source value.
    /// \return Reference to `*this`.
    template <typename U>
    requires(!std::same_as<std::remove_cvref_t<U>, WriteAccessProxy>) &&
        std::constructible_from<T, U&&> WriteAccessProxy& operator=(U&& u)
    {
      this->emplace(std::forward<U>(u));
      return *this;
    }

    /// \brief Construct or replace the underlying value in place.
    /// \tparam Args Constructor argument types.
    /// \param args Constructor arguments.
    /// \return Reference to the underlying value.
    template <typename... Args>
    requires std::constructible_from<T, Args...> T& emplace(Args&&... args)
    {
      return this->writer().emplace(std::forward<Args>(args)...);
    }

    /// \brief In-place `+=` update; emplaces when storage is unconstructed.
    /// \tparam U Operand type.
    /// \param x Right-hand operand.
    /// \return Reference to `*this`.
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

    /// \brief In-place `-=` update; emplaces negated value when unconstructed.
    /// \tparam U Operand type.
    /// \param x Right-hand operand.
    /// \return Reference to `*this`.
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

    /// \brief In-place `*=` update.
    /// \tparam U Operand type.
    /// \param x Right-hand operand.
    /// \return Reference to `*this`.
    template <typename U>
    requires requires(T& value, U&& x)
    {
      value *= std::forward<U>(x);
    } WriteAccessProxy& operator*=(U&& x)
    {
      this->get() *= std::forward<U>(x);
      return *this;
    }

    /// \brief In-place `/=` update.
    /// \tparam U Operand type.
    /// \param x Right-hand operand.
    /// \return Reference to `*this`.
    template <typename U>
    requires requires(T& value, U&& x)
    {
      value /= std::forward<U>(x);
    } WriteAccessProxy& operator/=(U&& x)
    {
      this->get() /= std::forward<U>(x);
      return *this;
    }

    /// \brief Move out the stored value.
    /// \return Moved value.
    T take() { return this->writer().storage().take(); }

    /// \brief Move out the value and release the writer epoch.
    /// \return Moved value.
    T take_release()
    {
      T value = this->take();
      this->release();
      return value;
    }

    /// \brief Release the writer epoch held by this proxy.
    void release() noexcept
    {
#if UNI20_DEBUG_ASYNC_TASKS
      if (auto state = proxy_state_.lock(); state)
      {
        state->alive.store(false, std::memory_order_release);
      }
#endif
      if (writer_)
      {
        writer_->release();
        writer_ = nullptr;
      }
    }

    /// \brief Access the mutable stored value.
    /// \return Reference to the stored value.
    T& get() const { return this->writer().data(); }

    operator T&() const { return this->get(); }

    T* operator->() const { return std::addressof(this->get()); }

  private:
    /// \brief Construct from a non-owning writer pointer.
    /// \param writer Writer pointer borrowed from a `WriteBuffer`.
#if UNI20_DEBUG_ASYNC_TASKS
    /// \param proxy_state Shared debug lifetime state for use-after-release checks.
#endif
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

    /// \brief Internal writer accessor with optional debug lifetime checks.
    /// \return Reference to the underlying writer.
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

    /// \brief Assign by constructing/replacing the underlying value.
    /// \tparam U Source type constructible as `T`.
    /// \param u Source value.
    /// \return Reference to `*this`.
    template <typename U>
    requires(!std::same_as<std::remove_cvref_t<U>, OwningWriteAccessProxy>) &&
        std::constructible_from<T, U&&> OwningWriteAccessProxy& operator=(U&& u)
    {
      this->emplace(std::forward<U>(u));
      return *this;
    }

    /// \brief Construct or replace the underlying value in place.
    /// \tparam Args Constructor argument types.
    /// \param args Constructor arguments.
    /// \return Reference to the underlying value.
    template <typename... Args>
    requires std::constructible_from<T, Args...> T& emplace(Args&&... args)
    {
      return writer_.emplace(std::forward<Args>(args)...);
    }

    /// \brief In-place `+=` update; emplaces when storage is unconstructed.
    /// \tparam U Operand type.
    /// \param x Right-hand operand.
    /// \return Reference to `*this`.
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

    /// \brief In-place `-=` update; emplaces negated value when unconstructed.
    /// \tparam U Operand type.
    /// \param x Right-hand operand.
    /// \return Reference to `*this`.
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

    /// \brief In-place `*=` update.
    /// \tparam U Operand type.
    /// \param x Right-hand operand.
    /// \return Reference to `*this`.
    template <typename U>
    requires requires(T& value, U&& x)
    {
      value *= std::forward<U>(x);
    } OwningWriteAccessProxy& operator*=(U&& x)
    {
      this->get() *= std::forward<U>(x);
      return *this;
    }

    /// \brief In-place `/=` update.
    /// \tparam U Operand type.
    /// \param x Right-hand operand.
    /// \return Reference to `*this`.
    template <typename U>
    requires requires(T& value, U&& x)
    {
      value /= std::forward<U>(x);
    } OwningWriteAccessProxy& operator/=(U&& x)
    {
      this->get() /= std::forward<U>(x);
      return *this;
    }

    /// \brief Access mutable stored value.
    /// \return Reference to the stored value.
    T& get() { return writer_.data(); }
    /// \brief Access const stored value.
    /// \return Const reference to the stored value.
    T const& get() const { return writer_.data(); }

    /// \brief Move out the stored value.
    /// \return Moved value.
    T take() { return writer_.storage().take(); }

    /// \brief Move out the value and release the writer epoch.
    /// \return Moved value.
    T take_release()
    {
      T value = this->take();
      this->release();
      return value;
    }

    /// \brief Release the writer epoch held by this proxy.
    void release() noexcept { writer_.release(); }

    operator T&() { return this->get(); }
    operator T const&() const { return this->get(); }

    T* operator->() { return std::addressof(this->get()); }
    T const* operator->() const { return std::addressof(this->get()); }

  private:
    /// \brief Construct from an owning writer.
    /// \param writer Writer handle transferred into this proxy.
    explicit OwningWriteAccessProxy(EpochContextWriter<T>&& writer) : writer_(std::move(writer)) {}

    friend class WriteBuffer<T>;
    friend class OwningWriteAwaiter<T>;

    EpochContextWriter<T> writer_;
};

/// \brief Awaiter that transfers a writer handle into an owning write proxy.
template <typename T> class OwningWriteAwaiter {
  public:
    /// \brief Construct from an owning writer.
    /// \param writer Writer handle transferred into this awaiter.
    explicit OwningWriteAwaiter(EpochContextWriter<T>&& writer) : writer_(std::move(writer)) {}

    /// \brief Reports whether the writer epoch is immediately writable.
    /// \return `true` when no suspension is needed.
    bool await_ready() const noexcept { return writer_.ready(); }

    /// \brief Suspend until the writer epoch becomes writable.
    /// \param task Owning task to suspend and enqueue.
    void await_suspend(AsyncTask&& task) noexcept
    {
      TRACE_MODULE(ASYNC, "OwningWriteAwaiter::await_suspend()", this, task.h_);
      writer_.suspend(std::move(task), false);
    }

    /// \brief Resume and transfer writer ownership to an owning write proxy.
    /// \return Owning write proxy.
    OwningWriteAccessProxy<T> await_resume()
    {
      writer_.resume();
      return OwningWriteAccessProxy<T>(std::move(writer_));
    }

    /// \brief Returns the epoch context used for exception propagation.
    /// \return Shared pointer to the epoch context.
    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return writer_.epoch_context_shared(); }

  private:
    EpochContextWriter<T> writer_;
};

/// \brief Alias for the non-owning write access proxy type.
template <typename T> using WriteAwaitProxy = WriteAccessProxy<T>;
/// \brief Alias for the owning write access proxy type.
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

    /// \brief Schedule assignment through the consumed writer handle.
    /// \tparam U Source type.
    /// \param u Source expression to assign.
    template <typename U>
    requires(!std::same_as<std::remove_cvref_t<U>, WriteAssignProxy>) void operator=(U&& u)
    {
      async_assign(std::forward<U>(u), WriteBuffer<T>(std::move(writer_)));
    }

  private:
    /// \brief Construct from an owning writer.
    /// \param writer Writer handle transferred into this proxy.
    explicit WriteAssignProxy(EpochContextWriter<T>&& writer) : writer_(std::move(writer)) {}

    friend class WriteBuffer<T>;

    EpochContextWriter<T> writer_;
};

/// \brief Alias for assignment-only writer proxy.
template <typename T> using WriteProxy = WriteAssignProxy<T>;

/// \brief Build an assignment-only proxy from a write buffer.
/// \tparam T Stored value type.
/// \return Assignment-only writer proxy.
template <typename T> WriteAssignProxy<T> WriteBuffer<T>::write()
{
  return WriteAssignProxy<T>(std::move(writer_));
}

} // namespace uni20::async
