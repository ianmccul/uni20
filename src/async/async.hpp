/// \file async.hpp
/// \brief The Async<T> container: coroutine‐safe asynchronous read/write.
/// \ingroup async_api

// NOTE: Immediately-invoked coroutine lambdas must not capture variables.
// Captures (by reference or value) are stored in the lambda frame, which is destroyed
// after the lambda returns. If the coroutine suspends, any captured state becomes dangling.
// Instead, pass all needed data via parameters, which are safely moved into the coroutine frame.

#pragma once

#include "async_node.hpp"
#include "async_errors.hpp"
#include "buffers.hpp"
#include "common/demangle.hpp"
#include "config.hpp"
#include "epoch_queue.hpp"
#include <atomic>
#include <concepts>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace uni20::async
{

class DebugScheduler;

/// \brief Tag type to construct an Async without an initial value pointer.
struct deferred_t
{};

/// \brief Tag constant for deferred Async construction.
inline constexpr deferred_t deferred{};

/// \brief Tag type to construct an Async without starting the queue object.
struct async_do_not_start_t
{};

/// \brief Tag constant to construct an Async without starting the queue object.
inline constexpr async_do_not_start_t async_do_not_start{};

template <typename T> class ReverseValue; // forward declaration so we can add it as a friend of Async<T>

/// \brief Async<T> is a move-only container for asynchronously accessed data.
///
/// `Async<T>` stores a value of type `T` and mediates access through
/// epoch-based coordination. The value and access queue are jointly
/// refcounted by internal shared state, allowing buffer handles
/// to outlive the owning Async container.
///
/// Copying is disabled: deep copy must be performed via explicit kernels.
///
/// \note Buffers maintain shared ownership of the internal state, so `ReadBuffer<T>` and
///       `WriteBuffer<T>` may safely outlive the Async.
/// \note The value of T must be copyable or movable as appropriate for construction.
/// \tparam T Stored value type.
/// \ingroup async_api
template <typename T> class Async {
  public:
    using value_type = T;

    /// \brief Initializes async state without constructing the stored value.
    Async() : storage_(make_unconstructed_shared_storage<T>()), queue_()
    {
      queue_.latest()->start();
#if UNI20_DEBUG_DAG
      queue_.initialize_node(storage_->get());
#endif
    }

    /// \brief Initializes async state without constructing the stored value or starting the queue.
    Async(async_do_not_start_t tag) : storage_(make_unconstructed_shared_storage<T>()), queue_() {}

    /// \brief Construct from a value convertible to T.
    /// \tparam U Value type convertible to T.
    /// \param val Initial value forwarded into the Async storage.
    /// \ingroup async_api
    template <typename U>
      requires std::convertible_to<U, T>
    Async(U&& val) : storage_(make_shared_storage<T>(std::forward<U>(val))), queue_()
    {
      queue_.latest()->start_reading();
      // queue_.initialize(true);
#if UNI20_DEBUG_DAG
      queue_.initialize_node(storage_->get());
#endif
    }

    /// \brief Explicitly construct from a value that requires explicit conversion.
    /// \tparam U Source type that can explicitly construct T.
    /// \param u Value forwarded to construct the stored T instance.
    /// \ingroup async_api
    template <typename U>
      requires std::constructible_from<T, U> && (!std::convertible_to<U, T>)
    explicit Async(U&& u) : storage_(make_shared_storage<T>(static_cast<T>(std::forward<U>(u)))), queue_()
    {
      queue_.latest()->start_reading();
      // queue_.initialize(true);
#if UNI20_DEBUG_DAG
      queue_.initialize_node(storage_->get());
#endif
    }

    /// \brief Construct the stored value in place using forwarded arguments.
    /// \tparam Args Argument types forwarded to `T`'s constructor.
    /// \param args Arguments used to initialize the contained value.
    /// \ingroup async_api
    template <typename... Args>
      requires std::constructible_from<T, Args...>
    Async(Args&&... args) : storage_(make_shared_storage<T>(std::forward<Args>(args)...)), queue_()
    {
      queue_.latest()->start_reading();
      // queue_.initialize(true);
#if UNI20_DEBUG_DAG
      queue_.initialize_node(storage_->get());
#endif
    }

    /// \brief Construct the stored value in place from an initializer list.
    /// \tparam U Element type accepted by `T`'s initializer list constructor.
    /// \tparam Args Additional argument types forwarded to `T`'s constructor.
    /// \param init Initializer list forwarded to `T`.
    /// \param args Additional arguments forwarded to `T`.
    /// \note This mirrors similar constuctors where std::in_place is used.
    /// \ingroup async_api
    template <typename U, typename... Args>
      requires std::constructible_from<T, std::initializer_list<U>
                                       &,
                                       Args...>
      Async(std::initializer_list<U> init, Args&&... args)
        : storage_(make_shared_storage<T>(init, std::forward<Args>(args)...)), queue_()
    {
      queue_.latest()->start_reading();
      // queue_.initialize(true);
#if UNI20_DEBUG_DAG
      queue_.initialize_node(storage_->get());
#endif
    }

    /// \brief Construct a new Async<T> by copying the value from another Async<T>.
    ///
    /// \note This constructor schedules a coroutine that reads the current or eventual value of `rhs`
    ///       and writes it into the initial epoch of the newly constructed `*this`.
    ///
    /// \warning This is not a structural copy — it does not replicate the state or dependencies of `rhs`.
    ///          Coroutine handles, epoch queues, and computation histories are not copied.
    ///
    /// \see `async_assign` for explicit value-level copy scheduling.
    /// \ingroup async_api
    Async(const Async& rhs) : Async() { async_assign(rhs, *this); }

    /// \brief Construct an Async that defers pointer initialization while sharing ownership.
    ///
    /// This constructor aliases the control block of \p control so that the lifetime of the
    /// referenced object is tied to the same reference count as the source `std::shared_ptr`.
    /// The stored pointer is installed from `control.get()` immediately so that deferred views
    /// participate in the same sequencing as the originating Async value.
    ///
    /// \tparam Control Type of the shared pointer used for aliasing the control block.
    /// \param tag `async::deferred` tag to select deferred construction.
    /// \param control Shared pointer whose control block should be reused for this Async value.
    /// \param queue Queue to reuse for sequencing; defaults to a fresh queue when omitted.
    /// \throws std::invalid_argument if \p control is null.
    /// \warning The caller must ensure that `control.get()` remains valid for the Async lifetime.
    /// \ingroup async_api
    //     template <typename Control>
    //       requires std::convertible_to<Control*, T*>
    //     Async(deferred_t tag, std::shared_ptr<Control> control, std::shared_ptr<EpochQueue> queue)
    //         : storage_(std::make_shared<detail::StorageBuffer<T>>()), queue_(std::move(queue))
    //     {
    //       (void)tag;
    //       DEBUG_CHECK(storage_);
    //       if (!control) throw std::invalid_argument("Async deferred control block cannot be null");
    //       auto* ptr = control.get();
    //       storage_->reset_external_pointer(ptr, control);
    //       queue_.initialize(initial_value_initialized());
    // #if UNI20_DEBUG_DAG
    //       queue_.initialize_node(storage_->get());
    // #endif
    //     }

    /// \brief Construct an Async that defers pointer initialization while sharing ownership.
    /// \tparam Control Type of the shared pointer used for aliasing the control block.
    /// \param tag `async::deferred` tag to select deferred construction.
    /// \param control Shared pointer whose control block should be reused for this Async value.
    /// \ingroup async_api
    template <typename Control>
      requires std::convertible_to<Control*, T*>
    Async(deferred_t tag, std::shared_ptr<Control> control) : storage_(make_unconstructed_shared_storage<T>()), queue_()
    {
      (void)tag;
      if (!control) throw std::invalid_argument("Async deferred control block cannot be null");
      storage_.emplace(*control);
    }

    /// \brief Construct a deferred Async that aliases another Async's storage while keeping a separate queue.
    ///
    /// The constructed Async retains the parent's storage lifetime via a shared
    /// control block and installs the parent's pointer immediately so that the
    /// view participates in the same storage without needing additional setup.
    /// A fresh queue is created for the deferred view; if queue sharing is required,
    /// use the overload that accepts an explicit queue pointer.
    ///
    /// \tparam U Value type of the parent Async whose storage is retained.
    /// \param tag `async::deferred` tag to select deferred construction.
    /// \param parent Async whose storage and queue lifetimes should be preserved.
    template <typename U>
      requires std::convertible_to<U*, T*>
    Async(deferred_t tag, Async<U>& parent) : storage_(parent.storage()), queue_()
    {
      (void)tag;
      (void)parent;
    }

    /// \brief Copy-assign from another Async<T>, overwriting this instance's value timeline.
    ///
    /// \note This operator first resets the internal epoch queue of `*this` by move-assigning a fresh `Async<T>`.
    ///       It then schedules a coroutine that awaits `rhs` and writes its result to `*this`.
    ///
    /// \warning This operation does not preserve prior epochs or dependencies of `*this`.
    ///          If you wish to serialize with prior writes, use `async_assign(rhs, *this)` directly.
    ///
    /// \code
    ///   Async<T> x, y;
    ///   x = y;              // copies value from y into x, resets x's causal history
    ///
    ///   x = Async<T>{};     // explicitly reset x
    ///   async_assign(y, x); // equivalent to copy-assignment
    /// \endcode
    ///
    /// \see Async::operator=(Async&&) for structural replacement
    /// \see async_assign for explicit value-copy semantics
    /// \param rhs Source Async whose value timeline is copied.
    /// \return Reference to *this after scheduling the copy.
    /// \ingroup async_api
    Async& operator=(const Async& rhs)
    {
      if (this != &rhs)
      {
        *this = Async<T>{}; // reset the epoch queue
        async_assign(rhs, *this);
      }
      return *this;
    }

    /// \note `Async<T>` supports standard move construction and assignment. These operations
    ///       transfer the handle (logical reference to the async value), not the value itself.
    ///       To schedule a value transfer from one async object to another, use `async_move(...)` explicitly.
    ///       The results of both operations are rather similar.
    /// \brief Move-construct from another Async<T> handle.
    /// \param other Source Async.
    /// \ingroup async_api
    Async(Async&& other) noexcept = default;
    /// \brief Move-assign from another Async<T> handle.
    /// \param other Source Async.
    /// \return Reference to *this after ownership transfer.
    /// \ingroup async_api
    Async& operator=(Async&& other) noexcept
    {
      if (this != &other)
      {
        storage_ = std::move(other.storage_);
        queue_ = std::move(other.queue_);
      }
      return *this;
    }

    ~Async() = default;

    /// \brief Begin an asynchronous read of the value.
    /// \return A ReadBuffer<T> which may be co_awaited.
    /// \ingroup async_api
    ReadBuffer<T> read() const
    {
      (void)try_get_value();
      return ReadBuffer<T>(queue_.create_read_context(storage_));
    }

    /// \brief Begin an asynchronous mutation of the current value.
    /// \return A WriteBuffer<T> which may be co_awaited.
    /// \ingroup async_api
    WriteBuffer<T> mutate()
    {
      require_value();
      return WriteBuffer<T>(queue_.create_write_context(storage_));
    }

    /// \brief Begin writing a fresh value, treating the storage as uninitialized until completion.
    /// \return A WriteBuffer<T> which may be co_awaited.
    /// \ingroup async_api
    WriteBuffer<T> write()
    {
      return WriteBuffer<T>(queue_.create_write_context(storage_));
    }

    /// \brief Begin constructing the value in-place using placement new semantics.
    /// \return A WriteBuffer<T>; call `co_await buffer.emplace(...)` to construct in place.
    /// \ingroup async_api
    WriteBuffer<T> emplace() noexcept
    {
      DEBUG_CHECK(storage_);
      return WriteBuffer<T>(queue_.create_write_context(storage_));
    }

    // template <typename Sched> T& get_wait(Sched& sched)
    // {
    //   while (queue_.has_pending_writers())
    //   {
    //     TRACE_MODULE(ASYNC, "Has pending writers");
    //     sched.run();
    //   }
    //   return *value_;
    // }

    /// \brief Block the current thread until the value becomes available.
    /// \return Reference to the stored value once the pending writers have completed.
    /// \ingroup async_api
    T const& get_wait() const;

    /// \brief Block using an explicit scheduler until the value is ready.
    /// The scheduler will be driven until all pending writers complete, then the
    /// value reference is returned. This overload is useful for deterministic
    /// execution in tests where the scheduling context must be controlled.
    /// \param sched Scheduler instance used to make progress while waiting.
    /// \return Reference to the stored value once writes have finished.
    /// \ingroup async_api
    T const& get_wait(IScheduler& sched) const;

    // TODO: we could have a version that returns a ref-counted proxy, which enables reference rather than copy

    /// \brief Block until the value is available, then move it out of the Async.
    /// \return The stored value, moved out of the Async container.
    /// \ingroup async_api
    T move_from_wait();

    /// \brief Overwrite the stored value directly.
    /// Intended for debugging and test scaffolding where a synchronous update
    /// is acceptable. No synchronization with pending writers or readers is
    /// performed.
    /// \param x New value to store.
    /// \ingroup internal
    void unsafe_set(T const& x)
    {
      auto* ptr = require_value();
      *ptr = x;
    }

    /// \brief Return a copy of the stored value.
    /// This helper is primarily for diagnostics; it will throw if the value
    /// has not been initialized.
    /// \return Copy of the contained value.
    /// \ingroup internal
    T unsafe_value() const { return *require_value(); }

    /// \brief Access the stored value without synchronization.
    /// \return Direct reference to stored value (for diagnostics only).
    /// \ingroup async_api
    T const& unsafe_value_ref() const { return *require_value(); }

    /// \brief Access the stored value without synchronization.
    /// \return Mutable reference to the stored value for diagnostic use.
    /// \ingroup async_api
    T& unsafe_value_ref() { return *require_value(); }

    /// \brief Inspect the shared epoch queue.
    /// \details
    /// Buffer handles can outlive
    /// the originating Async object, so they must retain the same queue instance to keep
    /// epoch transitions and task lifetime semantics valid.
    /// \return Shared access to the epoch queue implementation.
    /// \ingroup async_api
    EpochQueue const& queue() const { return queue_; }
    shared_storage<T> const& storage() const { return storage_; }

    /// \brief Access the stored value pointer with shared ownership semantics.
    ///
    /// The returned pointer aliases the StorageBuffer control block so that the
    /// lifetime of the referenced value is tied to the same shared ownership as
    /// the Async container itself.
    std::shared_ptr<T> value_ptr() const
    {
      if (!storage_.valid()) return {};
      return std::shared_ptr<T>(storage_.get(), [storage = storage_](T*) mutable { storage.reset(); });
    }

  private:
    T* try_get_value() const
    {
      if (!storage_.valid()) throw async_storage_missing{};
      return storage_.get();
    }

    T* require_value() const
    {
      if (auto* ptr = try_get_value()) return ptr;
      throw async_value_uninitialized{};
    }

    friend class ReverseValue<T>;

    mutable shared_storage<T> storage_;
    /// \brief Shared queue state retained by Async and all derived buffers.
    /// \details Kept as shared ownership (rather than value) so in-flight ReadBuffer/
    ///          WriteBuffer objects remain valid even after the originating Async is moved
    ///          or destroyed.
    EpochQueue queue_;
};

/// \brief Convenience helper that forwards to Async<T>::read().
/// \tparam T Stored value type.
/// \param a Async container providing the read buffer.
/// \return Read buffer obtained from the Async instance.
/// \ingroup async_api
template <typename T> ReadBuffer<T> read(Async<T> const& a) { return a.read(); }
/// \brief Convenience helper that forwards to Async<T>::mutate().
/// \tparam T Stored value type.
/// \param a Async container providing the write buffer.
/// \return Write buffer obtained from the Async instance.
/// \ingroup async_api
template <typename T> WriteBuffer<T> mutate(Async<T>& a) { return a.mutate(); }
/// \brief Convenience helper that forwards to Async<T>::write().
/// \tparam T Stored value type.
/// \param a Async container providing the write buffer.
/// \return Write buffer obtained from the Async instance.
/// \ingroup async_api
template <typename T> WriteBuffer<T> write(Async<T>& a) { return a.write(); }

template <typename T> WriteBuffer<T> emplace_buffer(Async<T>& a) { return a.emplace(); }

} // namespace uni20::async

#include "async_impl.hpp"
