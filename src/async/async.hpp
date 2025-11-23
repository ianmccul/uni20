/// \file async.hpp
/// \brief The Async<T> container: coroutine‐safe asynchronous read/write.
/// \ingroup async_api

// NOTE: Immediately-invoked coroutine lambdas must not capture variables.
// Captures (by reference or value) are stored in the lambda frame, which is destroyed
// after the lambda returns. If the coroutine suspends, any captured state becomes dangling.
// Instead, pass all needed data via parameters, which are safely moved into the coroutine frame.

#pragma once

#include "async_node.hpp"
#include "buffers.hpp"
#include "common/demangle.hpp"
#include "config.hpp"
#include "epoch_queue.hpp"
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <initializer_list>

namespace uni20::async
{

class DebugScheduler;

/// \brief Tag type to construct an Async without an initial value pointer.
struct deferred_t
{};

/// \brief Tag constant for deferred Async construction.
inline constexpr deferred_t deferred{};

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
/// \note Buffers maintain shared ownership of the internal state, so `ReadBuffer<T>`, `MutableBuffer<T>`, and
///       `WriteBuffer<T>` may safely outlive the Async.
/// \note The value of T must be copyable or movable as appropriate for construction.
/// \tparam T Stored value type.
/// \ingroup async_api
template <typename T> class Async {
  public:
    using value_type = T;

    /// \brief Initializes async state without constructing the stored value.
    /// \ingroup async_api
    Async() : value_(detail::make_deferred_shared<T>()), queue_(std::make_shared<EpochQueue>())
    {
      queue_->initialize(true);
#if UNI20_DEBUG_DAG
      queue_->initialize_node(value_.get());
#endif
    }

    /// \brief Construct from a value convertible to T.
    /// \tparam U Value type convertible to T.
    /// \param val Initial value forwarded into the Async storage.
    /// \ingroup async_api
    template <typename U>
      requires std::convertible_to<U, T>
    Async(U&& val) : value_(std::make_shared<T>(std::forward<U>(val))), queue_(std::make_shared<EpochQueue>())
    {
      queue_->initialize(true);
#if UNI20_DEBUG_DAG
      queue_->initialize_node(value_.get());
#endif
    }

    /// \brief Explicitly construct from a value that requires explicit conversion.
    /// \tparam U Source type that can explicitly construct T.
    /// \param u Value forwarded to construct the stored T instance.
    /// \ingroup async_api
    template <typename U>
      requires std::constructible_from<T, U> && (!std::convertible_to<U, T>)
    explicit Async(U&& u)
        : value_(std::make_shared<T>(static_cast<T>(std::forward<U>(u)))), queue_(std::make_shared<EpochQueue>())
    {
      queue_->initialize(true);
#if UNI20_DEBUG_DAG
      queue_->initialize_node(value_.get());
#endif
    }

    /// \brief Construct the stored value in place using forwarded arguments.
    /// \tparam Args Argument types forwarded to `T`'s constructor.
    /// \param args Arguments used to initialize the contained value.
    /// \ingroup async_api
    template <typename... Args>
      requires std::constructible_from<T, Args...>
    Async(std::in_place_t, Args&&... args)
        : value_(std::make_shared<T>(std::forward<Args>(args)...)), queue_(std::make_shared<EpochQueue>())
    {
      queue_->initialize(true);
#if UNI20_DEBUG_DAG
      queue_->initialize_node(value_.get());
#endif
    }

    /// \brief Construct the stored value in place from an initializer list.
    /// \tparam U Element type accepted by `T`'s initializer list constructor.
    /// \tparam Args Additional argument types forwarded to `T`'s constructor.
    /// \param init Initializer list forwarded to `T`.
    /// \param args Additional arguments forwarded to `T`.
    /// \ingroup async_api
    template <typename U, typename... Args>
      requires std::constructible_from<T, std::initializer_list<U>&, Args...>
    Async(std::in_place_t, std::initializer_list<U> init, Args&&... args)
        : value_(std::make_shared<T>(init, std::forward<Args>(args)...)), queue_(std::make_shared<EpochQueue>())
    {
      queue_->initialize(true);
#if UNI20_DEBUG_DAG
      queue_->initialize_node(value_.get());
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
    /// The stored pointer is left null until a later call to \ref emplace or \ref reset_value.
    /// When supplied, \p queue is also shared so that deferred views participate in the same
    /// sequencing as the originating Async value.
    ///
    /// \tparam Control Type of the shared pointer used for aliasing the control block.
    /// \param tag `async::deferred` tag to select deferred construction.
    /// \param control Shared pointer whose control block should be reused for this Async value.
    /// \param queue Queue to reuse for sequencing; defaults to a fresh queue when omitted.
    /// \warning The pointer installed via \ref emplace or \ref reset_value must remain valid for
    ///          the duration of this Async's lifetime; the control block is shared, but the pointed
    ///          value is not owned.
    /// \ingroup async_api
    template <typename Control>
    Async(deferred_t tag, std::shared_ptr<Control> control, std::shared_ptr<EpochQueue> queue)
        : value_(std::shared_ptr<T>(std::move(control), static_cast<T*>(nullptr))), queue_(std::move(queue))
    {
      (void)tag;
      DEBUG_CHECK(value_);
      DEBUG_CHECK(queue_);
      queue_->initialize(false);
#if UNI20_DEBUG_DAG
      queue_->initialize_node(value_.get());
#endif
    }

    /// \brief Construct an Async that defers pointer initialization while sharing ownership.
    /// \tparam Control Type of the shared pointer used for aliasing the control block.
    /// \param tag `async::deferred` tag to select deferred construction.
    /// \param control Shared pointer whose control block should be reused for this Async value.
    /// \warning The pointer installed via \ref emplace or \ref reset_value must remain valid for
    ///          the duration of this Async's lifetime; the control block is shared, but the pointed
    ///          value is not owned.
    /// \ingroup async_api
    template <typename Control>
    Async(deferred_t tag, std::shared_ptr<Control> control)
        : Async(tag, std::move(control), std::make_shared<EpochQueue>())
    {}

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
    Async(Async&&) noexcept = default;
    /// \brief Move-assign from another Async<T> handle.
    /// \param other Source Async.
    /// \return Reference to *this after ownership transfer.
    /// \ingroup async_api
    Async& operator=(Async&&) noexcept = default;

    ~Async() = default;

    /// \brief Begin an asynchronous read of the value.
    /// \return A ReadBuffer<T> which may be co_awaited.
    /// \ingroup async_api
    ReadBuffer<T> read() const
    {
      DEBUG_CHECK(queue_);
      ensure_default_value();
      return ReadBuffer<T>(queue_->create_read_context(value_, queue_));
    }

    /// \brief Begin an asynchronous mutation of the current value.
    /// \return A MutableBuffer<T> which may be co_awaited.
    /// \ingroup async_api
    MutableBuffer<T> mutate()
    {
      DEBUG_CHECK(queue_);
      ensure_default_value();
      return MutableBuffer<T>(queue_->create_write_context(value_, queue_));
    }

    /// \brief Begin writing a fresh value, treating the storage as uninitialized until completion.
    /// \return A WriteBuffer<T> which may be co_awaited.
    /// \ingroup async_api
    WriteBuffer<T> write()
    {
      DEBUG_CHECK(queue_);
      ensure_default_value();
      return WriteBuffer<T>(queue_->create_write_context(value_, queue_));
    }

    EmplaceBuffer<T> emplace() noexcept
    {
      DEBUG_CHECK(queue_);
      detail::alias_deferred_pointer(value_);
      DEBUG_CHECK(value_);
      return EmplaceBuffer<T>(queue_->create_write_context(value_, queue_), detail::get_deferred_state(value_));
    }

    // template <typename Sched> T& get_wait(Sched& sched)
    // {
    //   DEBUG_CHECK(queue_);
    //   while (queue_->has_pending_writers())
    //   {
    //     TRACE_MODULE(ASYNC, "Has pending writers");
    //     sched.run();
    //   }
    //   return *value_;
    // }

    T const& get_wait() const;

    T const& get_wait(IScheduler& sched) const;

    // TODO: we could have a version that returns a ref-counted proxy, which enables reference rather than copy

    /// \brief Block until the value is available, then move it out of the Async.
    /// \return The stored value, moved out of the Async container.
    /// \ingroup async_api
    T move_from_wait();

    // FiXME: this is a hack for debugging
    void set(T const& x)
    {
      DEBUG_CHECK(queue_);
      ensure_default_value();
      *value_ = x;
    }

    /// \brief Install the value pointer for a deferred Async.
    ///
    /// The control block remains shared with the originating `std::shared_ptr`, but ownership of
    /// the pointed-to object is *not* transferred. The caller must ensure that \p ptr remains
    /// valid for the lifetime of this Async instance.
    ///
    /// \param ptr Pointer to the value managed externally.
    /// \ingroup async_api
    void reset_value(T* ptr)
    {
      DEBUG_CHECK(queue_);
      auto control = std::shared_ptr<void>(value_);
      value_ = std::shared_ptr<T>(std::move(control), ptr);
#if UNI20_DEBUG_DAG
      queue_->initialize_node(value_.get());
#endif
    }

    /// \brief Convenience alias for \ref reset_value.
    /// \param ptr Pointer to the value managed externally.
    /// \ingroup async_api
    void emplace(T* ptr) { this->reset_value(ptr); }

    // FiXME: this is a hack for debugging
    T value() const
    {
      DEBUG_CHECK(queue_);
      ensure_default_value();
      return *value_;
    }

    /// \brief Access the stored value without synchronization.
    /// \return Direct reference to stored value (for diagnostics only).
    /// \ingroup async_api
    T const& unsafe_value_ref() const
    {
      DEBUG_CHECK(queue_);
      ensure_default_value();
      return *value_;
    }
    T& unsafe_value_ref()
    {
      DEBUG_CHECK(queue_);
      ensure_default_value();
      return *value_;
    }

    /// \brief Inspect the shared implementation block.
    /// \return Access to underlying implementation (shared with buffers).
    /// \ingroup async_api
    std::shared_ptr<EpochQueue> const& queue() const { return queue_; }
    std::shared_ptr<T> const& value_ptr() const { return value_; }

  private:
    T* ensure_default_value() const
    {
      if (auto* existing = value_.get()) return existing;

      if constexpr (std::is_default_constructible_v<T>)
      {
        if (auto* initialized = detail::construct_deferred(value_)) return initialized;
        value_ = std::make_shared<T>();
        return value_.get();
      }
      else
      {
        throw std::logic_error("Async value requires initialization before access");
      }
    }

    // Add a new epoch to the front of the queue; used by ReverseValue for reverse mode autodifferentiation
    /// \brief Prepend a reverse-mode epoch to the queue.
    /// \return Writer and reader handles for the new epoch.
    /// \ingroup internal
    EpochQueue::EpochPair<T> prepend_epoch()
    {
      DEBUG_TRACE_MODULE(ASYNC, "Prepending epoch!");
      DEBUG_CHECK(queue_);
      ensure_default_value();
      return queue_->prepend_epoch(value_, queue_);
    }

    friend class ReverseValue<T>;

    mutable std::shared_ptr<T> value_;
    std::shared_ptr<EpochQueue> queue_;
};

/// \brief Convenience helper that forwards to Async<T>::read().
/// \tparam T Stored value type.
/// \param a Async container providing the read buffer.
/// \return Read buffer obtained from the Async instance.
/// \ingroup async_api
template <typename T> ReadBuffer<T> read(Async<T> const& a) { return a.read(); }
/// \brief Convenience helper that forwards to Async<T>::mutate().
/// \tparam T Stored value type.
/// \param a Async container providing the mutable buffer.
/// \return Mutable buffer obtained from the Async instance.
/// \ingroup async_api
template <typename T> MutableBuffer<T> mutate(Async<T>& a) { return a.mutate(); }
/// \brief Convenience helper that forwards to Async<T>::write().
/// \tparam T Stored value type.
/// \param a Async container providing the write buffer.
/// \return Write buffer obtained from the Async instance.
/// \ingroup async_api
template <typename T> WriteBuffer<T> write(Async<T>& a) { return a.write(); }

template <typename T> EmplaceBuffer<T> emplace_buffer(Async<T>& a) { return a.emplace(); }

} // namespace uni20::async

#include "async_impl.hpp"
