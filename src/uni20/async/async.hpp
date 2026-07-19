/// \file async.hpp
/// \brief The Async<T> container: coroutine‐safe asynchronous read/write.

// NOTE: Immediately-invoked coroutine lambdas must not capture variables.
// Captures (by reference or value) are stored in the lambda frame, which is destroyed
// after the lambda returns. If the coroutine suspends, any captured state becomes dangling.
// Instead, pass all needed data via parameters, which are safely moved into the coroutine frame.

#pragma once

#include "async_errors.hpp"
#include "async_node.hpp"
#include "async_traits.hpp"
#include "buffers.hpp"
#include "epoch_queue.hpp"
#include <concepts>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <uni20/common/demangle.hpp>
#include <uni20/config.hpp>
#include <utility>

namespace uni20::async
{

class DebugScheduler;

/// \brief Tag type to construct an Async without starting the queue object.
struct async_do_not_start_t
{};

/// \brief Tag constant to construct an Async without starting the queue object.
inline constexpr async_do_not_start_t async_do_not_start{};

template <typename T> class ReverseValue; // forward declaration so we can add it as a friend of Async<T>

namespace detail
{

template <typename Source>
concept async_source = std::same_as<std::remove_cvref_t<Source>, Async<async_value_t<Source>>>;

template <typename Target, typename Source>
concept value_rebind_source = (async_source<Source> && std::constructible_from<Target, async_value_t<Source> const&>) ||
                              (!async_source<Source> && std::constructible_from<Target, Source&&>);

template <typename Target, typename Source>
concept async_assignment_source =
    (!std::same_as<std::remove_cvref_t<Source>, Async<Target>>) &&
    ((!is_async_alias_v<Target> && value_rebind_source<Target, Source>) ||
     (is_async_alias_v<Target> && async_alias_assignable_from<Target, async_value_t<Source> const&>));

} // namespace detail

template <typename T, typename U>
  requires(!is_async_alias_v<T> && detail::async_source<U> && std::constructible_from<T, async_value_t<U> const&>)
void async_initialize(Async<T>& destination, U&& source);

/// \brief Construct an async alias that retains a parent value and shares its epoch queue.
template <typename Alias, typename Parent, typename... Args>
[[nodiscard]] Async<Alias> make_async_alias(Async<Parent> const& parent, Args&&... args);

/// \brief Construct an async alias from a mutable parent descriptor.
/// \details The alias constructor receives the address reserved for the parent
///          value followed by \p args. The address may identify unconstructed
///          storage and must not be dereferenced before the shared epoch is readable.
template <typename Alias, typename Parent, typename... Args>
[[nodiscard]] Async<Alias> make_async_alias_from_parent(Async<Parent>& parent, Args&&... args);

/// \brief Construct a read-only async alias from a parent descriptor.
template <typename Alias, typename Parent, typename... Args>
[[nodiscard]] Async<Alias> make_async_alias_from_parent(Async<Parent> const& parent, Args&&... args);

/// \brief Async<T> is a container for coroutine-safe asynchronous read/write.
///
/// `Async<T>` stores a value of type `T` and mediates access through
/// epoch-based coordination. The value, queue, and individual epoch contexts
/// use shared ownership where required. Buffer handles retain their storage and
/// selected epoch context, so they may outlive the owning Async container.
///
/// \note Copy construction is a value-level operation by default. Types for
///       which `is_async_alias_v<T>` is true copy the storage handle, lifetime
///       owner, and dependency timeline instead.
///
/// \note Buffers maintain shared ownership of the internal state, so `ReadBuffer<T>` and
///       `WriteBuffer<T>` may safely outlive the Async.
/// \note `T` need not be default-constructible, copyable, or movable. Individual
///       operations participate only when their required construction or
///       assignment expressions are available.
/// \tparam T Stored value type.
template <typename T> class Async {
  public:
    using value_type = T;

    static_assert(std::is_object_v<T> && !std::is_array_v<T>, "Async<T> requires a non-array object type");
    static_assert(std::destructible<T>, "Async<T> requires a non-throwing destructor");

    /// \brief Initializes async state without constructing the stored value.
    Async()
      requires(!is_async_alias_v<T>)
        : storage_(make_unconstructed_shared_storage<T>()), queue_(std::make_shared<EpochQueue>())
    {
      queue_->latest()->start();
#if UNI20_DEBUG_DAG
      queue_->initialize_node(storage_);
#endif
    }

    /// \brief Initializes async state without constructing the stored value or starting the queue.
    /// \param tag Sentinel tag selecting non-starting construction.
    Async(async_do_not_start_t tag)
      requires(!is_async_alias_v<T>)
        : storage_(make_unconstructed_shared_storage<T>()), queue_(std::make_shared<EpochQueue>())
    {
      (void)tag;
#if UNI20_DEBUG_DAG
      queue_->initialize_node(storage_);
#endif
    }

    /// \brief Construct from a value convertible to T.
    /// \tparam U Value type convertible to T.
    /// \param val Initial value forwarded into the Async storage.
    template <typename U>
      requires(std::convertible_to<U, T> && !is_async_alias_v<T>)
    Async(U&& val) : storage_(make_shared_storage<T>(std::forward<U>(val))), queue_(std::make_shared<EpochQueue>())
    {
      queue_->latest()->start_reading();
#if UNI20_DEBUG_DAG
      queue_->initialize_node(storage_);
#endif
    }

    /// \brief Explicitly construct from a value that requires explicit conversion.
    /// \tparam U Source type that can explicitly construct T.
    /// \param u Value forwarded to construct the stored T instance.
    template <typename U>
      requires(std::constructible_from<T, U> && (!std::convertible_to<U, T>) && !is_async_alias_v<T>)
    explicit Async(U&& u) : storage_(make_shared_storage<T>(std::forward<U>(u))), queue_(std::make_shared<EpochQueue>())
    {
      queue_->latest()->start_reading();
#if UNI20_DEBUG_DAG
      queue_->initialize_node(storage_);
#endif
    }

    /// \brief Construct the stored value in place using forwarded arguments.
    /// \tparam Args Argument types forwarded to `T`'s constructor.
    /// \param args Arguments used to initialize the contained value.
    template <typename... Args>
      requires(std::constructible_from<T, Args...> && !is_async_alias_v<T>)
    Async(Args&&... args)
        : storage_(make_shared_storage<T>(std::forward<Args>(args)...)), queue_(std::make_shared<EpochQueue>())
    {
      queue_->latest()->start_reading();
#if UNI20_DEBUG_DAG
      queue_->initialize_node(storage_);
#endif
    }

    /// \brief Construct the stored value in place from an initializer list.
    /// \tparam U Element type accepted by `T`'s initializer list constructor.
    /// \tparam Args Additional argument types forwarded to `T`'s constructor.
    /// \param init Initializer list forwarded to `T`.
    /// \param args Additional arguments forwarded to `T`.
    /// \note This mirrors similar constructors where std::in_place is used.
    template <typename U, typename... Args>
      requires(std::constructible_from<T, std::initializer_list<U>&, Args...> && !is_async_alias_v<T>)
    Async(std::initializer_list<U> init, Args&&... args)
        : storage_(make_shared_storage<T>(init, std::forward<Args>(args)...)), queue_(std::make_shared<EpochQueue>())
    {
      queue_->latest()->start_reading();
#if UNI20_DEBUG_DAG
      queue_->initialize_node(storage_);
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
    Async(Async const& rhs)
      requires(!is_async_alias_v<T> && std::constructible_from<T, T const&>)
        : Async()
    {
      async_initialize(*this, rhs);
    }

    /// \brief Forbid copying an independent async value whose payload cannot be copy-constructed.
    Async(Async const&)
      requires(!is_async_alias_v<T> && !std::constructible_from<T, T const&>)
    = delete;

    /// \brief Structurally copy an async alias handle.
    /// \details Alias copies retain the same descriptor storage, lifetime owner,
    ///          and epoch queue.
    Async(Async const& rhs)
      requires(is_async_alias_v<T>)
        : storage_(rhs.storage_), queue_(rhs.queue_)
    {}

    /// \brief Assign an optional debug label for task-registry graph output.
    /// \param label Human-readable label used only by debug diagnostics.
    /// \return Reference to this async value for call chaining.
    Async& debug_name(std::string const& label)
    {
#if UNI20_DEBUG_DAG
      TaskRegistry::name_async_value(queue_->debug_node(), label);
#else
      static_cast<void>(label);
#endif
      return *this;
    }

    /// \brief Assign an optional debug label for task-registry graph output.
    /// \param label Human-readable label used only by debug diagnostics.
    /// \return Constant reference to this async value for call chaining.
    Async const& debug_name(std::string const& label) const
    {
#if UNI20_DEBUG_DAG
      TaskRegistry::name_async_value(queue_->debug_node(), label);
#else
      static_cast<void>(label);
#endif
      return *this;
    }

    /// \brief Assign an immediate or heterogeneous async value.
    /// \details Independent values detach onto fresh storage and a fresh epoch
    ///          queue. Async aliases retain their descriptor, owner, and queue,
    ///          and participate only when ADL finds a matching `assign_through`
    ///          operation. Read-only aliases therefore have no assignment.
    /// \tparam U Immediate value or heterogeneous async source type.
    /// \param rhs Source expression.
    /// \return Reference to this async handle.
    template <typename U>
      requires detail::async_assignment_source<T, U>
    Async& operator=(U&& rhs);

    /// \brief Copy-assign from another Async<T>, overwriting this instance's value timeline.
    ///
    /// \note This operator first resets the internal epoch queue of `*this` by move-assigning a fresh `Async<T>`.
    ///       It then schedules a coroutine that awaits `rhs` and writes its result to `*this`.
    ///
    /// \warning This operation does not preserve prior epochs or dependencies of `*this`.
    ///          If you wish to serialize with prior writes, use `async_assign(*this, rhs)` directly.
    ///
    /// \code
    ///   Async<T> x, y;
    ///   x = y;              // copies value from y into x, resets x's causal history
    ///
    ///   x = Async<T>{};     // explicitly reset x
    ///   async_assign(x, y); // equivalent to copy-assignment
    /// \endcode
    ///
    /// \see Async::operator=(Async&&) for independent-value move rebinding
    /// \see async_assign for explicit value-copy semantics
    /// \param rhs Source Async whose value timeline is copied.
    /// \return Reference to *this after scheduling the copy.
    Async& operator=(Async const& rhs)
      requires(!is_async_alias_v<T> && std::constructible_from<T, T const&>)
    {
      if (this != &rhs)
      {
        Async<T> replacement;
        async_initialize(replacement, rhs);
        *this = std::move(replacement);
      }
      return *this;
    }

    /// \brief Forbid copy assignment when an independent payload cannot be copy-constructed.
    Async& operator=(Async const&)
      requires(!is_async_alias_v<T> && !std::constructible_from<T, T const&>)
    = delete;

    /// \brief Assign through a mutable async alias from another exact-type alias.
    /// \details The destination descriptor, owner, and queue remain unchanged.
    /// \param rhs Alias whose referenced value is the assignment source.
    /// \return Reference to this alias.
    Async& operator=(Async const& rhs)
      requires(is_async_alias_v<T> && detail::async_alias_assignable_from<T, T const&>)
    {
      if (this != &rhs) async_assign(*this, rhs);
      return *this;
    }

    /// \brief Forbid assignment to a read-only async alias.
    Async& operator=(Async const&)
      requires(is_async_alias_v<T> && !detail::async_alias_assignable_from<T, T const&>)
    = delete;

    /// \brief Move-construct from another Async<T> handle.
    /// \note Move construction always transfers the handle. Independent-value
    ///       move assignment also transfers the handle; alias move assignment
    ///       remains write-through so an established alias is never retargeted.
    /// \param other Source Async.
    Async(Async&& other) noexcept = default;
    /// \brief Rebind an independent async value from another Async<T> handle.
    /// \param other Source Async.
    /// \return Reference to *this after ownership transfer.
    Async& operator=(Async&& other) noexcept
      requires(!is_async_alias_v<T>)
    {
      if (this != &other)
      {
        storage_ = std::move(other.storage_);
        queue_ = std::move(other.queue_);
      }
      return *this;
    }

    /// \brief Assign through a mutable async alias from an exact-type rvalue alias.
    /// \param other Alias whose referenced value is the assignment source.
    /// \return Reference to this alias.
    Async& operator=(Async&& other)
      requires(is_async_alias_v<T> && detail::async_alias_assignable_from<T, T const&>)
    {
      if (this != &other) async_assign(*this, other);
      return *this;
    }

    /// \brief Forbid assignment to a read-only async alias.
    Async& operator=(Async&&)
      requires(is_async_alias_v<T> && !detail::async_alias_assignable_from<T, T const&>)
    = delete;

    ~Async() = default;

    /// \brief Begin an asynchronous read of the value.
    /// \return A ReadBuffer<T> which may be co_awaited.
    ReadBuffer<T> read() const { return ReadBuffer<T>(queue_->create_read_context(storage_)); }

    /// \brief Begin exclusive asynchronous access to the payload.
    /// \details The returned proxy is capability-aware. Independent values
    ///          support replacement operations; aliases expose their descriptor
    ///          read-only and support only matching write-through assignments.
    /// \return Exclusive write buffer for the next epoch.
    WriteBuffer<T> write() { return WriteBuffer<T>(queue_->create_write_context(storage_)); }

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
    void wait() const;

    /// \brief Block using an explicit scheduler until the value is ready.
    /// The scheduler will be driven until all pending writers complete.
    /// \param sched Scheduler instance used to make progress while waiting.
    void wait(IAsyncScheduler& sched) const;

    /// \brief Block the current thread until the value becomes available.
    /// \return Reference to the stored value once the pending writers have completed.
    [[nodiscard]] T const& get_wait() const;

    /// \brief Block using an explicit scheduler until the value is ready.
    /// The scheduler will be driven until all pending writers complete, then the
    /// value reference is returned. This overload is useful for deterministic
    /// execution in tests where the scheduling context must be controlled.
    /// \param sched Scheduler instance used to make progress while waiting.
    /// \return Reference to the stored value once writes have finished.
    [[nodiscard]] T const& get_wait(IAsyncScheduler& sched) const;

    // TODO: we could have a version that returns a ref-counted proxy, which enables reference rather than copy

    /// \brief Block until the value is available, then move it out of the Async.
    /// \return The stored value, moved out of the Async container.
    T move_from_wait()
      requires(!is_async_alias_v<T> && std::constructible_from<T, T &&>);

    /// \brief Overwrite the stored value directly.
    /// Intended for debugging and test scaffolding where a synchronous update
    /// is acceptable. No synchronization with pending writers or readers is
    /// performed.
    /// \param x New value to store.
    void unsafe_set(T const& x)
      requires(!is_async_alias_v<T> && std::assignable_from<T&, T const&>)
    {
      auto* ptr = require_value();
      *ptr = x;
    }

    /// \brief Return a copy of the stored value.
    /// This helper is primarily for diagnostics; it will throw if the value
    /// has not been initialized.
    /// \return Copy of the contained value.
    T unsafe_value() const
      requires(!is_async_alias_v<T> && std::constructible_from<T, T const&>)
    {
      return *require_value();
    }

    /// \brief Access the stored value without synchronization.
    /// \return Direct reference to stored value (for diagnostics only).
    T const& unsafe_value_ref() const { return *require_value(); }

    /// \brief Access the stored value without synchronization.
    /// \return Mutable reference to the stored value for diagnostic use.
    T& unsafe_value_ref()
      requires(!is_async_alias_v<T>)
    {
      return *require_value();
    }

    /// \brief Inspect the shared epoch queue.
    /// \details Parent values and aliases share this exact mutable queue.
    ///          Individual buffers retain their selected epoch contexts.
    /// \return Access to the epoch queue implementation.
    EpochQueue const& queue() const { return *queue_; }
    /// \brief Access the shared storage control block.
    /// \return Shared storage object for the contained value.
    shared_storage<T> const& storage() const { return storage_; }

    /// \brief Return the address reserved for the contained value when constructing an alias descriptor.
    /// \warning The pointer does not grant synchronized access and must not be
    ///          dereferenced outside an active buffer. It may identify
    ///          reserved but unconstructed storage.
    /// \return Mutable address reserved for the contained value.
    [[nodiscard]] T* storage_address() noexcept
      requires(!is_async_alias_v<T>)
    {
      return storage_.storage_address();
    }

    /// \brief Return the const address reserved for the contained value when constructing an alias descriptor.
    /// \warning The pointer does not grant synchronized access and must not be
    ///          dereferenced outside an active buffer. It may identify
    ///          reserved but unconstructed storage.
    /// \return Const address reserved for the contained value.
    [[nodiscard]] T const* storage_address() const noexcept { return storage_.storage_address(); }

    /// \brief Access the mutable stored value pointer with shared ownership semantics.
    /// \details This overload is unavailable for `not_assignable` aliases.
    ///
    /// The returned pointer retains the `shared_storage` control block so that the
    /// lifetime of the referenced value is tied to the same shared ownership as
    /// the Async container itself.
    /// \return Mutable pointer retaining the `shared_storage` control block.
    std::shared_ptr<T> value_ptr()
      requires(!is_async_alias_v<T>)
    {
      if (!storage_.valid()) return {};
      return std::shared_ptr<T>(storage_.get(), [storage = storage_](T*) mutable { storage.reset(); });
    }

    /// \brief Access the const stored value pointer with shared ownership semantics.
    /// \return Const pointer retaining the `shared_storage` control block.
    std::shared_ptr<T const> value_ptr() const
    {
      if (!storage_.valid()) return {};
      return std::shared_ptr<T const>(storage_.get(), [storage = storage_](T const*) mutable { storage.reset(); });
    }

  private:
    /// \brief Construct from alias storage and an existing epoch queue.
    Async(shared_storage<T> storage, std::shared_ptr<EpochQueue> queue)
        : storage_(std::move(storage)), queue_(std::move(queue))
    {
      CHECK(storage_.has_lifetime_owner());
      CHECK(queue_);
    }

    /// \brief Returns the raw stored pointer when available.
    /// \return Pointer to the stored value, or `nullptr` if unconstructed.
    /// \throws async_storage_missing when storage metadata is unavailable.
    T* try_get_value() const
    {
      if (!storage_.valid()) throw async_storage_missing{};
      return storage_.get();
    }

    /// \brief Returns the stored pointer and enforces constructed-value presence.
    /// \return Pointer to the constructed value.
    /// \throws async_value_uninitialized when the value is not yet constructed.
    T* require_value() const
    {
      if (auto* ptr = try_get_value()) return ptr;
      throw async_value_uninitialized{};
    }

    friend class ReverseValue<T>;
    template <typename Alias, typename Parent, typename... Args>
    friend Async<Alias> make_async_alias(Async<Parent> const& parent, Args&&... args);
    template <typename Alias, typename Parent, typename... Args>
    friend Async<Alias> make_async_alias_from_parent(Async<Parent>& parent, Args&&... args);
    template <typename Alias, typename Parent, typename... Args>
    friend Async<Alias> make_async_alias_from_parent(Async<Parent> const& parent, Args&&... args);

    mutable shared_storage<T> storage_;
    /// \brief Mutable timeline shared by a parent `Async` and all of its aliases.
    /// \details Buffers retain individual epoch contexts; sharing the queue object
    ///          ensures future parent and alias buffers advance one timeline.
    std::shared_ptr<EpochQueue> queue_;
};

template <typename Alias, typename Parent, typename... Args>
[[nodiscard]] Async<Alias> make_async_alias(Async<Parent> const& parent, Args&&... args)
{
  static_assert(is_async_alias_v<Alias>, "Async aliases require is_async_alias_v<Alias>");
  return Async<Alias>(make_shared_storage_alias<Alias>(parent.storage_, std::forward<Args>(args)...), parent.queue_);
}

template <typename Alias, typename Parent, typename... Args>
[[nodiscard]] Async<Alias> make_async_alias_from_parent(Async<Parent>& parent, Args&&... args)
{
  static_assert(is_async_alias_v<Alias>, "Async aliases require is_async_alias_v<Alias>");
  return Async<Alias>(
      make_shared_storage_alias<Alias>(parent.storage_, parent.storage_.storage_address(), std::forward<Args>(args)...),
      parent.queue_);
}

template <typename Alias, typename Parent, typename... Args>
[[nodiscard]] Async<Alias> make_async_alias_from_parent(Async<Parent> const& parent, Args&&... args)
{
  static_assert(is_async_alias_v<Alias>, "Async aliases require is_async_alias_v<Alias>");
  return Async<Alias>(make_shared_storage_alias<Alias>(parent.storage_,
                                                       std::as_const(parent.storage_).storage_address(),
                                                       std::forward<Args>(args)...),
                      parent.queue_);
}

/// \brief Convenience helper that forwards to Async<T>::read().
/// \tparam T Stored value type.
/// \param a Async container providing the read buffer.
/// \return Read buffer obtained from the Async instance.
template <typename T> ReadBuffer<T> read(Async<T> const& a) { return a.read(); }
/// \brief Convenience helper that forwards to Async<T>::write().
/// \tparam T Stored value type.
/// \param a Async container providing the write buffer.
/// \return Write buffer obtained from the Async instance.
template <typename T> WriteBuffer<T> write(Async<T>& a) { return a.write(); }

} // namespace uni20::async

#include "async_impl.hpp"
