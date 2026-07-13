#pragma once
/// \file shared_storage.hpp
/// \brief Reference-counted optional in-place storage used by async buffers.
#include <atomic>
#include <concepts>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <uni20/common/trace.hpp>
#include <utility>

namespace uni20::async
{

class NodeInfo;

namespace detail
{

/// \brief Type-erased ownership state shared by all `shared_storage` control blocks.
/// \details An alias control block retains one reference to its lifetime owner.
///          Its local references are therefore a shard of the owner's strong
///          reference count rather than one owner reference per alias handle.
class shared_storage_control_block {
  public:
    explicit shared_storage_control_block(shared_storage_control_block* lifetime_owner = nullptr) noexcept
        : lifetime_owner_(lifetime_owner)
    {
      if (lifetime_owner_) lifetime_owner_->add_ref();
    }

    shared_storage_control_block(shared_storage_control_block const&) = delete;
    shared_storage_control_block& operator=(shared_storage_control_block const&) = delete;

    virtual ~shared_storage_control_block()
    {
      if (lifetime_owner_) lifetime_owner_->release_ref();
    }

    void add_ref() noexcept { strong_count_.fetch_add(1, std::memory_order_relaxed); }

    void release_ref() noexcept
    {
      if (strong_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this;
    }

    [[nodiscard]] std::size_t use_count() const noexcept { return strong_count_.load(std::memory_order_relaxed); }

    [[nodiscard]] bool constructed() const noexcept
    {
      return this->local_value_constructed() && (!lifetime_owner_ || lifetime_owner_->constructed());
    }

    [[nodiscard]] bool has_lifetime_owner() const noexcept { return lifetime_owner_ != nullptr; }

  private:
    [[nodiscard]] virtual bool local_value_constructed() const noexcept = 0;

    std::atomic<std::size_t> strong_count_{1};
    shared_storage_control_block* lifetime_owner_ = nullptr;
};

} // namespace detail

/// \brief A lightweight, thread-safe, reference-counted storage for a single object.
/// \details
/// Unlike `std::shared_ptr<T>`, `shared_storage<T>` can exist in an *unconstructed* state,
/// allowing delayed or conditional in-place construction via `.emplace()`.
///
/// This is useful for async or deferred initialization scenarios, where
/// the lifetime and ownership of an object must be shared across tasks,
/// but construction may not yet have occurred.
///
/// ### Key features:
/// - Shared ownership via atomic reference counting
/// - Deferred or repeated construction via `.emplace()`
/// - Thread-safe refcounting (like `std::shared_ptr`)
/// - Minimal overhead (single heap allocation)
///
/// ### Example:
/// ```cpp
/// shared_storage<MyType> s = make_shared_storage<MyType>();
///
/// if (!s.constructed())
///     s.emplace(42, "hello");
///
/// MyType& ref = s.get();
/// s.destroy();  // explicitly destroy the contained object
/// ```
/// Alias storage can own a lightweight value such as a tensor-view descriptor
/// while retaining one strong reference to another `shared_storage` control
/// block that owns the underlying data.
template <typename T> class shared_storage {
  private:
    friend class NodeInfo;

    /// \brief Control block holding storage, construction flag, and reference count.
    struct control_block final : detail::shared_storage_control_block
    {
        explicit control_block(detail::shared_storage_control_block* lifetime_owner = nullptr) noexcept
            : detail::shared_storage_control_block(lifetime_owner)
        {}

        alignas(T) unsigned char storage[sizeof(T)];
        std::atomic<bool> constructed_{false};

        /// \brief Returns the address reserved for the in-place value.
        /// \warning The returned pointer must not be dereferenced before construction.
        T* storage_address() noexcept { return reinterpret_cast<T*>(storage); }

        /// \brief Returns a typed pointer to a live in-place value.
        /// \return Pointer to the storage region as `T*`.
        T* ptr() noexcept { return std::launder(this->storage_address()); }

        /// \brief Construct a `T` object in-place.
        /// \tparam Args Constructor argument types.
        /// \param args Constructor arguments.
        template <typename... Args> void construct(Args&&... args)
        {
          DEBUG_CHECK(!constructed_.load(std::memory_order_relaxed));
          ::new (storage) T(std::forward<Args>(args)...);
          constructed_.store(true, std::memory_order_release);
        }

        /// \brief Destroy the in-place object when currently constructed.
        void destroy_object() noexcept
        {
          if (constructed_.exchange(false, std::memory_order_acq_rel)) ptr()->~T();
        }

        [[nodiscard]] bool value_constructed_locally() const noexcept
        {
          return constructed_.load(std::memory_order_acquire);
        }

        ~control_block() override { this->destroy_object(); }

      private:
        [[nodiscard]] bool local_value_constructed() const noexcept override
        {
          return this->value_constructed_locally();
        }
    };

    control_block* ctrl_ = nullptr;

    template <typename U, typename... Args> friend shared_storage<U> make_shared_storage(Args&&... args);
    template <typename U> friend shared_storage<U> make_unconstructed_shared_storage();
    template <typename U, typename Owner, typename... Args>
    friend shared_storage<U> make_shared_storage_alias(shared_storage<Owner> const& owner, Args&&... args);

    /// \brief Construct from a raw control block pointer.
    /// \param c Control block pointer.
    explicit shared_storage(control_block* c) noexcept : ctrl_(c) {}

    /// \brief Reports construction state for diagnostic observers.
    /// \param control Raw control-block pointer.
    /// \return `true` when the control block currently contains a constructed value.
    [[nodiscard]] static bool diagnostic_constructed_from_control(void const* control) noexcept
    {
      auto const* block = static_cast<control_block const*>(control);
      return block && block->constructed();
    }

    /// \brief Returns the current value address for diagnostic observers.
    /// \param control Raw control-block pointer.
    /// \return Constructed value address, or `nullptr` when unconstructed.
    [[nodiscard]] static void const* diagnostic_value_address_from_control(void const* control) noexcept
    {
      auto* block = const_cast<control_block*>(static_cast<control_block const*>(control));
      return block && block->constructed() ? static_cast<void const*>(block->ptr()) : nullptr;
    }

  public:
    using element_type = T;

    /// \brief Construct an empty handle with no control block.
    shared_storage() noexcept = default;

    /// \brief Copy constructor increments control block reference count.
    /// \param other Source handle.
    shared_storage(const shared_storage& other) noexcept : ctrl_(other.ctrl_)
    {
      if (ctrl_) ctrl_->add_ref();
    }

    /// \brief Move constructor transfers ownership of the control block pointer.
    /// \param other Source handle.
    shared_storage(shared_storage&& other) noexcept : ctrl_(std::exchange(other.ctrl_, nullptr)) {}

    /// \brief Copy assignment shares the control block.
    /// \param other Source handle.
    /// \return Reference to `*this`.
    shared_storage& operator=(const shared_storage& other) noexcept
    {
      if (this != &other)
      {
        this->reset();
        ctrl_ = other.ctrl_;
        if (ctrl_) ctrl_->add_ref();
      }
      return *this;
    }

    /// \brief Move assignment transfers the control block pointer.
    /// \param other Source handle.
    /// \return Reference to `*this`.
    shared_storage& operator=(shared_storage&& other) noexcept
    {
      if (this != &other)
      {
        this->reset();
        ctrl_ = std::exchange(other.ctrl_, nullptr);
      }
      return *this;
    }

    /// \brief Destructor releases one reference to the control block.
    ~shared_storage() { reset(); }

    /// \brief Release this handle's reference to the control block.
    void reset() noexcept
    {
      if (ctrl_)
      {
        ctrl_->release_ref();
        ctrl_ = nullptr;
      }
    }

    /// \brief Reports whether a value is currently constructed.
    /// \return `true` if the control block exists and holds a constructed value.
    [[nodiscard]] bool constructed() const noexcept { return ctrl_ && ctrl_->constructed(); }

    /// \brief Reports whether a control block is present.
    /// \return `true` when this handle owns or shares storage metadata.
    [[nodiscard]] bool valid() const noexcept { return ctrl_ != nullptr; }

    /// \brief Returns the storage control-block address for diagnostics.
    /// \return Raw control-block address, or `nullptr` when no control block is present.
    [[nodiscard]] void const* control_address() const noexcept { return static_cast<void const*>(ctrl_); }

    /// \brief Returns the current strong reference count.
    /// \return Number of `shared_storage` handles sharing this control block.
    [[nodiscard]] size_t use_count() const noexcept { return ctrl_ ? ctrl_->use_count() : 0; }

    /// \brief Reports whether this value retains another storage control block.
    /// \return `true` for an alias value with a separate lifetime owner.
    [[nodiscard]] bool has_lifetime_owner() const noexcept { return ctrl_ && ctrl_->has_lifetime_owner(); }

    /// \brief Return the address reserved for the contained value.
    /// \details This address is stable for the lifetime of the control block and
    ///          is available before the value is constructed. It is intended for
    ///          lazy descriptors that store an address but do not dereference it
    ///          until their shared epoch is readable.
    /// \warning Do not dereference the returned pointer unless `constructed()` is true.
    [[nodiscard]] T* storage_address() noexcept { return ctrl_ ? ctrl_->storage_address() : nullptr; }

    /// \brief Return the const address reserved for the contained value.
    /// \warning Do not dereference the returned pointer unless `constructed()` is true.
    [[nodiscard]] T const* storage_address() const noexcept { return ctrl_ ? ctrl_->storage_address() : nullptr; }

    /// \brief Destroy any existing value and construct a new one in place.
    /// \tparam Args Constructor argument types.
    /// \param args Constructor arguments forwarded to `T`.
    /// \return Reference to the newly constructed value.
    template <typename... Args>
      requires std::constructible_from<T, Args...>
    T& emplace(Args&&... args)
    {
      DEBUG_CHECK(ctrl_, "shared_storage must be initialized with make_shared_storage()");
      if (ctrl_->value_constructed_locally()) ctrl_->destroy_object();
      ctrl_->construct(std::forward<Args>(args)...);
      return *ctrl_->ptr();
    }

    /// \brief Returns `true` when no control block is present.
    /// \return Negation of `valid()`.
    bool operator!() const noexcept { return !ctrl_; }; // no control block
    /// \brief Returns `true` when a control block is present.
    /// \return Equivalent to `valid()`.
    explicit operator bool() const noexcept { return bool(ctrl_); }; // has control block

    /// \brief Destroy the contained object while keeping the control block alive.
    void destroy() noexcept
    {
      if (ctrl_) ctrl_->destroy_object();
    }

    /// \brief Returns a mutable pointer to the constructed value.
    /// \return `nullptr` when no constructed value is present.
    [[nodiscard]] T* get() noexcept { return constructed() ? ctrl_->ptr() : nullptr; }
    /// \brief Returns a const pointer to the constructed value.
    /// \return `nullptr` when no constructed value is present.
    [[nodiscard]] const T* get() const noexcept { return this->constructed() ? ctrl_->ptr() : nullptr; }

    /// \brief Dereference access to the constructed value.
    /// \return Reference to the contained value.
    T& operator*() noexcept
    {
      DEBUG_CHECK(constructed());
      return *this->get();
    }
    /// \brief Dereference access to the constructed value.
    /// \return Const reference to the contained value.
    const T& operator*() const noexcept
    {
      DEBUG_CHECK(constructed());
      return *this->get();
    }

    /// \brief Pointer-style mutable access to the constructed value.
    /// \return Pointer to the contained value.
    T* operator->() noexcept
    {
      DEBUG_CHECK(constructed());
      return this->get();
    }
    /// \brief Pointer-style const access to the constructed value.
    /// \return Pointer to the contained value.
    const T* operator->() const noexcept
    {
      DEBUG_CHECK(constructed());
      return get();
    }

    /// \brief Move the contained value out and destroy the in-place object.
    /// \return Moved value.
    T take() noexcept(std::is_nothrow_move_constructible_v<T>)
    {
      T x = std::move(**this);
      this->destroy();
      return x;
    }

    /// \brief Construct storage with an immediately-constructed value.
    /// \tparam Args Constructor argument types.
    /// \param args Constructor arguments forwarded to `T`.
    /// \return New `shared_storage<T>` containing a constructed value.
    template <typename... Args> static shared_storage make_constructed(Args&&... args)
    {
      auto* c = new control_block{};
      try
      {
        c->construct(std::forward<Args>(args)...);
      }
      catch (...)
      {
        delete c;
        throw;
      }
      return shared_storage(c);
    }

    /// \brief Construct a value whose control block retains a separate lifetime owner.
    template <typename... Args>
    static shared_storage make_constructed_alias(detail::shared_storage_control_block* lifetime_owner, Args&&... args)
    {
      auto* c = new control_block{lifetime_owner};
      try
      {
        c->construct(std::forward<Args>(args)...);
      }
      catch (...)
      {
        delete c;
        throw;
      }
      return shared_storage(c);
    }

    /// \brief Compare whether two handles reference the same control block.
    /// \param a Left operand.
    /// \param b Right operand.
    /// \return `true` when both handles share identical storage metadata.
    friend bool operator==(const shared_storage& a, const shared_storage& b) noexcept { return a.ctrl_ == b.ctrl_; }
};

/// \brief Create a new shared_storage<T> with an unconstructed T.
template <typename T> [[nodiscard]] inline shared_storage<T> make_unconstructed_shared_storage()
{
  using ctrl_t = typename shared_storage<T>::control_block;
  return shared_storage<T>(new ctrl_t{});
}

/// \brief Create a new shared_storage<T> with T(args...) in-place.
/// \tparam T Stored value type.
/// \tparam Args Constructor argument types.
/// \param args Constructor arguments forwarded to `T`.
/// \return New `shared_storage<T>` with a constructed value.
template <typename T, typename... Args> [[nodiscard]] inline shared_storage<T> make_shared_storage(Args&&... args)
{
  return shared_storage<T>::make_constructed(std::forward<Args>(args)...);
}

/// \brief Create storage for a value that keeps another storage value alive.
/// \details The alias has its own local strong count. Its control block retains
///          one reference to `owner`, which is released after the alias value is
///          destroyed. Copies of the alias increment only the local count.
/// \tparam T Alias value type.
/// \tparam Owner Lifetime-owner value type.
/// \tparam Args Alias constructor argument types.
/// \param owner Storage whose lifetime must cover the alias value.
/// \param args Arguments forwarded to the alias value constructor.
/// \return Constructed alias storage sharing the owner's lifetime.
template <typename T, typename Owner, typename... Args>
[[nodiscard]] inline shared_storage<T> make_shared_storage_alias(shared_storage<Owner> const& owner, Args&&... args)
{
  CHECK(owner.valid());
  return shared_storage<T>::make_constructed_alias(owner.ctrl_, std::forward<Args>(args)...);
}

} // namespace uni20::async
