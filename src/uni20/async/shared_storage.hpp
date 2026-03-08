#pragma once
/// \file shared_storage.hpp
/// \brief Reference-counted optional in-place storage used by async buffers.
#include <uni20/common/trace.hpp>
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace uni20::async
{

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
/// There is no facility (yet!) to share ownership with subobjects. This may be a useful facility
/// (eg for tensor views)
template <typename T> class shared_storage {
  private:
    /// \brief Control block holding storage, construction flag, and reference count.
    struct control_block
    {
        alignas(T) unsigned char storage[sizeof(T)];
        std::atomic<size_t> strong_count{1};
        std::atomic<bool> constructed{false};

        /// \brief Returns a typed pointer to in-place storage.
        /// \return Pointer to the storage region as `T*`.
        T* ptr() noexcept { return std::launder(reinterpret_cast<T*>(storage)); }

        /// \brief Construct a `T` object in-place.
        /// \tparam Args Constructor argument types.
        /// \param args Constructor arguments.
        template <typename... Args> void construct(Args&&... args)
        {
          DEBUG_CHECK(!constructed.load(std::memory_order_relaxed));
          ::new (storage) T(std::forward<Args>(args)...);
          constructed.store(true, std::memory_order_release);
        }

        /// \brief Destroy the in-place object when currently constructed.
        void destroy_object() noexcept
        {
          if (constructed.exchange(false, std::memory_order_acq_rel)) ptr()->~T();
        }

        /// \brief Increment the strong reference count.
        void add_ref() noexcept { strong_count.fetch_add(1, std::memory_order_relaxed); }

        /// \brief Decrement the strong reference count and delete on zero.
        void release_ref() noexcept
        {
          if (strong_count.fetch_sub(1, std::memory_order_acq_rel) == 1)
          {
            destroy_object();
            delete this;
          }
        }
    };

    control_block* ctrl_ = nullptr;

    template <typename U, typename... Args> friend shared_storage<U> make_shared_storage(Args&&... args);
    template <typename U> friend shared_storage<U> make_unconstructed_shared_storage();

    /// \brief Construct from a raw control block pointer.
    /// \param c Control block pointer.
    explicit shared_storage(control_block* c) noexcept : ctrl_(c) {}

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
    bool constructed() const noexcept { return ctrl_ && ctrl_->constructed.load(std::memory_order_acquire); }

    /// \brief Reports whether a control block is present.
    /// \return `true` when this handle owns or shares storage metadata.
    bool valid() const noexcept { return ctrl_ != nullptr; }

    /// \brief Returns the current strong reference count.
    /// \return Number of `shared_storage` handles sharing this control block.
    size_t use_count() const noexcept { return ctrl_ ? ctrl_->strong_count.load(std::memory_order_relaxed) : 0; }

    /// \brief Destroy any existing value and construct a new one in place.
    /// \tparam Args Constructor argument types.
    /// \param args Constructor arguments forwarded to `T`.
    /// \return Reference to the newly constructed value.
    template <typename... Args>
    requires std::constructible_from<T, Args...> T& emplace(Args&&... args)
    {
      DEBUG_CHECK(ctrl_, "shared_storage must be initialized with make_shared_storage()");
      if (this->constructed()) ctrl_->destroy_object();
      ctrl_->construct(std::forward<Args>(args)...);
      return *ctrl_->ptr();
    }

    /// \brief Returns `true` when no control block is present.
    /// \return Negation of `valid()`.
    bool operator!() const noexcept { return !ctrl_; };              // no control block
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
    T* get() noexcept { return constructed() ? ctrl_->ptr() : nullptr; }
    /// \brief Returns a const pointer to the constructed value.
    /// \return `nullptr` when no constructed value is present.
    const T* get() const noexcept { return this->constructed() ? ctrl_->ptr() : nullptr; }

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

} // namespace uni20::async
