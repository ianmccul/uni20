#pragma once
#include <atomic>
#include <cassert>
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
/// - Optional weak references via `weak_storage<T>`
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
template <typename T> class shared_storage {
    struct control_block
    {
        std::atomic<size_t> strong_count{1};
        std::atomic<size_t> weak_count{1};
        std::atomic<bool> constructed{false};
        alignas(T) unsigned char storage[sizeof(T)];

        T* ptr() noexcept { return std::launder(reinterpret_cast<T*>(storage)); }

        void destroy_object() noexcept
        {
          if (constructed.exchange(false, std::memory_order_acq_rel)) ptr()->~T();
        }

        void add_strong() noexcept { strong_count.fetch_add(1, std::memory_order_relaxed); }
        void release_strong() noexcept
        {
          if (strong_count.fetch_sub(1, std::memory_order_acq_rel) == 1)
          {
            destroy_object();
            release_weak();
          }
        }

        void add_weak() noexcept { weak_count.fetch_add(1, std::memory_order_relaxed); }
        void release_weak() noexcept
        {
          if (weak_count.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this;
        }
    };

    control_block* ctrl_ = nullptr;

    explicit shared_storage(control_block* c) noexcept : ctrl_(c) {}

  public:
    using element_type = T;

    /// Construct an empty handle (no ownership)
    shared_storage() noexcept = default;

    /// Copy constructor (increments refcount)
    shared_storage(const shared_storage& other) noexcept : ctrl_(other.ctrl_)
    {
      if (ctrl_) ctrl_->add_strong();
    }

    /// Move constructor (steals ownership)
    shared_storage(shared_storage&& other) noexcept : ctrl_(std::exchange(other.ctrl_, nullptr)) {}

    /// Copy assignment
    shared_storage& operator=(const shared_storage& other) noexcept
    {
      if (this != &other)
      {
        reset();
        ctrl_ = other.ctrl_;
        if (ctrl_) ctrl_->add_strong();
      }
      return *this;
    }

    /// Move assignment
    shared_storage& operator=(shared_storage&& other) noexcept
    {
      if (this != &other)
      {
        reset();
        ctrl_ = std::exchange(other.ctrl_, nullptr);
      }
      return *this;
    }

    ~shared_storage() { reset(); }

    /// Reset to empty (decrements reference count)
    void reset() noexcept
    {
      if (ctrl_)
      {
        ctrl_->release_strong();
        ctrl_ = nullptr;
      }
    }

    /// Returns true if the managed object exists
    bool constructed() const noexcept { return ctrl_ && ctrl_->constructed.load(std::memory_order_acquire); }

    /// Returns true if this handle manages ownership
    bool valid() const noexcept { return ctrl_ != nullptr; }

    /// Returns the use count
    size_t use_count() const noexcept { return ctrl_ ? ctrl_->strong_count.load(std::memory_order_relaxed) : 0; }

    /// Construct the object in-place
    template <typename... Args> T& emplace(Args&&... args)
    {
      assert(ctrl_ && "shared_storage must be initialized with make_shared_storage()");
      assert(!constructed() && "Object already constructed");
      T* ptr = ctrl_->ptr();
      ::new (ptr) T(std::forward<Args>(args)...);
      ctrl_->constructed.store(true, std::memory_order_release);
      return *ptr;
    }

    /// Destroys the contained object (if constructed)
    void destroy() noexcept
    {
      if (ctrl_) ctrl_->destroy_object();
    }

    /// Returns a pointer to the managed object, or nullptr if not constructed
    T* get() noexcept { return constructed() ? ctrl_->ptr() : nullptr; }

    /// Returns a const pointer to the managed object, or nullptr if not constructed
    const T* get() const noexcept { return constructed() ? ctrl_->ptr() : nullptr; }

    /// Dereference operator (requires constructed object)
    T& operator*() noexcept
    {
      assert(constructed());
      return *get();
    }

    const T& operator*() const noexcept
    {
      assert(constructed());
      return *get();
    }

    /// Arrow operator (requires constructed object)
    T* operator->() noexcept
    {
      assert(constructed());
      return get();
    }

    const T* operator->() const noexcept
    {
      assert(constructed());
      return get();
    }

    /// Get a weak reference to this storage
    // (defined later)
    friend class weak_storage<T>;

    friend bool operator==(const shared_storage& a, const shared_storage& b) noexcept { return a.ctrl_ == b.ctrl_; }
};

/// \brief Non-owning reference to a shared_storage
template <typename T> class weak_storage {
    using control_block = typename shared_storage<T>::control_block;
    control_block* ctrl_ = nullptr;

  public:
    weak_storage() noexcept = default;
    weak_storage(const shared_storage<T>& strong) noexcept : ctrl_(strong.ctrl_)
    {
      if (ctrl_) ctrl_->add_weak();
    }
    weak_storage(const weak_storage& other) noexcept : ctrl_(other.ctrl_)
    {
      if (ctrl_) ctrl_->add_weak();
    }
    weak_storage(weak_storage&& other) noexcept : ctrl_(std::exchange(other.ctrl_, nullptr)) {}

    weak_storage& operator=(const weak_storage& other) noexcept
    {
      if (this != &other)
      {
        reset();
        ctrl_ = other.ctrl_;
        if (ctrl_) ctrl_->add_weak();
      }
      return *this;
    }

    weak_storage& operator=(weak_storage&& other) noexcept
    {
      if (this != &other)
      {
        reset();
        ctrl_ = std::exchange(other.ctrl_, nullptr);
      }
      return *this;
    }

    ~weak_storage() { reset(); }

    void reset() noexcept
    {
      if (ctrl_)
      {
        ctrl_->release_weak();
        ctrl_ = nullptr;
      }
    }

    bool expired() const noexcept { return !ctrl_ || ctrl_->strong_count.load(std::memory_order_acquire) == 0; }

    shared_storage<T> lock() const noexcept
    {
      if (!ctrl_) return {};
      size_t count = ctrl_->strong_count.load(std::memory_order_acquire);
      while (count != 0)
      {
        if (ctrl_->strong_count.compare_exchange_weak(count, count + 1, std::memory_order_acq_rel))
          return shared_storage<T>(ctrl_);
      }
      return {};
    }
};

/// \brief Allocate a new shared_storage<T> (unconstructed)
template <typename T> [[nodiscard]] inline shared_storage<T> make_shared_storage()
{
  using ctrl_t = typename shared_storage<T>::control_block;
  return shared_storage<T>(new ctrl_t{});
}

} // namespace uni20::async
