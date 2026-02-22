#pragma once
#include "common/trace.hpp"
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
    struct control_block
    {
        alignas(T) unsigned char storage[sizeof(T)];
        std::atomic<size_t> strong_count{1};
        std::atomic<bool> constructed{false};

        T* ptr() noexcept { return std::launder(reinterpret_cast<T*>(storage)); }

        template <typename... Args> void construct(Args&&... args)
        {
          DEBUG_CHECK(!constructed.load(std::memory_order_relaxed));
          ::new (storage) T(std::forward<Args>(args)...);
          constructed.store(true, std::memory_order_release);
        }

        void destroy_object() noexcept
        {
          if (constructed.exchange(false, std::memory_order_acq_rel)) ptr()->~T();
        }

        void add_ref() noexcept { strong_count.fetch_add(1, std::memory_order_relaxed); }

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

    explicit shared_storage(control_block* c) noexcept : ctrl_(c) {}

  public:
    using element_type = T;

    shared_storage() noexcept = default;

    shared_storage(const shared_storage& other) noexcept : ctrl_(other.ctrl_)
    {
      if (ctrl_) ctrl_->add_ref();
    }

    shared_storage(shared_storage&& other) noexcept : ctrl_(std::exchange(other.ctrl_, nullptr)) {}

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

    shared_storage& operator=(shared_storage&& other) noexcept
    {
      if (this != &other)
      {
        this->reset();
        ctrl_ = std::exchange(other.ctrl_, nullptr);
      }
      return *this;
    }

    ~shared_storage() { reset(); }

    void reset() noexcept
    {
      if (ctrl_)
      {
        ctrl_->release_ref();
        ctrl_ = nullptr;
      }
    }

    bool constructed() const noexcept { return ctrl_ && ctrl_->constructed.load(std::memory_order_acquire); }

    bool valid() const noexcept { return ctrl_ != nullptr; }

    size_t use_count() const noexcept { return ctrl_ ? ctrl_->strong_count.load(std::memory_order_relaxed) : 0; }

    template <typename... Args>
    requires std::constructible_from<T, Args...> T& emplace(Args&&... args)
    {
      DEBUG_CHECK(ctrl_, "shared_storage must be initialized with make_shared_storage()");
      if (this->constructed()) ctrl_->destroy_object();
      ctrl_->construct(std::forward<Args>(args)...);
      return *ctrl_->ptr();
    }

    bool operator!() const noexcept { return !ctrl_; };              // no control block
    explicit operator bool() const noexcept { return bool(ctrl_); }; // has control block

    void destroy() noexcept
    {
      if (ctrl_) ctrl_->destroy_object();
    }

    T* get() noexcept { return constructed() ? ctrl_->ptr() : nullptr; }
    const T* get() const noexcept { return this->constructed() ? ctrl_->ptr() : nullptr; }

    T& operator*() noexcept
    {
      DEBUG_CHECK(constructed());
      return *this->get();
    }
    const T& operator*() const noexcept
    {
      DEBUG_CHECK(constructed());
      return *this->get();
    }

    T* operator->() noexcept
    {
      DEBUG_CHECK(constructed());
      return this->get();
    }
    const T* operator->() const noexcept
    {
      DEBUG_CHECK(constructed());
      return get();
    }

    T take() noexcept(std::is_nothrow_move_constructible_v<T>)
    {
      T x = std::move(**this);
      this->destroy();
      return x;
    }

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

    friend bool operator==(const shared_storage& a, const shared_storage& b) noexcept { return a.ctrl_ == b.ctrl_; }
};

/// \brief Create a new shared_storage<T> with an unconstructed T.
template <typename T> [[nodiscard]] inline shared_storage<T> make_unconstructed_shared_storage()
{
  using ctrl_t = typename shared_storage<T>::control_block;
  return shared_storage<T>(new ctrl_t{});
}

/// \brief Create a new shared_storage<T> with T(args...) in-place
template <typename T, typename... Args> [[nodiscard]] inline shared_storage<T> make_shared_storage(Args&&... args)
{
  return shared_storage<T>::make_constructed(std::forward<Args>(args)...);
}

} // namespace uni20::async
