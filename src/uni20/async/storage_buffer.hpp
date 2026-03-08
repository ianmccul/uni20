#pragma once

/**
 * \file storage_buffer.hpp
 * \brief Defines an in-place storage helper used by async shared storage.
 */

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

namespace uni20::async::detail
{

/// \brief Manages optional in-place or externally-owned storage for a value.
/// \tparam T Stored value type.
template <typename T> class StorageBuffer {
  public:
    /// \brief Constructs an empty storage buffer.
    StorageBuffer() = default;

    /// \brief Constructs a value in-place from forwarded arguments.
    /// \tparam Args Constructor argument types.
    /// \param args Constructor arguments for `T`.
    template <typename... Args> explicit StorageBuffer(Args&&... args) { this->construct(std::forward<Args>(args)...); }

    StorageBuffer(StorageBuffer const&) = delete;
    StorageBuffer& operator=(StorageBuffer const&) = delete;

    StorageBuffer(StorageBuffer&&) = delete;
    StorageBuffer& operator=(StorageBuffer&&) = delete;

    /// \brief Destroys an in-place owned object if one is currently constructed.
    ~StorageBuffer() { this->destroy_if_owned(); }

    /// \brief Returns the current value pointer when a value is available.
    /// \return Pointer to the current value, or `nullptr` if unconstructed.
    T* get() noexcept { return value_ptr_cache_.load(std::memory_order_acquire); }
    /// \brief Returns the current value pointer when a value is available.
    /// \return Pointer to the current value, or `nullptr` if unconstructed.
    T const* get() const noexcept { return value_ptr_cache_.load(std::memory_order_acquire); }

    /// \brief Reports whether the buffer currently represents a constructed value.
    /// \return `true` when a value is available, otherwise `false`.
    bool constructed() const noexcept { return constructed_.load(std::memory_order_acquire); }

    /// \brief Constructs a new in-place value and drops any previous storage binding.
    /// \tparam Args Constructor argument types.
    /// \param args Constructor arguments forwarded to `T`.
    /// \return Pointer to the newly-constructed in-place value.
    template <typename... Args> T* construct(Args&&... args)
    {
      this->destroy_if_owned();
      external_owner_.reset();
      auto* ptr = this->storage_ptr();
      std::construct_at(ptr, std::forward<Args>(args)...);
      owns_storage_ = true;
      constructed_.store(true, std::memory_order_release);
      value_ptr_cache_.store(ptr, std::memory_order_release);
      return ptr;
    }

    /// \brief Returns the current value pointer, default-constructing when possible.
    /// \return Pointer to the value, or `nullptr` when `T` is not default-constructible.
    T* ensure_default()
    {
      if (auto* cached = this->get()) return cached;

      if constexpr (!std::is_default_constructible_v<T>)
      {
        return nullptr;
      }
      else
      {
        std::call_once(default_once_, [this]() { this->construct(); });
        return this->get();
      }
    }

    /// \brief Binds the buffer to an externally-owned value pointer.
    /// \param ptr External value pointer to expose.
    /// \param owner Optional shared owner that keeps `ptr` alive.
    /// \return The bound pointer `ptr`.
    T* reset_external_pointer(T* ptr, std::shared_ptr<void> owner = {}) noexcept
    {
      this->destroy_if_owned();
      external_owner_ = std::move(owner);
      owns_storage_ = false;
      constructed_.store(true, std::memory_order_release);
      value_ptr_cache_.store(ptr, std::memory_order_release);
      return ptr;
    }

    /// \brief Updates the shared ownership object for external storage.
    /// \param owner Shared owner object.
    void set_external_owner(std::shared_ptr<void> owner) noexcept { external_owner_ = std::move(owner); }

    /// \brief Returns the shared owner associated with external storage.
    /// \return Shared owner object.
    std::shared_ptr<void> const& external_owner() const noexcept { return external_owner_; }

  private:
    /// \brief Returns the typed pointer to in-place storage.
    /// \return Pointer to the aligned in-place storage block.
    T* storage_ptr() noexcept { return std::launder(reinterpret_cast<T*>(storage_)); }
    /// \brief Returns the typed pointer to in-place storage.
    /// \return Pointer to the aligned in-place storage block.
    T const* storage_ptr() const noexcept { return std::launder(reinterpret_cast<T const*>(storage_)); }

    /// \brief Destroys the in-place value when this buffer owns it.
    void destroy_if_owned() noexcept
    {
      if (constructed_.load(std::memory_order_acquire) && owns_storage_)
      {
        std::destroy_at(this->storage_ptr());
      }
      constructed_.store(false, std::memory_order_release);
      value_ptr_cache_.store(nullptr, std::memory_order_release);
    }

    alignas(T) std::byte storage_[sizeof(T)]{};
    std::atomic<T*> value_ptr_cache_{nullptr};
    std::atomic<bool> constructed_{false};
    bool owns_storage_{false};
    std::once_flag default_once_{};
    std::shared_ptr<void> external_owner_{};
};

} // namespace uni20::async::detail
