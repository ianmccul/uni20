#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace uni20::async::detail
{

template <typename T> class StorageBuffer
{
  public:
    StorageBuffer() = default;

    template <typename... Args>
    explicit StorageBuffer(std::in_place_t, Args&&... args)
    {
      this->construct(std::forward<Args>(args)...);
    }

    StorageBuffer(StorageBuffer const&) = delete;
    StorageBuffer& operator=(StorageBuffer const&) = delete;

    StorageBuffer(StorageBuffer&&) = delete;
    StorageBuffer& operator=(StorageBuffer&&) = delete;

    ~StorageBuffer() { this->destroy_if_owned(); }

    T* get() noexcept { return value_ptr_cache_.load(std::memory_order_acquire); }
    T const* get() const noexcept { return value_ptr_cache_.load(std::memory_order_acquire); }

    bool constructed() const noexcept { return constructed_.load(std::memory_order_acquire); }

    template <typename... Args> T* construct(Args&&... args)
    {
      this->destroy_if_owned();
      auto* ptr = this->storage_ptr();
      std::construct_at(ptr, std::forward<Args>(args)...);
      owns_storage_ = true;
      constructed_.store(true, std::memory_order_release);
      value_ptr_cache_.store(ptr, std::memory_order_release);
      return ptr;
    }

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

    T* reset_external_pointer(T* ptr) noexcept
    {
      this->destroy_if_owned();
      owns_storage_ = false;
      constructed_.store(true, std::memory_order_release);
      value_ptr_cache_.store(ptr, std::memory_order_release);
      return ptr;
    }

  private:
    T* storage_ptr() noexcept { return std::launder(reinterpret_cast<T*>(storage_)); }
    T const* storage_ptr() const noexcept { return std::launder(reinterpret_cast<T const*>(storage_)); }

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
};

} // namespace uni20::async::detail

