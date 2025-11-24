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
      external_owner_.reset();
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

    T* reset_external_pointer(T* ptr, std::shared_ptr<void> owner = {}) noexcept
    {
      this->destroy_if_owned();
      external_owner_ = std::move(owner);
      owns_storage_ = false;
      constructed_.store(true, std::memory_order_release);
      value_ptr_cache_.store(ptr, std::memory_order_release);
      return ptr;
    }

    void set_external_owner(std::shared_ptr<void> owner) noexcept
    {
      external_owner_ = std::move(owner);
    }

    std::shared_ptr<void> const& external_owner() const noexcept { return external_owner_; }

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
    std::shared_ptr<void> external_owner_{};
};

} // namespace uni20::async::detail

