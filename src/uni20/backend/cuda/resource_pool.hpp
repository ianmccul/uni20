#pragma once

/**
 * \file resource_pool.hpp
 * \ingroup backend_cuda
 * \brief Fixed-capacity pools and move-only leases for CUDA provider resources.
 */

#include <uni20/common/trace.hpp>

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace uni20::cuda
{

template <class Resource> class ResourcePool;
template <class Resource> class ResourceLease;

namespace detail
{

/// \brief Intrusive callback node used by non-blocking provider-resource acquisition.
template <class Resource> class ResourceWaiter {
  public:
    using notify_type = void (*)(ResourceWaiter&, ResourceLease<Resource>) noexcept;

    explicit ResourceWaiter(notify_type notify) noexcept : notify_(notify) { CHECK(notify_ != nullptr); }
    ResourceWaiter(ResourceWaiter const&) = delete;
    ResourceWaiter& operator=(ResourceWaiter const&) = delete;

  private:
    void notify(ResourceLease<Resource> lease) noexcept { notify_(*this, std::move(lease)); }

    notify_type notify_;
    ResourceWaiter* next_ = nullptr;
    bool queued_ = false;

    friend class ResourcePool<Resource>;
};

} // namespace detail

/// \brief Move-only exclusive lease of one provider-resource pool slot.
/// \details Destroying or explicitly releasing the lease returns the resource
///          to the oldest queued waiter, or to the pool's idle set.
template <class Resource> class ResourceLease {
  public:
    ResourceLease() noexcept = default;
    ResourceLease(ResourceLease const&) = delete;
    ResourceLease& operator=(ResourceLease const&) = delete;

    ResourceLease(ResourceLease&& other) noexcept
        : pool_(std::exchange(other.pool_, nullptr)), slot_(std::exchange(other.slot_, 0))
    {}

    ResourceLease& operator=(ResourceLease&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        pool_ = std::exchange(other.pool_, nullptr);
        slot_ = std::exchange(other.slot_, 0);
      }
      return *this;
    }

    ~ResourceLease() { this->release(); }

    /// \brief Return whether this object currently owns a pool slot.
    [[nodiscard]] explicit operator bool() const noexcept { return pool_ != nullptr; }

    /// \brief Access the exclusively leased provider resource.
    [[nodiscard]] Resource& get() const;

    /// \brief Return the resource to its pool before this object's destruction.
    void release() noexcept;

  private:
    ResourceLease(ResourcePool<Resource>& pool, std::size_t slot) noexcept : pool_(&pool), slot_(slot) {}

    ResourcePool<Resource>* pool_ = nullptr;
    std::size_t slot_ = 0;

    friend class ResourcePool<Resource>;
};

/// \brief Fixed-capacity pool of preconstructed provider resources.
/// \details Non-blocking waiters receive released resources in FIFO order. The
///          pool is independent of scheduler and CUDA stream policy and must
///          outlive all leases and queued acquisition awaiters.
template <class Resource> class ResourcePool {
  public:
    using resource_type = Resource;
    using lease_type = ResourceLease<Resource>;
    using waiter_type = detail::ResourceWaiter<Resource>;

    /// \brief Take ownership of the resources that form this pool's slots.
    explicit ResourcePool(std::vector<Resource> resources)
        : resources_(std::move(resources)), idle_(resources_.size(), true), idle_count_(resources_.size())
    {
      CHECK(!resources_.empty(), "resource pool requires at least one slot");
    }

    ResourcePool(ResourcePool const&) = delete;
    ResourcePool& operator=(ResourcePool const&) = delete;
    ResourcePool(ResourcePool&&) = delete;
    ResourcePool& operator=(ResourcePool&&) = delete;

    ~ResourcePool()
    {
      std::lock_guard lock(mutex_);
      CHECK(waiter_head_ == nullptr, "destroying resource pool with queued waiters");
      CHECK(idle_count_ == resources_.size(), "destroying resource pool with active leases", idle_count_,
            resources_.size());
    }

    /// \brief Try to acquire one idle resource without waiting.
    [[nodiscard]] std::optional<lease_type> try_acquire()
    {
      std::lock_guard lock(mutex_);
      auto const slot = this->take_idle_slot_locked();
      if (!slot) return std::nullopt;
      return lease_type(*this, *slot);
    }

    /// \brief Block until one resource can be leased.
    [[nodiscard]] lease_type acquire()
    {
      std::unique_lock lock(mutex_);
      available_.wait(lock, [this] { return idle_count_ != 0; });
      auto const slot = this->take_idle_slot_locked();
      CHECK(slot.has_value());
      return lease_type(*this, *slot);
    }

    /// \brief Return the fixed number of resource slots.
    [[nodiscard]] std::size_t size() const noexcept { return resources_.size(); }

    /// \brief Return the number of resources currently available for acquisition.
    [[nodiscard]] std::size_t idle_count() const noexcept
    {
      std::lock_guard lock(mutex_);
      return idle_count_;
    }

    /// \brief Register an internal awaiter unless a resource is immediately available.
    [[nodiscard]] std::optional<lease_type> acquire_or_enqueue(waiter_type& waiter) noexcept
    {
      std::lock_guard lock(mutex_);
      CHECK(!waiter.queued_);
      if (auto const slot = this->take_idle_slot_locked())
      {
        return lease_type(*this, *slot);
      }

      waiter.queued_ = true;
      waiter.next_ = nullptr;
      if (waiter_tail_ == nullptr)
      {
        waiter_head_ = &waiter;
      }
      else
      {
        waiter_tail_->next_ = &waiter;
      }
      waiter_tail_ = &waiter;
      return std::nullopt;
    }

  private:
    [[nodiscard]] std::optional<std::size_t> take_idle_slot_locked() noexcept
    {
      if (idle_count_ == 0) return std::nullopt;
      for (std::size_t slot = 0; slot < idle_.size(); ++slot)
      {
        if (idle_[slot])
        {
          idle_[slot] = false;
          --idle_count_;
          return slot;
        }
      }
      PANIC("resource pool idle count is inconsistent", idle_count_, idle_.size());
    }

    [[nodiscard]] Resource& resource(std::size_t slot)
    {
      CHECK(slot < resources_.size(), slot, resources_.size());
      return resources_[slot];
    }

    void release(std::size_t slot) noexcept
    {
      waiter_type* waiter = nullptr;
      {
        std::lock_guard lock(mutex_);
        CHECK(slot < resources_.size(), slot, resources_.size());
        CHECK(!idle_[slot], "resource pool slot was released twice", slot);

        if (waiter_head_ == nullptr)
        {
          idle_[slot] = true;
          ++idle_count_;
          available_.notify_one();
          return;
        }

        waiter = waiter_head_;
        waiter_head_ = waiter->next_;
        if (waiter_head_ == nullptr) waiter_tail_ = nullptr;
        waiter->next_ = nullptr;
        waiter->queued_ = false;
      }

      waiter->notify(lease_type(*this, slot));
    }

    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::vector<Resource> resources_;
    std::vector<bool> idle_;
    std::size_t idle_count_;
    waiter_type* waiter_head_ = nullptr;
    waiter_type* waiter_tail_ = nullptr;

    friend class ResourceLease<Resource>;
};

template <class Resource> Resource& ResourceLease<Resource>::get() const
{
  CHECK(pool_ != nullptr, "cannot access an empty resource lease");
  return pool_->resource(slot_);
}

template <class Resource> void ResourceLease<Resource>::release() noexcept
{
  if (pool_ == nullptr) return;
  auto* pool = std::exchange(pool_, nullptr);
  auto const slot = std::exchange(slot_, 0);
  pool->release(slot);
}

} // namespace uni20::cuda
