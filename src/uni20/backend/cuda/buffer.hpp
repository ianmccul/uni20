#pragma once

/**
 * \file buffer.hpp
 * \ingroup backend_cuda
 * \brief Typed CUDA device buffers and scoped stream access guards.
 */

#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/backend/cuda/device.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/common/trace.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::cuda
{

template <typename T = std::byte> class CudaBuffer;
template <typename T> class ReadAccess;
template <typename T> class WriteAccess;
template <typename T> class BlockingReadAccess;
template <typename T> class BlockingWriteAccess;
template <typename T> class OwningReadAccess;
template <typename T> class PartitionedReadAccess;
template <typename T> class PartitionedWriteAccess;
class AccessCompletion;

/// \brief One logical range used to partition a CUDA allocation.
struct CudaBufferRange
{
    std::size_t offset = 0;
    std::size_t size = 0;
};

namespace detail
{

[[noreturn]] inline void cuda_cleanup_failure(cudaError_t status, char const* operation, int device)
{
  PANIC("CUDA cleanup operation failed", operation, device, cudaGetErrorName(status), cudaGetErrorString(status));
}

inline void check_cuda_cleanup(cudaError_t status, char const* operation, int device) noexcept
{
  if (status != cudaSuccess)
  {
    cuda_cleanup_failure(status, operation, device);
  }
}

class BufferCleanupDeviceGuard {
  public:
    explicit BufferCleanupDeviceGuard(int device) noexcept
    {
      check_cuda_cleanup(cudaGetDevice(&previous_device_), "cudaGetDevice", device);
      restore_ = previous_device_ != device;
      if (restore_)
      {
        check_cuda_cleanup(cudaSetDevice(device), "cudaSetDevice", device);
      }
    }

    BufferCleanupDeviceGuard(BufferCleanupDeviceGuard const&) = delete;
    BufferCleanupDeviceGuard& operator=(BufferCleanupDeviceGuard const&) = delete;

    ~BufferCleanupDeviceGuard()
    {
      if (restore_)
      {
        check_cuda_cleanup(cudaSetDevice(previous_device_), "cudaSetDevice restore", previous_device_);
      }
    }

  private:
    int previous_device_ = -1;
    bool restore_ = false;
};

inline void synchronize_after_failed_publication(Stream const& stream, char const* operation) noexcept
{
  try
  {
    stream.synchronize();
  }
  catch (...)
  {
    PANIC("CUDA buffer completion publication failed and stream synchronization also failed", operation,
          stream.device());
  }
}

inline void synchronize_after_failed_publication(cudaStream_t stream, int device, char const* operation) noexcept
{
  BufferCleanupDeviceGuard guard(device);
  cudaError_t const status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess)
  {
    PANIC("CUDA buffer completion publication failed and stream synchronization also failed", operation, device,
          cudaGetErrorName(status), cudaGetErrorString(status));
  }
}

template <typename T> class CudaAllocation {
  public:
    CudaAllocation(DeviceResources& resources, std::size_t size) : resources_(&resources), size_(size)
    {
      CHECK(size <= std::numeric_limits<std::size_t>::max() / sizeof(T), size, sizeof(T));
      std::size_t const bytes = size * sizeof(T);
      if (size != 0)
      {
        if (resources.device().capabilities().memory_pools_supported)
          this->allocate_stream_ordered(bytes);
        else
          this->allocate_blocking(bytes);
      }
      resources_->register_allocation();
      registered_ = true;
    }

    CudaAllocation(CudaAllocation const&) = delete;
    CudaAllocation& operator=(CudaAllocation const&) = delete;
    ~CudaAllocation() { this->reset(); }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] T const* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] DeviceResources& resources() const
    {
      CHECK(resources_ != nullptr);
      return *resources_;
    }
    [[nodiscard]] Completion const& initial_completion() const noexcept { return initial_completion_; }
    void require_blocking_deallocation() noexcept { force_blocking_deallocation_.store(true); }
    void defer_release_after(Completion const& writer, std::span<Completion const> readers) noexcept
    {
      if (writer) resources_->defer_allocation_release_after(std::span(&writer, 1));
      if (!readers.empty()) resources_->defer_allocation_release_after(readers);
    }

  private:
    void allocate_blocking(std::size_t bytes)
    {
      void* raw_data = nullptr;
      ScopedDevice guard(resources_->device().ordinal());
      check(cudaMalloc(&raw_data, bytes), "cudaMalloc", resources_->device().ordinal());
      data_ = static_cast<T*>(raw_data);
    }

    void allocate_stream_ordered(std::size_t bytes)
    {
      auto stream = resources_->streams().acquire();
      void* raw_data = nullptr;
      ScopedDevice guard(resources_->device().ordinal());
      check(cudaMallocAsync(&raw_data, bytes, stream.native_handle()), "cudaMallocAsync",
            resources_->device().ordinal());
      data_ = static_cast<T*>(raw_data);

      try
      {
        initial_completion_ = stream.record_completion();
      }
      catch (...)
      {
        synchronize_after_failed_publication(stream, "publish CUDA async allocation completion");
        check_cuda_cleanup(cudaFreeAsync(data_, stream.native_handle()),
                           "cudaFreeAsync after cudaMallocAsync publication failure", resources_->device().ordinal());
        synchronize_after_failed_publication(stream, "free CUDA async allocation after publication failure");
        data_ = nullptr;
        size_ = 0;
        throw;
      }
    }

    void reset() noexcept
    {
      if (resources_ == nullptr) return;
      if (initial_completion_) resources_->defer_allocation_release_after(std::span(&initial_completion_, 1));

      if (data_ != nullptr)
      {
        resources_->release_allocation(data_, force_blocking_deallocation_.load());
      }

      auto* resources = std::exchange(resources_, nullptr);
      data_ = nullptr;
      size_ = 0;
      initial_completion_ = {};
      if (registered_) resources->unregister_allocation();
      registered_ = false;
    }

    DeviceResources* resources_ = nullptr;
    T* data_ = nullptr;
    std::size_t size_ = 0;
    Completion initial_completion_;
    std::atomic_bool force_blocking_deallocation_ = false;
    bool registered_ = false;
};

} // namespace detail

/// \brief Move-only logical CUDA buffer and completion state.
/// \details A buffer may own an entire CUDA allocation or one statically
///          disjoint region of a shared allocation. Each logical buffer has an
///          independent access ledger. Raw device pointers are exposed only
///          through scoped access objects.
template <typename T> class CudaBuffer {
  public:
    static_assert(std::is_object_v<T>, "CUDA buffers require an object element type");
    static_assert(!std::is_const_v<T>, "CUDA buffers own mutable storage; constness belongs on read guards");

    using element_type = T;
    using read_access_type = ReadAccess<T>;
    using write_access_type = WriteAccess<T>;
    using blocking_read_access_type = BlockingReadAccess<T>;
    using blocking_write_access_type = BlockingWriteAccess<T>;

    /// \brief Allocate `size` elements on the installed runtime's default CUDA device.
    explicit CudaBuffer(std::size_t size) : CudaBuffer(device_resources(), size) {}

    /// \brief Allocate `size` elements from one CUDA device's resources.
    CudaBuffer(DeviceResources& resources, std::size_t size)
        : allocation_(std::make_shared<detail::CudaAllocation<T>>(resources, size)), size_(size),
          writer_completion_(allocation_->initial_completion())
    {}

    CudaBuffer(CudaBuffer const&) = delete;
    CudaBuffer& operator=(CudaBuffer const&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept { this->move_from(other); }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept
    {
      if (this != &other)
      {
        this->reset();
        this->move_from(other);
      }
      return *this;
    }

    ~CudaBuffer() { this->reset(); }

    /// \brief Return the number of typed elements in the allocation.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// \brief Return the allocation size in bytes.
    [[nodiscard]] std::size_t size_bytes() const noexcept { return size_ * sizeof(T); }

    /// \brief Return whether this is a zero-sized or moved-from allocation.
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// \brief Return the device resources borrowed by this buffer.
    [[nodiscard]] DeviceResources& resources() const
    {
      CHECK(allocation_ != nullptr);
      return allocation_->resources();
    }

    /// \brief Return the allocation's CUDA device.
    [[nodiscard]] Device device() const { return this->resources().device(); }

    /// \brief Return this logical buffer's offset in its physical allocation.
    [[nodiscard]] std::size_t allocation_offset() const noexcept { return allocation_offset_; }

    /// \brief Return whether another logical buffer shares this physical allocation.
    [[nodiscard]] bool shares_allocation_with(CudaBuffer const& other) const noexcept
    {
      return allocation_ != nullptr && allocation_ == other.allocation_;
    }

    /// \brief Consume this buffer into statically disjoint logical child buffers.
    /// \details Every range is relative to this logical buffer. Ranges must be
    ///          sorted, non-overlapping, and contained within the buffer. The
    ///          source becomes moved-from and cannot bypass the child ledgers.
    [[nodiscard]] auto partition(std::span<CudaBufferRange const> ranges) && -> std::vector<CudaBuffer>
    {
      CHECK(allocation_ != nullptr, "cannot partition a moved-from CUDA buffer");
      std::size_t previous_end = 0;
      for (auto const& range : ranges)
      {
        CHECK(range.offset >= previous_end, range.offset, previous_end);
        CHECK(range.offset <= size_ && range.size <= size_ - range.offset, range.offset, range.size, size_);
        previous_end = range.offset + range.size;
      }

      std::lock_guard lock(state_mutex_);
      CHECK(live_read_accesses_ == 0 && !live_write_access_,
            "cannot partition a CUDA buffer while access guards are live", this->data(), live_read_accesses_,
            live_write_access_);
      std::vector<CudaBuffer> result;
      result.reserve(ranges.size());
      for (auto const& range : ranges)
      {
        result.push_back(CudaBuffer(allocation_, allocation_offset_ + range.offset, range.size, writer_completion_,
                                    reader_completions_));
      }
      allocation_.reset();
      allocation_offset_ = 0;
      size_ = 0;
      writer_completion_ = {};
      reader_completions_.clear();
      return result;
    }

    /// \brief Wait for every submitted operation currently involving this buffer.
    void synchronize() const
    {
      Completion writer_completion;
      std::vector<Completion> reader_completions;
      {
        std::lock_guard lock(state_mutex_);
        CHECK(live_read_accesses_ == 0 && !live_write_access_,
              "cannot synchronize a CUDA buffer while access guards are live", this->data(), live_read_accesses_,
              live_write_access_);
        writer_completion = writer_completion_;
        reader_completions = reader_completions_;
      }
      if (writer_completion)
      {
        writer_completion.synchronize();
      }
      for (Completion const& reader_completion : reader_completions)
      {
        reader_completion.synchronize();
      }
    }

    /// \brief Acquire scoped read-only access synchronized with `stream`.
    [[nodiscard]] ReadAccess<T> read_synchronized_with(Stream const& stream) const
    {
      return ReadAccess<T>(*this, stream);
    }

    /// \brief Acquire scoped read/write access synchronized with `stream`.
    [[nodiscard]] WriteAccess<T> write_synchronized_with(Stream const& stream) { return WriteAccess<T>(*this, stream); }

    /// \brief Acquire read-only access after host-waiting for prior CUDA work.
    [[nodiscard]] BlockingReadAccess<T> blocking_read_access() const { return BlockingReadAccess<T>(*this); }

    /// \brief Acquire read/write access after host-waiting for prior CUDA work.
    [[nodiscard]] BlockingWriteAccess<T> blocking_write_access() { return BlockingWriteAccess<T>(*this); }

    /// \brief Move this buffer into an owning host-synchronized read access.
    [[nodiscard]] OwningReadAccess<T> into_blocking_read_access() &&;

    /// \brief Move this buffer into an owning stream-ordered read access.
    [[nodiscard]] OwningReadAccess<T> into_read_synchronized_with(Stream const& stream) &&;

  private:
    CudaBuffer(std::shared_ptr<detail::CudaAllocation<T>> allocation, std::size_t allocation_offset, std::size_t size,
               Completion writer_completion, std::vector<Completion> reader_completions)
        : allocation_(std::move(allocation)), allocation_offset_(allocation_offset), size_(size),
          writer_completion_(std::move(writer_completion)), reader_completions_(std::move(reader_completions))
    {}

    [[nodiscard]] T* data() noexcept
    {
      return allocation_ == nullptr || allocation_->data() == nullptr ? nullptr
                                                                      : allocation_->data() + allocation_offset_;
    }
    [[nodiscard]] T const* data() const noexcept
    {
      return allocation_ == nullptr || allocation_->data() == nullptr ? nullptr
                                                                      : allocation_->data() + allocation_offset_;
    }

    void acquire_read_access(Stream const& stream) const
    {
      CHECK(stream, "cannot synchronize CUDA buffer access with an empty stream", this->data());
      Completion writer_completion;
      {
        std::lock_guard lock(state_mutex_);
        CHECK(!live_write_access_, "cannot acquire CUDA read access while a write access is live", this->data());
        writer_completion = writer_completion_;
        ++live_read_accesses_;
      }

      try
      {
        if (writer_completion)
        {
          stream.wait_on(writer_completion);
        }
      }
      catch (...)
      {
        std::lock_guard lock(state_mutex_);
        CHECK(live_read_accesses_ > 0);
        --live_read_accesses_;
        throw;
      }
    }

    void acquire_write_access(Stream const& stream) const
    {
      CHECK(stream, "cannot synchronize CUDA buffer access with an empty stream", this->data());
      Completion writer_completion;
      std::vector<Completion> reader_completions;
      {
        std::lock_guard lock(state_mutex_);
        CHECK(!live_write_access_ && live_read_accesses_ == 0,
              "cannot acquire CUDA write access while another access is live", this->data(), live_read_accesses_,
              live_write_access_);
        writer_completion = writer_completion_;
        std::erase_if(reader_completions_, [](Completion const& completion) { return completion.ready(); });
        reader_completions = reader_completions_;
        live_write_access_ = true;
      }

      try
      {
        if (writer_completion)
        {
          stream.wait_on(writer_completion);
        }
        for (Completion const& reader_completion : reader_completions)
        {
          stream.wait_on(reader_completion);
        }
      }
      catch (...)
      {
        std::lock_guard lock(state_mutex_);
        CHECK(live_write_access_);
        live_write_access_ = false;
        throw;
      }
    }

    void acquire_blocking_read_access() const
    {
      Completion writer_completion;
      {
        std::lock_guard lock(state_mutex_);
        CHECK(!live_write_access_, "cannot acquire blocking CUDA read access while a write access is live",
              this->data());
        writer_completion = writer_completion_;
        ++live_read_accesses_;
      }

      try
      {
        if (writer_completion)
        {
          writer_completion.synchronize();
        }
      }
      catch (...)
      {
        this->release_blocking_read_access();
        throw;
      }
    }

    void acquire_blocking_write_access() const
    {
      Completion writer_completion;
      std::vector<Completion> reader_completions;
      {
        std::lock_guard lock(state_mutex_);
        CHECK(!live_write_access_ && live_read_accesses_ == 0,
              "cannot acquire blocking CUDA write access while another access is live", this->data(),
              live_read_accesses_, live_write_access_);
        writer_completion = writer_completion_;
        reader_completions = reader_completions_;
        live_write_access_ = true;
      }

      try
      {
        if (writer_completion)
        {
          writer_completion.synchronize();
        }
        for (Completion const& reader_completion : reader_completions)
        {
          reader_completion.synchronize();
        }
      }
      catch (...)
      {
        std::lock_guard lock(state_mutex_);
        CHECK(live_write_access_);
        live_write_access_ = false;
        throw;
      }
    }

    void release_blocking_read_access() const noexcept
    {
      std::lock_guard lock(state_mutex_);
      CHECK(live_read_accesses_ > 0);
      --live_read_accesses_;
    }

    void release_blocking_write_access(Completion completion = {}) const noexcept
    {
      std::lock_guard lock(state_mutex_);
      CHECK(live_write_access_);
      writer_completion_ = std::move(completion);
      reader_completions_.clear();
      live_write_access_ = false;
    }

    void release_read_access(Stream const& stream) const noexcept
    {
      try
      {
        Completion completion = stream.record_completion();
        this->release_read_access(std::move(completion));
        return;
      }
      catch (...)
      {
        detail::synchronize_after_failed_publication(stream, "publish CUDA buffer reader completion");
      }
      this->release_read_access(Completion{});
    }

    void release_write_access(Stream const& stream) const noexcept
    {
      try
      {
        Completion completion = stream.record_completion();
        this->release_write_access(std::move(completion));
        return;
      }
      catch (...)
      {
        detail::synchronize_after_failed_publication(stream, "publish CUDA buffer writer completion");
      }

      this->release_write_access(Completion{});
    }

    void release_read_access(Completion completion) const noexcept
    {
      std::lock_guard lock(state_mutex_);
      CHECK(live_read_accesses_ > 0);
      if (completion)
      {
        if (reader_completions_.size() == reader_completions_.capacity())
        {
          std::erase_if(reader_completions_,
                        [](Completion const& reader_completion) { return reader_completion.ready(); });
        }
        reader_completions_.push_back(std::move(completion));
      }
      --live_read_accesses_;
    }

    void release_write_access(Completion completion) const noexcept
    {
      std::lock_guard lock(state_mutex_);
      CHECK(live_write_access_);
      writer_completion_ = std::move(completion);
      reader_completions_.clear();
      live_write_access_ = false;
    }

    void prepare_owning_read_access(Stream const& stream) const
    {
      CHECK(stream, "cannot synchronize owning CUDA buffer access with an empty stream", this->data());
      Completion writer_completion;
      {
        std::lock_guard lock(state_mutex_);
        CHECK(live_read_accesses_ == 0 && !live_write_access_,
              "cannot take owning CUDA buffer access while another access is live", this->data(), live_read_accesses_,
              live_write_access_);
        writer_completion = writer_completion_;
      }
      if (writer_completion)
      {
        stream.wait_on(writer_completion);
      }
    }

    void prepare_owning_blocking_read_access() const
    {
      Completion writer_completion;
      {
        std::lock_guard lock(state_mutex_);
        CHECK(live_read_accesses_ == 0 && !live_write_access_,
              "cannot take owning CUDA buffer access while another access is live", this->data(), live_read_accesses_,
              live_write_access_);
        writer_completion = writer_completion_;
      }
      if (writer_completion)
      {
        writer_completion.synchronize();
      }
    }

    void publish_owning_read_completion(Stream const& stream) const noexcept
    {
      try
      {
        Completion completion = stream.record_completion();
        std::lock_guard lock(state_mutex_);
        if (reader_completions_.size() == reader_completions_.capacity())
        {
          std::erase_if(reader_completions_,
                        [](Completion const& reader_completion) { return reader_completion.ready(); });
        }
        reader_completions_.push_back(std::move(completion));
        return;
      }
      catch (...)
      {
        detail::synchronize_after_failed_publication(stream, "publish owning CUDA buffer reader completion");
      }
    }

    void reset() noexcept { this->reset_impl(); }

    void reset_owned() noexcept
    {
      if (allocation_ != nullptr) allocation_->require_blocking_deallocation();
      this->reset_impl();
    }

    void reset_impl() noexcept
    {
      if (allocation_ == nullptr)
      {
        return;
      }

      Completion writer_completion;
      std::vector<Completion> reader_completions;
      {
        std::lock_guard lock(state_mutex_);
        CHECK(live_read_accesses_ == 0 && !live_write_access_,
              "cannot destroy or reset a CUDA buffer while access guards are live", this->data(), live_read_accesses_,
              live_write_access_);
        writer_completion = std::move(writer_completion_);
        reader_completions = std::move(reader_completions_);
      }
      allocation_->defer_release_after(writer_completion, reader_completions);

      allocation_.reset();
      allocation_offset_ = 0;
      size_ = 0;
      writer_completion_ = {};
      reader_completions_.clear();
    }

    void move_from(CudaBuffer& other) noexcept
    {
      if (other.allocation_ == nullptr)
      {
        return;
      }

      std::lock_guard lock(other.state_mutex_);
      CHECK(other.live_read_accesses_ == 0 && !other.live_write_access_,
            "cannot move a CUDA buffer while access guards are live", other.data(), other.live_read_accesses_,
            other.live_write_access_);
      allocation_ = std::move(other.allocation_);
      allocation_offset_ = std::exchange(other.allocation_offset_, 0);
      size_ = other.size_;
      writer_completion_ = std::move(other.writer_completion_);
      reader_completions_ = std::move(other.reader_completions_);
      live_read_accesses_ = 0;
      live_write_access_ = false;
      other.size_ = 0;
    }

    std::shared_ptr<detail::CudaAllocation<T>> allocation_;
    std::size_t allocation_offset_ = 0;
    std::size_t size_ = 0;
    mutable std::mutex state_mutex_;
    mutable Completion writer_completion_;
    mutable std::vector<Completion> reader_completions_;
    mutable std::size_t live_read_accesses_ = 0;
    mutable bool live_write_access_ = false;

    template <typename> friend class ReadAccess;
    template <typename> friend class WriteAccess;
    template <typename> friend class BlockingReadAccess;
    template <typename> friend class BlockingWriteAccess;
    template <typename> friend class OwningReadAccess;
    template <typename> friend class PartitionedReadAccess;
    template <typename> friend class PartitionedWriteAccess;
};

/// \brief One CUDA allocation partitioned into independent logical buffers.
/// \details Construction validates the disjoint ranges once. Each child owns
///          an independent completion ledger while sharing the physical
///          allocation and device resources with its siblings.
template <typename T> class PartitionedCudaBuffer {
  public:
    using element_type = T;
    using buffer_type = CudaBuffer<T>;

    PartitionedCudaBuffer(DeviceResources& resources, std::size_t size, std::span<CudaBufferRange const> ranges)
        : resources_(&resources), size_(size), buffers_(make_buffers(resources, size, ranges))
    {}

    PartitionedCudaBuffer(PartitionedCudaBuffer const&) = delete;
    PartitionedCudaBuffer& operator=(PartitionedCudaBuffer const&) = delete;
    PartitionedCudaBuffer(PartitionedCudaBuffer&& other) noexcept
        : resources_(std::exchange(other.resources_, nullptr)), size_(std::exchange(other.size_, 0)),
          buffers_(std::move(other.buffers_))
    {}
    PartitionedCudaBuffer& operator=(PartitionedCudaBuffer&& other) noexcept
    {
      if (this != &other)
      {
        buffers_ = std::move(other.buffers_);
        resources_ = std::exchange(other.resources_, nullptr);
        size_ = std::exchange(other.size_, 0);
      }
      return *this;
    }

    /// \brief Return the total physical allocation size in elements.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// \brief Return the number of logical child buffers.
    [[nodiscard]] std::size_t buffer_count() const noexcept { return buffers_.size(); }

    /// \brief Return one mutable logical child buffer.
    [[nodiscard]] CudaBuffer<T>& buffer(std::size_t ordinal) { return buffers_[ordinal]; }

    /// \brief Return one read-only logical child buffer.
    [[nodiscard]] CudaBuffer<T> const& buffer(std::size_t ordinal) const { return buffers_[ordinal]; }

    /// \brief Return the device resources shared by every child buffer.
    [[nodiscard]] DeviceResources& resources() const
    {
      CHECK(resources_ != nullptr);
      return *resources_;
    }

    /// \brief Return the CUDA device containing the physical allocation.
    [[nodiscard]] Device device() const { return this->resources().device(); }

    /// \brief Acquire read access to the complete contiguous allocation.
    /// \details The access waits on every child ledger and publishes one shared
    ///          completion to all children on release.
    [[nodiscard]] auto read_synchronized_with(Stream const& stream) const -> PartitionedReadAccess<T>;

    /// \brief Acquire write access to the complete contiguous allocation.
    /// \details The access waits on every child ledger and publishes one shared
    ///          completion to all children on release.
    [[nodiscard]] auto write_synchronized_with(Stream const& stream) -> PartitionedWriteAccess<T>;

  private:
    void require_complete_partition() const
    {
      std::size_t next_offset = 0;
      for (auto const& buffer : buffers_)
      {
        CHECK_EQUAL(buffer.allocation_offset(), next_offset);
        next_offset += buffer.size();
      }
      CHECK_EQUAL(next_offset, size_);
    }

    static auto make_buffers(DeviceResources& resources, std::size_t size,
                             std::span<CudaBufferRange const> ranges) -> std::vector<CudaBuffer<T>>
    {
      CudaBuffer<T> allocation(resources, size);
      return std::move(allocation).partition(ranges);
    }

    DeviceResources* resources_ = nullptr;
    std::size_t size_ = 0;
    std::vector<CudaBuffer<T>> buffers_;

    template <typename> friend class PartitionedReadAccess;
    template <typename> friend class PartitionedWriteAccess;
};

/// \brief Scoped read-only access to a CUDA buffer on one stream.
/// \details The constructor installs predecessor waits on the stream. The
///          destructor records a reader completion at the current stream tail
///          and publishes it back to the buffer.
template <typename T> class ReadAccess {
  public:
    using element_type = T;

    ReadAccess(ReadAccess const&) = delete;
    ReadAccess& operator=(ReadAccess const&) = delete;
    ReadAccess(ReadAccess&& other) noexcept
        : storage_(std::exchange(other.storage_, nullptr)), stream_(std::move(other.stream_))
    {}
    ReadAccess& operator=(ReadAccess&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        storage_ = std::exchange(other.storage_, nullptr);
        stream_ = std::move(other.stream_);
      }
      return *this;
    }
    ~ReadAccess() { this->release(); }

    /// \brief Return the typed device pointer for read-only CUDA work.
    [[nodiscard]] T const* data() const noexcept { return storage_ == nullptr ? nullptr : storage_->data(); }

    /// \brief Return the CUDA buffer retained by this access object.
    [[nodiscard]] CudaBuffer<T> const& storage() const
    {
      CHECK(storage_ != nullptr, "cannot inspect storage through a released CUDA read access");
      return *storage_;
    }

    /// \brief Return the number of typed elements in the view.
    [[nodiscard]] std::size_t size() const noexcept { return storage_ == nullptr ? 0 : storage_->size(); }

    /// \brief Return the number of bytes in the view.
    [[nodiscard]] std::size_t size_bytes() const noexcept { return storage_ == nullptr ? 0 : storage_->size_bytes(); }

    /// \brief Publish this read completion and end the scoped access.
    void release() noexcept
    {
      if (storage_ == nullptr)
      {
        return;
      }
      storage_->release_read_access(stream_);
      storage_ = nullptr;
      stream_ = {};
    }

    /// \brief Publish a caller-recorded completion and end scoped access.
    void release_with_completion(Completion completion) noexcept
    {
      if (storage_ == nullptr) return;
      CHECK(completion);
      storage_->release_read_access(std::move(completion));
      storage_ = nullptr;
      stream_ = {};
    }

    /// \brief End access after the caller has synchronized the operation stream.
    void release_after_synchronization() noexcept
    {
      if (storage_ == nullptr) return;
      storage_->release_read_access(Completion{});
      storage_ = nullptr;
      stream_ = {};
    }

  private:
    ReadAccess(CudaBuffer<T> const& storage, Stream const& stream) : stream_(stream)
    {
      storage.acquire_read_access(stream_);
      storage_ = &storage;
    }

    CudaBuffer<T> const* storage_ = nullptr;
    Stream stream_;

    friend class CudaBuffer<T>;
};

/// \brief Scoped read/write access to a CUDA buffer on one stream.
/// \details The constructor installs predecessor waits on the stream. The
///          destructor records an exclusive-writer completion at the current
///          stream tail and publishes it back to the buffer.
template <typename T> class WriteAccess {
  public:
    using element_type = T;

    WriteAccess(WriteAccess const&) = delete;
    WriteAccess& operator=(WriteAccess const&) = delete;
    WriteAccess(WriteAccess&& other) noexcept
        : storage_(std::exchange(other.storage_, nullptr)), stream_(std::move(other.stream_))
    {}
    WriteAccess& operator=(WriteAccess&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        storage_ = std::exchange(other.storage_, nullptr);
        stream_ = std::move(other.stream_);
      }
      return *this;
    }
    ~WriteAccess() { this->release(); }

    /// \brief Return the typed device pointer for mutating CUDA work.
    [[nodiscard]] T* data() const noexcept { return storage_ == nullptr ? nullptr : storage_->data(); }

    /// \brief Return the mutable CUDA buffer retained by this access object.
    [[nodiscard]] CudaBuffer<T>& storage()
    {
      CHECK(storage_ != nullptr, "cannot inspect storage through a released CUDA write access");
      return *storage_;
    }

    /// \brief Return the read-only CUDA buffer retained by this access object.
    [[nodiscard]] CudaBuffer<T> const& storage() const
    {
      CHECK(storage_ != nullptr, "cannot inspect storage through a released CUDA write access");
      return *storage_;
    }

    /// \brief Return the number of typed elements in the view.
    [[nodiscard]] std::size_t size() const noexcept { return storage_ == nullptr ? 0 : storage_->size(); }

    /// \brief Return the number of bytes in the view.
    [[nodiscard]] std::size_t size_bytes() const noexcept { return storage_ == nullptr ? 0 : storage_->size_bytes(); }

    /// \brief Publish this write completion and end the scoped access.
    void release() noexcept
    {
      if (storage_ == nullptr)
      {
        return;
      }
      storage_->release_write_access(stream_);
      storage_ = nullptr;
      stream_ = {};
    }

    /// \brief Publish a caller-recorded completion and end scoped access.
    void release_with_completion(Completion completion) noexcept
    {
      if (storage_ == nullptr) return;
      CHECK(completion);
      storage_->release_write_access(std::move(completion));
      storage_ = nullptr;
      stream_ = {};
    }

    /// \brief End access after the caller has synchronized the operation stream.
    void release_after_synchronization() noexcept
    {
      if (storage_ == nullptr) return;
      storage_->release_write_access(Completion{});
      storage_ = nullptr;
      stream_ = {};
    }

  private:
    WriteAccess(CudaBuffer<T>& storage, Stream const& stream) : stream_(stream)
    {
      storage.acquire_write_access(stream_);
      storage_ = &storage;
    }

    CudaBuffer<T>* storage_ = nullptr;
    Stream stream_;

    friend class CudaBuffer<T>;
};

/// \brief Scoped read access to every child of one partitioned CUDA allocation.
/// \details Construction installs predecessor waits for every logical child.
///          Release records one completion and publishes the shared token to
///          every child ledger.
template <typename T> class PartitionedReadAccess {
  public:
    using element_type = T;

    PartitionedReadAccess(PartitionedReadAccess const&) = delete;
    PartitionedReadAccess& operator=(PartitionedReadAccess const&) = delete;
    PartitionedReadAccess(PartitionedReadAccess&& other) noexcept
        : storage_(std::exchange(other.storage_, nullptr)), accesses_(std::move(other.accesses_)),
          stream_(std::move(other.stream_))
    {}
    PartitionedReadAccess& operator=(PartitionedReadAccess&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        storage_ = std::exchange(other.storage_, nullptr);
        accesses_ = std::move(other.accesses_);
        stream_ = std::move(other.stream_);
      }
      return *this;
    }
    ~PartitionedReadAccess() { this->release(); }

    /// \brief Return the first element of the complete physical allocation.
    [[nodiscard]] T const* data() const noexcept { return accesses_.empty() ? nullptr : accesses_.front().data(); }

    /// \brief Return the physical allocation size in elements.
    [[nodiscard]] std::size_t size() const noexcept { return storage_ == nullptr ? 0 : storage_->size(); }

    /// \brief Record and publish one shared reader completion.
    void release() noexcept
    {
      if (storage_ == nullptr) return;
      try
      {
        this->release_with_completion(stream_.record_completion());
        return;
      }
      catch (...)
      {
        detail::synchronize_after_failed_publication(stream_, "publish partitioned CUDA reader completion");
      }
      this->release_after_synchronization();
    }

    /// \brief Publish a caller-recorded completion to every child ledger.
    void release_with_completion(Completion completion) noexcept
    {
      if (storage_ == nullptr) return;
      CHECK(completion);
      for (auto& access : accesses_)
        access.release_with_completion(completion);
      this->clear();
    }

    /// \brief End access after the caller has synchronized the operation stream.
    void release_after_synchronization() noexcept
    {
      if (storage_ == nullptr) return;
      for (auto& access : accesses_)
        access.release_after_synchronization();
      this->clear();
    }

  private:
    PartitionedReadAccess(PartitionedCudaBuffer<T> const& storage, Stream const& stream)
        : storage_(&storage), stream_(stream)
    {
      storage.require_complete_partition();
      accesses_.reserve(storage.buffers_.size());
      for (auto const& buffer : storage.buffers_)
        accesses_.push_back(buffer.read_synchronized_with(stream_));
    }

    void clear() noexcept
    {
      accesses_.clear();
      storage_ = nullptr;
      stream_ = {};
    }

    PartitionedCudaBuffer<T> const* storage_ = nullptr;
    std::vector<ReadAccess<T>> accesses_;
    Stream stream_;

    friend class PartitionedCudaBuffer<T>;
};

/// \brief Scoped write access to every child of one partitioned CUDA allocation.
/// \details Construction installs predecessor waits for every logical child.
///          Release records one completion and publishes the shared token to
///          every child ledger.
template <typename T> class PartitionedWriteAccess {
  public:
    using element_type = T;

    PartitionedWriteAccess(PartitionedWriteAccess const&) = delete;
    PartitionedWriteAccess& operator=(PartitionedWriteAccess const&) = delete;
    PartitionedWriteAccess(PartitionedWriteAccess&& other) noexcept
        : storage_(std::exchange(other.storage_, nullptr)), accesses_(std::move(other.accesses_)),
          stream_(std::move(other.stream_))
    {}
    PartitionedWriteAccess& operator=(PartitionedWriteAccess&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        storage_ = std::exchange(other.storage_, nullptr);
        accesses_ = std::move(other.accesses_);
        stream_ = std::move(other.stream_);
      }
      return *this;
    }
    ~PartitionedWriteAccess() { this->release(); }

    /// \brief Return the first element of the complete physical allocation.
    [[nodiscard]] T* data() const noexcept { return accesses_.empty() ? nullptr : accesses_.front().data(); }

    /// \brief Return the physical allocation size in elements.
    [[nodiscard]] std::size_t size() const noexcept { return storage_ == nullptr ? 0 : storage_->size(); }

    /// \brief Record and publish one shared writer completion.
    void release() noexcept
    {
      if (storage_ == nullptr) return;
      try
      {
        this->release_with_completion(stream_.record_completion());
        return;
      }
      catch (...)
      {
        detail::synchronize_after_failed_publication(stream_, "publish partitioned CUDA writer completion");
      }
      this->release_after_synchronization();
    }

    /// \brief Publish a caller-recorded completion to every child ledger.
    void release_with_completion(Completion completion) noexcept
    {
      if (storage_ == nullptr) return;
      CHECK(completion);
      for (auto& access : accesses_)
        access.release_with_completion(completion);
      this->clear();
    }

    /// \brief End access after the caller has synchronized the operation stream.
    void release_after_synchronization() noexcept
    {
      if (storage_ == nullptr) return;
      for (auto& access : accesses_)
        access.release_after_synchronization();
      this->clear();
    }

  private:
    PartitionedWriteAccess(PartitionedCudaBuffer<T>& storage, Stream const& stream)
        : storage_(&storage), stream_(stream)
    {
      storage.require_complete_partition();
      accesses_.reserve(storage.buffers_.size());
      for (auto& buffer : storage.buffers_)
        accesses_.push_back(buffer.write_synchronized_with(stream_));
    }

    void clear() noexcept
    {
      accesses_.clear();
      storage_ = nullptr;
      stream_ = {};
    }

    PartitionedCudaBuffer<T>* storage_ = nullptr;
    std::vector<WriteAccess<T>> accesses_;
    Stream stream_;

    friend class PartitionedCudaBuffer<T>;
};

/// \brief One operation-tail completion shared by multiple CUDA buffer accesses.
/// \details Record this after successfully submitting all work for one operation,
///          then release every participating access through it. If event
///          publication fails, construction synchronizes the stream and subsequent
///          releases use the already-synchronized path.
class AccessCompletion {
  public:
    /// \brief Record the current stream tail, synchronizing only if publication fails.
    explicit AccessCompletion(Stream const& stream) noexcept
    {
      try
      {
        completion_ = stream.record_completion();
      }
      catch (...)
      {
        detail::synchronize_after_failed_publication(stream, "publish shared CUDA buffer access completion");
      }
    }

    AccessCompletion(AccessCompletion const&) = delete;
    AccessCompletion& operator=(AccessCompletion const&) = delete;
    AccessCompletion(AccessCompletion&&) noexcept = default;
    AccessCompletion& operator=(AccessCompletion&&) noexcept = default;

    /// \brief Release one participating access through the shared completion.
    template <class Access> void release(Access& access) const noexcept
    {
      if (completion_)
        access.release_with_completion(completion_);
      else
        access.release_after_synchronization();
    }

  private:
    Completion completion_;
};

template <typename T>
auto PartitionedCudaBuffer<T>::read_synchronized_with(Stream const& stream) const -> PartitionedReadAccess<T>
{
  return PartitionedReadAccess<T>{*this, stream};
}

template <typename T>
auto PartitionedCudaBuffer<T>::write_synchronized_with(Stream const& stream) -> PartitionedWriteAccess<T>
{
  return PartitionedWriteAccess<T>{*this, stream};
}

/// \brief Scoped read-only CUDA pointer access after a host-side synchronization.
/// \details Construction waits for the buffer's prior writer completion. No
///          CUDA completion is published on release because the caller must
///          finish every operation using the pointer before releasing it.
template <typename T> class BlockingReadAccess {
  public:
    using element_type = T;

    BlockingReadAccess(BlockingReadAccess const&) = delete;
    BlockingReadAccess& operator=(BlockingReadAccess const&) = delete;
    BlockingReadAccess(BlockingReadAccess&& other) noexcept : storage_(std::exchange(other.storage_, nullptr)) {}
    BlockingReadAccess& operator=(BlockingReadAccess&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        storage_ = std::exchange(other.storage_, nullptr);
      }
      return *this;
    }
    ~BlockingReadAccess() { this->release(); }

    /// \brief Return the typed device pointer for a blocking runtime operation.
    [[nodiscard]] T const* data() const noexcept { return storage_ == nullptr ? nullptr : storage_->data(); }

    /// \brief Return the CUDA buffer retained by this access object.
    [[nodiscard]] CudaBuffer<T> const& storage() const
    {
      CHECK(storage_ != nullptr, "cannot inspect storage through a released blocking CUDA read access");
      return *storage_;
    }

    /// \brief Return the number of typed elements in the allocation.
    [[nodiscard]] std::size_t size() const noexcept { return storage_ == nullptr ? 0 : storage_->size(); }

    /// \brief End the scoped host-synchronized read access.
    void release() noexcept
    {
      if (storage_ == nullptr) return;
      storage_->release_blocking_read_access();
      storage_ = nullptr;
    }

  private:
    explicit BlockingReadAccess(CudaBuffer<T> const& storage)
    {
      storage.acquire_blocking_read_access();
      storage_ = &storage;
    }

    CudaBuffer<T> const* storage_ = nullptr;

    friend class CudaBuffer<T>;
};

/// \brief Scoped read/write CUDA pointer access after a host-side synchronization.
/// \details Construction waits for all prior reader and writer completions.
///          Release publishes no CUDA event because the guarded operation must
///          have completed synchronously on the host.
template <typename T> class BlockingWriteAccess {
  public:
    using element_type = T;

    BlockingWriteAccess(BlockingWriteAccess const&) = delete;
    BlockingWriteAccess& operator=(BlockingWriteAccess const&) = delete;
    BlockingWriteAccess(BlockingWriteAccess&& other) noexcept : storage_(std::exchange(other.storage_, nullptr)) {}
    BlockingWriteAccess& operator=(BlockingWriteAccess&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        storage_ = std::exchange(other.storage_, nullptr);
      }
      return *this;
    }
    ~BlockingWriteAccess() { this->release(); }

    /// \brief Return the typed device pointer for a blocking runtime operation.
    [[nodiscard]] T* data() const noexcept { return storage_ == nullptr ? nullptr : storage_->data(); }

    /// \brief Return the mutable CUDA buffer retained by this access object.
    [[nodiscard]] CudaBuffer<T>& storage()
    {
      CHECK(storage_ != nullptr, "cannot inspect storage through a released blocking CUDA write access");
      return *storage_;
    }

    /// \brief Return the read-only CUDA buffer retained by this access object.
    [[nodiscard]] CudaBuffer<T> const& storage() const
    {
      CHECK(storage_ != nullptr, "cannot inspect storage through a released blocking CUDA write access");
      return *storage_;
    }

    /// \brief Return the number of typed elements in the allocation.
    [[nodiscard]] std::size_t size() const noexcept { return storage_ == nullptr ? 0 : storage_->size(); }

    /// \brief End the scoped host-synchronized write access.
    void release() noexcept
    {
      if (storage_ == nullptr) return;
      storage_->release_blocking_write_access();
      storage_ = nullptr;
    }

    /// \brief Publish externally submitted device completion and end the scoped access.
    /// \details Use this when a blocking host operation has finished consuming
    ///          its host operands but leaves ordered device work outstanding.
    void release_with_completion(Completion completion) noexcept
    {
      if (storage_ == nullptr) return;
      CHECK(completion);
      storage_->release_blocking_write_access(std::move(completion));
      storage_ = nullptr;
    }

  private:
    explicit BlockingWriteAccess(CudaBuffer<T>& storage)
    {
      storage.acquire_blocking_write_access();
      storage_ = &storage;
    }

    CudaBuffer<T>* storage_ = nullptr;

    friend class CudaBuffer<T>;
};

/// \brief Read access that owns a moved CUDA buffer.
/// \details The owned buffer is moved before access begins, so the access state
///          needs no back-pointer and remains valid when this object moves.
///          Blocking access waits for the prior writer during construction.
///          Stream-ordered access installs the prior-writer wait and publishes
///          its reader completion on release.
template <typename T> class OwningReadAccess {
  public:
    using element_type = T;

    OwningReadAccess(OwningReadAccess const&) = delete;
    OwningReadAccess& operator=(OwningReadAccess const&) = delete;

    OwningReadAccess(OwningReadAccess&& other) noexcept
        : storage_(std::move(other.storage_)), stream_(std::move(other.stream_)),
          stream_ordered_(std::exchange(other.stream_ordered_, false)), active_(std::exchange(other.active_, false))
    {}

    OwningReadAccess& operator=(OwningReadAccess&&) = delete;

    ~OwningReadAccess() { this->release(); }

    /// \brief Return the typed device pointer owned by this access state.
    [[nodiscard]] T const* data() const noexcept { return active_ ? storage_.data() : nullptr; }

    /// \brief End access and destroy the owned allocation.
    void release() noexcept
    {
      if (!active_) return;
      bool const blocking_deallocation = !stream_ordered_;
      if (stream_ordered_)
      {
        storage_.publish_owning_read_completion(stream_);
      }
      stream_ = {};
      stream_ordered_ = false;
      active_ = false;
      if (blocking_deallocation)
        storage_.reset_owned();
      else
        storage_.reset();
    }

  private:
    explicit OwningReadAccess(CudaBuffer<T>&& storage) : storage_(std::move(storage)), active_(true)
    {
      storage_.allocation_->require_blocking_deallocation();
      storage_.prepare_owning_blocking_read_access();
    }

    OwningReadAccess(CudaBuffer<T>&& storage, Stream const& stream)
        : storage_(std::move(storage)), stream_(stream), stream_ordered_(true), active_(true)
    {
      storage_.prepare_owning_read_access(stream_);
    }

    CudaBuffer<T> storage_;
    Stream stream_;
    bool stream_ordered_ = false;
    bool active_ = false;

    friend class CudaBuffer<T>;
};

template <typename T> OwningReadAccess<T> CudaBuffer<T>::into_blocking_read_access() &&
{
  return OwningReadAccess<T>{std::move(*this)};
}

template <typename T> OwningReadAccess<T> CudaBuffer<T>::into_read_synchronized_with(Stream const& stream) &&
{
  return OwningReadAccess<T>{std::move(*this), stream};
}

} // namespace uni20::cuda
