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
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace uni20::cuda
{

template <typename T = std::byte> class CudaBuffer;
template <typename T> class ReadAccess;
template <typename T> class WriteAccess;

/// \brief Device-local resources shared by CUDA buffers and streams.
/// \details The context must outlive every buffer created from it and every
///          stream handle acquired from its stream pool.
class DeviceContext {
  public:
    /// \brief Configuration for one device-local CUDA context.
    struct Config
    {
        Device device;
        std::size_t stream_count;
        unsigned int stream_flags = cudaStreamNonBlocking;
    };

    /// \brief Create the stream resources for one CUDA device.
    explicit DeviceContext(Config config)
        : device_(config.device), streams_({.device = config.device.ordinal(),
                                            .stream_count = config.stream_count,
                                            .stream_flags = config.stream_flags})
    {}
    DeviceContext(DeviceContext const&) = delete;
    DeviceContext& operator=(DeviceContext const&) = delete;
    DeviceContext(DeviceContext&&) = delete;
    DeviceContext& operator=(DeviceContext&&) = delete;

    /// \brief Return the device served by this context.
    [[nodiscard]] Device device() const noexcept { return device_; }

    /// \brief Return the context's actually-idle stream pool.
    [[nodiscard]] StreamPool& streams() noexcept { return streams_; }
    [[nodiscard]] StreamPool const& streams() const noexcept { return streams_; }

    /// \brief Return the context-owned instance of one provider resource type.
    /// \details The resource is constructed on first use and then retained until
    ///          this context is destroyed. A context owns at most one instance
    ///          of each concrete resource type. Provider resources are destroyed
    ///          before the stream pool that they may reference.
    template <class Resource, class... Args> [[nodiscard]] Resource& provider_resource(Args&&... args)
    {
      std::lock_guard lock(provider_resources_mutex_);
      auto const key = std::type_index(typeid(Resource));
      auto found = provider_resources_.find(key);
      if (found != provider_resources_.end())
      {
        return static_cast<ProviderResourceModel<Resource>&>(*found->second).value;
      }

      auto resource = std::make_unique<ProviderResourceModel<Resource>>(std::forward<Args>(args)...);
      Resource& result = resource->value;
      provider_resources_.emplace(key, std::move(resource));
      return result;
    }

  private:
    struct ProviderResourceBase
    {
        virtual ~ProviderResourceBase() = default;
    };

    template <class Resource> struct ProviderResourceModel final : ProviderResourceBase
    {
        template <class... Args> explicit ProviderResourceModel(Args&&... args) : value(std::forward<Args>(args)...) {}

        Resource value;
    };

    Device device_;
    StreamPool streams_;
    mutable std::mutex state_mutex_;
    std::mutex provider_resources_mutex_;
    std::unordered_map<std::type_index, std::unique_ptr<ProviderResourceBase>> provider_resources_;

    template <typename> friend class CudaBuffer;
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

} // namespace detail

/// \brief Move-only typed CUDA device allocation and completion state.
/// \details Raw device pointers are exposed only through scoped
///          `read_synchronized_with(stream)` and
///          `write_synchronized_with(stream)` access objects.
template <typename T> class CudaBuffer {
  public:
    static_assert(std::is_object_v<T>, "CUDA buffers require an object element type");
    static_assert(!std::is_const_v<T>, "CUDA buffers own mutable storage; constness belongs on read guards");

    using element_type = T;
    using read_access_type = ReadAccess<T>;
    using write_access_type = WriteAccess<T>;

    /// \brief Allocate `size` elements on the context's CUDA device.
    CudaBuffer(DeviceContext& context, std::size_t size) : context_(&context), size_(size)
    {
      std::size_t const bytes = checked_size_bytes(size_);
      if (size_ == 0)
      {
        return;
      }

      if (context.device().capabilities().memory_pools_supported)
      {
        this->allocate_stream_ordered(bytes);
      }
      else
      {
        this->allocate_blocking(bytes);
      }
    }

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

    /// \brief Return the context that owns completion state for this buffer.
    [[nodiscard]] DeviceContext& context() const
    {
      CHECK(context_ != nullptr);
      return *context_;
    }

    /// \brief Return the allocation's CUDA device.
    [[nodiscard]] Device device() const { return this->context().device(); }

    /// \brief Wait for every submitted operation currently involving this buffer.
    void synchronize() const
    {
      Completion writer_completion;
      std::vector<Completion> reader_completions;
      {
        std::lock_guard lock(this->context().state_mutex_);
        CHECK(live_read_accesses_ == 0 && !live_write_access_,
              "cannot synchronize a CUDA buffer while access guards are live", data_, live_read_accesses_,
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

  private:
    [[nodiscard]] static std::size_t checked_size_bytes(std::size_t size)
    {
      CHECK(size <= std::numeric_limits<std::size_t>::max() / sizeof(T), size, sizeof(T));
      return size * sizeof(T);
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] T const* data() const noexcept { return data_; }

    void allocate_blocking(std::size_t bytes)
    {
      void* raw_data = nullptr;
      ScopedDevice guard(context_->device().ordinal());
      check(cudaMalloc(&raw_data, bytes), "cudaMalloc", context_->device().ordinal());
      data_ = static_cast<T*>(raw_data);
    }

    void allocate_stream_ordered(std::size_t bytes)
    {
      auto stream = context_->streams().acquire();
      void* raw_data = nullptr;
      ScopedDevice guard(context_->device().ordinal());
      check(cudaMallocAsync(&raw_data, bytes, stream.native_handle()), "cudaMallocAsync", context_->device().ordinal());
      data_ = static_cast<T*>(raw_data);

      try
      {
        Completion completion = stream.record_completion();
        std::lock_guard lock(context_->state_mutex_);
        writer_completion_ = std::move(completion);
      }
      catch (...)
      {
        detail::synchronize_after_failed_publication(stream, "publish CUDA async allocation completion");
        detail::check_cuda_cleanup(cudaFreeAsync(data_, stream.native_handle()),
                                   "cudaFreeAsync after cudaMallocAsync publication failure",
                                   context_->device().ordinal());
        detail::synchronize_after_failed_publication(stream, "free CUDA async allocation after publication failure");
        data_ = nullptr;
        size_ = 0;
        throw;
      }
    }

    void acquire_read_access(Stream const& stream) const
    {
      CHECK(stream, "cannot synchronize CUDA buffer access with an empty stream", data_);
      Completion writer_completion;
      {
        std::lock_guard lock(this->context().state_mutex_);
        CHECK(!live_write_access_, "cannot acquire CUDA read access while a write access is live", data_);
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
        std::lock_guard lock(this->context().state_mutex_);
        CHECK(live_read_accesses_ > 0);
        --live_read_accesses_;
        throw;
      }
    }

    void acquire_write_access(Stream const& stream) const
    {
      CHECK(stream, "cannot synchronize CUDA buffer access with an empty stream", data_);
      Completion writer_completion;
      std::vector<Completion> reader_completions;
      {
        std::lock_guard lock(this->context().state_mutex_);
        CHECK(!live_write_access_ && live_read_accesses_ == 0,
              "cannot acquire CUDA write access while another access is live", data_, live_read_accesses_,
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
        std::lock_guard lock(this->context().state_mutex_);
        CHECK(live_write_access_);
        live_write_access_ = false;
        throw;
      }
    }

    void release_read_access(Stream const& stream) const noexcept
    {
      try
      {
        Completion completion = stream.record_completion();
        std::lock_guard lock(this->context().state_mutex_);
        CHECK(live_read_accesses_ > 0);
        if (reader_completions_.size() == reader_completions_.capacity())
        {
          std::erase_if(reader_completions_,
                        [](Completion const& reader_completion) { return reader_completion.ready(); });
        }
        reader_completions_.push_back(std::move(completion));
        --live_read_accesses_;
        return;
      }
      catch (...)
      {
        detail::synchronize_after_failed_publication(stream, "publish CUDA buffer reader completion");
      }

      std::lock_guard lock(this->context().state_mutex_);
      CHECK(live_read_accesses_ > 0);
      --live_read_accesses_;
    }

    void release_write_access(Stream const& stream) const noexcept
    {
      try
      {
        Completion completion = stream.record_completion();
        std::lock_guard lock(this->context().state_mutex_);
        CHECK(live_write_access_);
        writer_completion_ = std::move(completion);
        reader_completions_.clear();
        live_write_access_ = false;
        return;
      }
      catch (...)
      {
        detail::synchronize_after_failed_publication(stream, "publish CUDA buffer writer completion");
      }

      std::lock_guard lock(this->context().state_mutex_);
      CHECK(live_write_access_);
      live_write_access_ = false;
    }

    void reset() noexcept
    {
      if (context_ == nullptr)
      {
        return;
      }

      Completion writer_completion;
      std::vector<Completion> reader_completions;
      {
        std::lock_guard lock(context_->state_mutex_);
        CHECK(live_read_accesses_ == 0 && !live_write_access_,
              "cannot destroy or reset a CUDA buffer while access guards are live", data_, live_read_accesses_,
              live_write_access_);
        writer_completion = std::move(writer_completion_);
        reader_completions = std::move(reader_completions_);
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
        PANIC("CUDA buffer completion synchronization failed during cleanup", context_->device().ordinal());
      }

      if (data_ != nullptr)
      {
        this->free_allocation();
      }

      context_ = nullptr;
      data_ = nullptr;
      size_ = 0;
    }

    void move_from(CudaBuffer& other) noexcept
    {
      if (other.context_ == nullptr)
      {
        return;
      }

      std::lock_guard lock(other.context_->state_mutex_);
      CHECK(other.live_read_accesses_ == 0 && !other.live_write_access_,
            "cannot move a CUDA buffer while access guards are live", other.data_, other.live_read_accesses_,
            other.live_write_access_);
      context_ = other.context_;
      data_ = other.data_;
      size_ = other.size_;
      writer_completion_ = std::move(other.writer_completion_);
      reader_completions_ = std::move(other.reader_completions_);
      live_read_accesses_ = 0;
      live_write_access_ = false;
      other.context_ = nullptr;
      other.data_ = nullptr;
      other.size_ = 0;
    }

    void free_allocation() noexcept
    {
      int const device = context_->device().ordinal();
      detail::BufferCleanupDeviceGuard guard(device);
      if (!context_->device().capabilities().memory_pools_supported)
      {
        detail::check_cuda_cleanup(cudaFree(data_), "cudaFree", device);
        return;
      }

      try
      {
        auto stream = context_->streams().acquire();
        detail::check_cuda_cleanup(cudaFreeAsync(data_, stream.native_handle()), "cudaFreeAsync", device);
        stream.synchronize();
      }
      catch (...)
      {
        PANIC("CUDA stream-ordered buffer cleanup failed", device);
      }
    }

    DeviceContext* context_ = nullptr;
    T* data_ = nullptr;
    std::size_t size_ = 0;
    mutable Completion writer_completion_;
    mutable std::vector<Completion> reader_completions_;
    mutable std::size_t live_read_accesses_ = 0;
    mutable bool live_write_access_ = false;

    template <typename> friend class ReadAccess;
    template <typename> friend class WriteAccess;
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

} // namespace uni20::cuda
