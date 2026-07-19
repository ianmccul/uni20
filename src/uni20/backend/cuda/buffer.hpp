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
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::cuda
{

template <typename T = std::byte> class CudaBuffer;
template <typename T> class ReadBuffer;
template <typename T> class WriteBuffer;

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

  private:
    Device device_;
    StreamPool streams_;
    mutable std::mutex state_mutex_;

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
/// \details Raw device pointers are exposed only through scoped `read(stream)`
///          and `write(stream)` guards.
template <typename T> class CudaBuffer {
  public:
    static_assert(std::is_object_v<T>, "CUDA buffers require an object element type");
    static_assert(!std::is_const_v<T>, "CUDA buffers own mutable storage; constness belongs on read guards");

    using element_type = T;
    using read_buffer_type = ReadBuffer<T>;
    using write_buffer_type = WriteBuffer<T>;

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

    CudaBuffer(CudaBuffer&& other) noexcept
        : context_(other.context_), data_(other.data_), size_(other.size_),
          writer_completion_(std::move(other.writer_completion_)),
          reader_completions_(std::move(other.reader_completions_))
    {
      other.context_ = nullptr;
      other.data_ = nullptr;
      other.size_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept
    {
      if (this != &other)
      {
        this->reset();
        context_ = other.context_;
        data_ = other.data_;
        size_ = other.size_;
        writer_completion_ = std::move(other.writer_completion_);
        reader_completions_ = std::move(other.reader_completions_);
        other.context_ = nullptr;
        other.data_ = nullptr;
        other.size_ = 0;
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

    /// \brief Acquire scoped read-only access on `stream`.
    [[nodiscard]] ReadBuffer<T> read(Stream const& stream) const { return ReadBuffer<T>(*this, stream); }

    /// \brief Acquire scoped read/write access on `stream`.
    [[nodiscard]] WriteBuffer<T> write(Stream const& stream) { return WriteBuffer<T>(*this, stream); }

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
        this->publish_writer(stream.record_completion());
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

    void install_read_waits(Stream const& stream) const
    {
      CHECK_EQUAL(stream.device(), this->device().ordinal());

      Completion writer_completion;
      {
        std::lock_guard lock(this->context().state_mutex_);
        writer_completion = writer_completion_;
      }

      if (writer_completion)
      {
        stream.wait_on(writer_completion);
      }
    }

    void install_write_waits(Stream const& stream) const
    {
      CHECK_EQUAL(stream.device(), this->device().ordinal());

      Completion writer_completion;
      std::vector<Completion> reader_completions;
      {
        std::lock_guard lock(this->context().state_mutex_);
        writer_completion = writer_completion_;
        std::erase_if(reader_completions_, [](Completion const& completion) { return completion.ready(); });
        reader_completions = reader_completions_;
      }

      if (writer_completion)
      {
        stream.wait_on(writer_completion);
      }
      for (Completion const& reader_completion : reader_completions)
      {
        stream.wait_on(reader_completion);
      }
    }

    void publish_read_after(Stream const& stream) const noexcept
    {
      try
      {
        this->publish_reader(stream.record_completion());
      }
      catch (...)
      {
        detail::synchronize_after_failed_publication(stream, "publish CUDA buffer reader completion");
      }
    }

    void publish_write_after(Stream const& stream) const noexcept
    {
      try
      {
        this->publish_writer(stream.record_completion());
      }
      catch (...)
      {
        detail::synchronize_after_failed_publication(stream, "publish CUDA buffer writer completion");
      }
    }

    void publish_reader(Completion const& completion) const
    {
      std::lock_guard lock(this->context().state_mutex_);
      if (reader_completions_.size() == reader_completions_.capacity())
      {
        std::erase_if(reader_completions_,
                      [](Completion const& reader_completion) { return reader_completion.ready(); });
      }
      reader_completions_.push_back(completion);
    }

    void publish_writer(Completion const& completion) const
    {
      std::lock_guard lock(this->context().state_mutex_);
      writer_completion_ = completion;
      reader_completions_.clear();
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

    template <typename> friend class ReadBuffer;
    template <typename> friend class WriteBuffer;
};

/// \brief Scoped read-only access to a CUDA buffer on one stream.
/// \details The constructor installs predecessor waits on the stream. The
///          destructor records a reader completion at the current stream tail
///          and publishes it back to the buffer.
template <typename T> class ReadBuffer {
  public:
    using element_type = T;

    ReadBuffer(ReadBuffer const&) = delete;
    ReadBuffer& operator=(ReadBuffer const&) = delete;
    ReadBuffer(ReadBuffer&& other) noexcept
        : storage_(std::exchange(other.storage_, nullptr)), stream_(std::move(other.stream_))
    {}
    ReadBuffer& operator=(ReadBuffer&&) = delete;
    ~ReadBuffer()
    {
      if (storage_ != nullptr)
      {
        storage_->publish_read_after(stream_);
      }
    }

    /// \brief Return the typed device pointer for read-only CUDA work.
    [[nodiscard]] T const* data() const noexcept { return storage_ == nullptr ? nullptr : storage_->data(); }

    /// \brief Return the number of typed elements in the view.
    [[nodiscard]] std::size_t size() const noexcept { return storage_ == nullptr ? 0 : storage_->size(); }

    /// \brief Return the number of bytes in the view.
    [[nodiscard]] std::size_t size_bytes() const noexcept { return storage_ == nullptr ? 0 : storage_->size_bytes(); }

  private:
    ReadBuffer(CudaBuffer<T> const& storage, Stream const& stream) : storage_(&storage), stream_(stream)
    {
      storage_->install_read_waits(stream_);
    }

    CudaBuffer<T> const* storage_;
    Stream stream_;

    friend class CudaBuffer<T>;
};

/// \brief Scoped read/write access to a CUDA buffer on one stream.
/// \details The constructor installs predecessor waits on the stream. The
///          destructor records an exclusive-writer completion at the current
///          stream tail and publishes it back to the buffer.
template <typename T> class WriteBuffer {
  public:
    using element_type = T;

    WriteBuffer(WriteBuffer const&) = delete;
    WriteBuffer& operator=(WriteBuffer const&) = delete;
    WriteBuffer(WriteBuffer&& other) noexcept
        : storage_(std::exchange(other.storage_, nullptr)), stream_(std::move(other.stream_))
    {}
    WriteBuffer& operator=(WriteBuffer&&) = delete;
    ~WriteBuffer()
    {
      if (storage_ != nullptr)
      {
        storage_->publish_write_after(stream_);
      }
    }

    /// \brief Return the typed device pointer for mutating CUDA work.
    [[nodiscard]] T* data() const noexcept { return storage_ == nullptr ? nullptr : storage_->data(); }

    /// \brief Return the number of typed elements in the view.
    [[nodiscard]] std::size_t size() const noexcept { return storage_ == nullptr ? 0 : storage_->size(); }

    /// \brief Return the number of bytes in the view.
    [[nodiscard]] std::size_t size_bytes() const noexcept { return storage_ == nullptr ? 0 : storage_->size_bytes(); }

  private:
    WriteBuffer(CudaBuffer<T>& storage, Stream const& stream) : storage_(&storage), stream_(stream)
    {
      storage_->install_write_waits(stream_);
    }

    CudaBuffer<T>* storage_;
    Stream stream_;

    friend class CudaBuffer<T>;
};

} // namespace uni20::cuda
