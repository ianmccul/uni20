#pragma once

/**
 * \file buffer.hpp
 * \ingroup backend_cuda
 * \brief Typed CUDA device buffers and scoped stream access guards.
 */

#include <uni20/backend/cuda/device.hpp>
#include <uni20/backend/cuda/runtime.hpp>

#include <cstddef>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::cuda
{

class BufferStorage;
template <typename T = std::byte>
class Buffer;
template <typename T>
class ReadBuffer;
template <typename T>
class WriteBuffer;

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
    explicit DeviceContext(Config config);
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

    friend class BufferStorage;
};

/// \brief Untyped storage and completion state behind typed CUDA buffers.
class BufferStorage {
  public:
    /// \brief Allocate `size_bytes` bytes on the context's CUDA device.
    BufferStorage(DeviceContext& context, std::size_t size_bytes);
    BufferStorage(BufferStorage const&) = delete;
    BufferStorage& operator=(BufferStorage const&) = delete;
    BufferStorage(BufferStorage&& other) noexcept;
    BufferStorage& operator=(BufferStorage&& other) noexcept;
    ~BufferStorage();

    /// \brief Return the allocation size in bytes.
    [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes_; }

    /// \brief Return whether this is a zero-sized or moved-from allocation.
    [[nodiscard]] bool empty() const noexcept { return size_bytes_ == 0; }

    /// \brief Return the context that owns completion state for this buffer.
    [[nodiscard]] DeviceContext& context() const;

    /// \brief Return the allocation's CUDA device.
    [[nodiscard]] Device device() const;

    /// \brief Wait for every submitted operation currently involving this buffer.
    void synchronize() const;

  private:
    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] void const* data() const noexcept { return data_; }

    void install_read_waits(Stream const& stream) const;
    void install_write_waits(Stream const& stream) const;
    void publish_read_after(Stream const& stream) const noexcept;
    void publish_write_after(Stream const& stream) const noexcept;
    void publish_reader(Completion const& completion) const;
    void publish_writer(Completion const& completion) const;
    void reset() noexcept;

    DeviceContext* context_ = nullptr;
    void* data_ = nullptr;
    std::size_t size_bytes_ = 0;
    mutable Completion writer_completion_;
    mutable std::vector<Completion> reader_completions_;

    template <typename>
    friend class Buffer;
    template <typename>
    friend class ReadBuffer;
    template <typename>
    friend class WriteBuffer;
};

/// \brief Scoped read-only access to a CUDA buffer on one stream.
/// \details The constructor installs predecessor waits on the stream. The
///          destructor records a reader completion at the current stream tail
///          and publishes it back to the buffer.
template <typename T>
class ReadBuffer {
  public:
    using element_type = T;

    ReadBuffer(ReadBuffer const&) = delete;
    ReadBuffer& operator=(ReadBuffer const&) = delete;
    ReadBuffer(ReadBuffer&& other) noexcept
        : storage_(other.storage_), stream_(std::move(other.stream_)), data_(other.data_), size_(other.size_),
          active_(std::exchange(other.active_, false))
    {
      other.storage_ = nullptr;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    ReadBuffer& operator=(ReadBuffer&&) = delete;
    ~ReadBuffer()
    {
      if (active_)
      {
        storage_->publish_read_after(stream_);
      }
    }

    /// \brief Return the typed device pointer for read-only CUDA work.
    [[nodiscard]] T const* data() const noexcept { return data_; }

    /// \brief Return the number of typed elements in the view.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// \brief Return the number of bytes in the view.
    [[nodiscard]] std::size_t size_bytes() const noexcept { return size_ * sizeof(T); }

  private:
    ReadBuffer(Buffer<T> const& buffer, Stream const& stream)
        : storage_(&buffer.storage_), stream_(stream), data_(static_cast<T const*>(buffer.storage_.data())),
          size_(buffer.size_), active_(true)
    {
      storage_->install_read_waits(stream_);
    }

    BufferStorage const* storage_;
    Stream stream_;
    T const* data_;
    std::size_t size_;
    bool active_;

    friend class Buffer<T>;
};

/// \brief Scoped read/write access to a CUDA buffer on one stream.
/// \details The constructor installs predecessor waits on the stream. The
///          destructor records an exclusive-writer completion at the current
///          stream tail and publishes it back to the buffer.
template <typename T>
class WriteBuffer {
  public:
    using element_type = T;

    WriteBuffer(WriteBuffer const&) = delete;
    WriteBuffer& operator=(WriteBuffer const&) = delete;
    WriteBuffer(WriteBuffer&& other) noexcept
        : storage_(other.storage_), stream_(std::move(other.stream_)), data_(other.data_), size_(other.size_),
          active_(std::exchange(other.active_, false))
    {
      other.storage_ = nullptr;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    WriteBuffer& operator=(WriteBuffer&&) = delete;
    ~WriteBuffer()
    {
      if (active_)
      {
        storage_->publish_write_after(stream_);
      }
    }

    /// \brief Return the typed device pointer for mutating CUDA work.
    [[nodiscard]] T* data() const noexcept { return data_; }

    /// \brief Return the number of typed elements in the view.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// \brief Return the number of bytes in the view.
    [[nodiscard]] std::size_t size_bytes() const noexcept { return size_ * sizeof(T); }

  private:
    WriteBuffer(Buffer<T>& buffer, Stream const& stream)
        : storage_(&buffer.storage_), stream_(stream), data_(static_cast<T*>(buffer.storage_.data())),
          size_(buffer.size_), active_(true)
    {
      storage_->install_write_waits(stream_);
    }

    BufferStorage* storage_;
    Stream stream_;
    T* data_;
    std::size_t size_;
    bool active_;

    friend class Buffer<T>;
};

/// \brief Move-only typed CUDA device allocation.
/// \details Raw device pointers are exposed only through scoped `read(stream)`
///          and `write(stream)` guards.
template <typename T>
class Buffer {
  public:
    static_assert(std::is_object_v<T>, "CUDA buffers require an object element type");
    static_assert(!std::is_const_v<T>, "CUDA buffers own mutable storage; constness belongs on read guards");

    using element_type = T;
    using read_buffer_type = ReadBuffer<T>;
    using write_buffer_type = WriteBuffer<T>;

    /// \brief Allocate `size` elements on the context's CUDA device.
    Buffer(DeviceContext& context, std::size_t size) : storage_(context, size * sizeof(T)), size_(size) {}
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;
    Buffer(Buffer&& other) noexcept : storage_(std::move(other.storage_)), size_(std::exchange(other.size_, 0)) {}
    Buffer& operator=(Buffer&& other) noexcept
    {
      if (this != &other)
      {
        storage_ = std::move(other.storage_);
        size_ = std::exchange(other.size_, 0);
      }
      return *this;
    }
    ~Buffer() = default;

    /// \brief Return the number of typed elements in the allocation.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// \brief Return the allocation size in bytes.
    [[nodiscard]] std::size_t size_bytes() const noexcept { return storage_.size_bytes(); }

    /// \brief Return whether this is a zero-sized or moved-from allocation.
    [[nodiscard]] bool empty() const noexcept { return storage_.empty(); }

    /// \brief Return the context that owns completion state for this buffer.
    [[nodiscard]] DeviceContext& context() const { return storage_.context(); }

    /// \brief Return the allocation's CUDA device.
    [[nodiscard]] Device device() const { return storage_.device(); }

    /// \brief Wait for every submitted operation currently involving this buffer.
    void synchronize() const { storage_.synchronize(); }

    /// \brief Acquire scoped read-only access on `stream`.
    [[nodiscard]] ReadBuffer<T> read(Stream const& stream) const { return ReadBuffer<T>(*this, stream); }

    /// \brief Acquire scoped read/write access on `stream`.
    [[nodiscard]] WriteBuffer<T> write(Stream const& stream) { return WriteBuffer<T>(*this, stream); }

  private:
    BufferStorage storage_;
    std::size_t size_;

    friend class ReadBuffer<T>;
    friend class WriteBuffer<T>;
};

} // namespace uni20::cuda
