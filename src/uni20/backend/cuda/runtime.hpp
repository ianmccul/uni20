#pragma once

/**
 * \file runtime.hpp
 * \ingroup backend_cuda
 * \brief CUDA device, stream, completion, and idle-stream-pool primitives.
 */

#include <uni20/common/trace.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace uni20::cuda
{

class Completion;
class StreamPool;

namespace detail
{
class StreamWaiter;
}

/// \brief Temporarily select one CUDA device and restore the previous device on destruction.
class ScopedDevice {
  public:
    /// \brief Select `device` for the lifetime of this guard.
    explicit ScopedDevice(int device);
    ScopedDevice(ScopedDevice const&) = delete;
    ScopedDevice& operator=(ScopedDevice const&) = delete;
    ~ScopedDevice();

    /// \brief Return the selected device ordinal.
    [[nodiscard]] int device() const noexcept { return device_; }

  private:
    int device_;
    int previous_device_;
    bool restore_;
};

/// \brief Reference-counted lease of one CUDA stream-pool slot.
/// \details Streams are acquired from `StreamPool`. Copies share the same
///          lease; the stream is returned to the pool only after the last
///          reference is destroyed and all queued CUDA work has completed.
class Stream {
  public:
    class State;

    Stream() = default;
    Stream(Stream const&) noexcept = default;
    Stream& operator=(Stream const&) noexcept = default;
    Stream(Stream&& other) noexcept = default;
    Stream& operator=(Stream&& other) noexcept = default;
    ~Stream() = default;

    /// \brief Return whether this handle refers to a leased CUDA stream.
    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(state_); }

    /// \brief Return the device that owns this stream.
    [[nodiscard]] int device() const noexcept;

    /// \brief Return the CUDA runtime stream handle.
    [[nodiscard]] cudaStream_t native_handle() const noexcept;

    /// \brief Wait on all work currently queued in this stream.
    void synchronize() const;

    /// \brief Record and return an immutable completion for the current stream tail.
    [[nodiscard]] Completion record_completion() const;

    /// \brief Make this stream wait for `completion` before executing subsequent work.
    /// \details The producer completion may belong to a different CUDA device.
    void wait_on(Completion const& completion) const;

  private:
    explicit Stream(std::shared_ptr<State> state) noexcept;

    std::shared_ptr<State> state_;

    friend class StreamPool;
};

/// \brief Shared completion token for one submitted CUDA operation.
class Completion {
  public:
    Completion() = default;

    /// \brief Return whether this token refers to a submitted operation.
    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(state_); }

    /// \brief Return the producer device ordinal.
    [[nodiscard]] int device() const noexcept;

    /// \brief Return whether the operation has completed.
    [[nodiscard]] bool ready() const;

    /// \brief Wait on the host for the operation to complete.
    void synchronize() const;

  private:
    class State;

    explicit Completion(std::shared_ptr<State> state);

    std::shared_ptr<State> state_;

    friend class Stream;
    friend class Stream::State;
};

namespace detail
{

/// \brief Intrusive callback node used by non-blocking stream acquisition.
class StreamWaiter {
  public:
    using notify_type = void (*)(StreamWaiter&, Stream) noexcept;

    explicit StreamWaiter(notify_type notify) noexcept : notify_(notify) { CHECK(notify_ != nullptr); }
    StreamWaiter(StreamWaiter const&) = delete;
    StreamWaiter& operator=(StreamWaiter const&) = delete;

  private:
    void notify(Stream stream) noexcept { notify_(*this, std::move(stream)); }

    notify_type notify_;
    StreamWaiter* next_ = nullptr;
    bool queued_ = false;

    friend class ::uni20::cuda::StreamPool;
};

} // namespace detail

/// \brief Device-local pool whose streams become available only after their queued work completes.
/// \details The pool must outlive every `Stream` handle acquired from it.
class StreamPool {
  public:
    /// \brief Configuration for a device-local idle-stream pool.
    struct Config
    {
        int device;
        std::size_t stream_count;
        unsigned int stream_flags = cudaStreamNonBlocking;
    };

    /// \brief Create `stream_count` streams for one CUDA device.
    explicit StreamPool(Config config);
    StreamPool(StreamPool const&) = delete;
    StreamPool& operator=(StreamPool const&) = delete;
    ~StreamPool();

    /// \brief Try to lease one actually idle stream without waiting.
    [[nodiscard]] std::optional<Stream> try_acquire();

    /// \brief Block until one actually idle stream can be leased.
    /// \details Pool-return CUDA callbacks notify blocked callers directly;
    ///          no scheduler participation is required.
    [[nodiscard]] Stream acquire();

    /// \brief Register an internal awaiter unless a stream is immediately available.
    [[nodiscard]] std::optional<Stream> acquire_or_enqueue(detail::StreamWaiter& waiter) noexcept;

    /// \brief Return the device served by this pool.
    [[nodiscard]] int device() const noexcept;

    /// \brief Return the total number of stream slots.
    [[nodiscard]] std::size_t size() const noexcept;

    /// \brief Return the number of streams whose prior work has completed.
    [[nodiscard]] std::size_t idle_stream_count() const noexcept;

    /// \brief Return the number of streams currently held by submitters.
    [[nodiscard]] std::size_t leased_stream_count() const noexcept;

    /// \brief Return the number of streams awaiting device completion.
    [[nodiscard]] std::size_t pending_stream_count() const noexcept;

    /// \brief Synchronize every stream and process all queued pool-return callbacks.
    void synchronize();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    friend class Stream::State;
};

} // namespace uni20::cuda
