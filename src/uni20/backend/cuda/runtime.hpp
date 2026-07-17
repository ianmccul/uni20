#pragma once

/**
 * \file runtime.hpp
 * \ingroup backend_cuda
 * \brief CUDA device, stream, event, completion, and idle-stream-pool primitives.
 */

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <optional>

namespace uni20::cuda
{

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

/// \brief Move-only owner of one CUDA stream.
class Stream {
  public:
    /// \brief Create a stream on `device`.
    explicit Stream(int device, unsigned int flags = cudaStreamNonBlocking);
    Stream(Stream const&) = delete;
    Stream& operator=(Stream const&) = delete;
    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) noexcept;
    ~Stream();

    /// \brief Return the device that owns this stream.
    [[nodiscard]] int device() const noexcept { return device_; }

    /// \brief Return the CUDA runtime stream handle.
    [[nodiscard]] cudaStream_t native_handle() const noexcept { return stream_; }

    /// \brief Wait on all work currently queued in this stream.
    void synchronize() const;

  private:
    int device_ = -1;
    cudaStream_t stream_ = nullptr;

    void reset() noexcept;
};

/// \brief Move-only owner of one non-timing CUDA event.
class Event {
  public:
    /// \brief Create a non-timing event on `device`.
    explicit Event(int device);
    Event(Event const&) = delete;
    Event& operator=(Event const&) = delete;
    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;
    ~Event();

    /// \brief Return the device on which this event was created.
    [[nodiscard]] int device() const noexcept { return device_; }

    /// \brief Return the CUDA runtime event handle.
    [[nodiscard]] cudaEvent_t native_handle() const noexcept { return event_; }

    /// \brief Record the event at the current tail of `stream`.
    void record(Stream const& stream) const;

    /// \brief Make `stream` wait for the most recently recorded event generation.
    void wait_on(Stream const& stream) const;

    /// \brief Return whether the recorded event generation has completed.
    [[nodiscard]] bool ready() const;

    /// \brief Wait for the recorded event generation to complete.
    void synchronize() const;

  private:
    int device_ = -1;
    cudaEvent_t event_ = nullptr;

    void reset() noexcept;
};

/// \brief Shared completion token for one submitted CUDA operation.
class Completion {
  public:
    Completion() = default;

    /// \brief Return whether this token refers to a submitted operation.
    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(event_); }

    /// \brief Return the producer device ordinal.
    [[nodiscard]] int device() const noexcept;

    /// \brief Return the underlying CUDA event handle.
    [[nodiscard]] cudaEvent_t native_handle() const noexcept;

    /// \brief Make `stream` wait for this completion.
    void wait_on(Stream const& stream) const;

    /// \brief Return whether the operation has completed.
    [[nodiscard]] bool ready() const;

    /// \brief Wait on the host for the operation to complete.
    void synchronize() const;

  private:
    explicit Completion(Event event);

    std::shared_ptr<Event> event_;

    friend class StreamPool;
};

/// \brief Device-local pool whose streams become available only after their queued work completes.
class StreamPool {
  public:
    class Lease;

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
    [[nodiscard]] std::optional<Lease> try_acquire();

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

    Completion submit(std::size_t slot);
    void release_without_submission(std::size_t slot) noexcept;

    friend class Lease;
};

/// \brief Affine lease of one idle stream slot.
/// \details The lease must be consumed with `submit()` after enqueueing work, or
///          with `release_without_submission()` if no work was enqueued.
class StreamPool::Lease {
  public:
    Lease() = default;
    Lease(Lease const&) = delete;
    Lease& operator=(Lease const&) = delete;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) noexcept;
    ~Lease();

    /// \brief Return whether this lease owns a stream slot.
    [[nodiscard]] explicit operator bool() const noexcept { return pool_ != nullptr; }

    /// \brief Return the leased stream.
    [[nodiscard]] Stream& stream() const noexcept;

    /// \brief Record completion, arrange idle-pool return, and consume this lease.
    [[nodiscard]] Completion submit();

    /// \brief Return an unused stream immediately and consume this lease.
    /// \pre No CUDA work or wait was enqueued into the leased stream.
    void release_without_submission() noexcept;

  private:
    Lease(StreamPool& pool, std::size_t slot) : pool_(&pool), slot_(slot) {}

    StreamPool* pool_ = nullptr;
    std::size_t slot_ = 0;

    friend class StreamPool;
};

} // namespace uni20::cuda
