#pragma once

/**
 * \file runtime.hpp
 * \ingroup backend_cuda
 * \brief CUDA device, stream, resource, and scoped runtime primitives.
 */

#include <uni20/backend/cuda/device.hpp>
#include <uni20/common/trace.hpp>

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace uni20::cuda
{

class Completion;
class DeviceResources;
class StreamPool;
template <typename T> class CudaBuffer;

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

    /// \brief Record a completion at the current tail of a native CUDA stream.
    /// \param device Device ordinal that owns `stream`.
    /// \param stream Native CUDA stream, including the default stream.
    [[nodiscard]] static Completion record(int device, cudaStream_t stream);

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

/// \brief Mutable stream and provider resources associated with one CUDA device.
/// \details Ordinary application code obtains the canonical instance from the
///          installed CUDA runtime. Direct construction remains useful for
///          focused tests and low-level bring-up. The resources object must
///          outlive every buffer, stream lease, and provider lease created from
///          it.
class DeviceResources {
  public:
    /// \brief Configuration for one device's mutable CUDA resources.
    struct Config
    {
        Device device;
        std::size_t stream_count;
        unsigned int stream_flags = cudaStreamNonBlocking;
    };

    /// \brief Create isolated resources for one CUDA device.
    explicit DeviceResources(Config config);
    DeviceResources(DeviceResources const&) = delete;
    DeviceResources& operator=(DeviceResources const&) = delete;
    DeviceResources(DeviceResources&&) = delete;
    DeviceResources& operator=(DeviceResources&&) = delete;
    ~DeviceResources() noexcept;

    /// \brief Return the device served by this resource set.
    [[nodiscard]] Device device() const noexcept { return device_; }

    /// \brief Return the device's actually-idle stream pool.
    [[nodiscard]] StreamPool& streams() noexcept { return streams_; }
    [[nodiscard]] StreamPool const& streams() const noexcept { return streams_; }

    /// \brief Return the resource-owned instance of one provider resource type.
    /// \details The resource is constructed on first use and retained until
    ///          shutdown. A device owns at most one instance of each concrete
    ///          resource type. Provider resources are destroyed before the
    ///          stream pool that they may reference.
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

    /// \brief Return the number of live CUDA buffers borrowing these resources.
    [[nodiscard]] std::size_t live_buffer_count() const noexcept
    {
      return live_buffer_count_.load(std::memory_order_acquire);
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

    void register_buffer() noexcept { live_buffer_count_.fetch_add(1, std::memory_order_relaxed); }
    void unregister_buffer() noexcept;
    void destroy_provider_resources() noexcept;

    Device device_;
    StreamPool streams_;
    std::mutex provider_resources_mutex_;
    std::unordered_map<std::type_index, std::unique_ptr<ProviderResourceBase>> provider_resources_;
    std::atomic<std::size_t> live_buffer_count_ = 0;

    template <typename> friend class CudaBuffer;
};

/// \brief Configuration for one scoped process-wide CUDA runtime installation.
struct RuntimeConfig
{
    /// CUDA device ordinals to enroll; an empty list enrolls every visible device.
    std::vector<int> device_ordinals{};
    /// Device used by lookup without an explicit ordinal; defaults to the first enrolled device.
    std::optional<int> default_device{};
    /// Number of actually-idle streams owned by each enrolled device.
    std::size_t streams_per_device = 8;
    /// CUDA stream creation flags used by every enrolled device.
    unsigned int stream_flags = cudaStreamNonBlocking;
};

/// \brief Scoped owner of the process-wide CUDA resource installation.
/// \details Construction installs globally discoverable per-device resources.
///          Destruction first uninstalls lookup, then drains and destroys those
///          resources. The owner must outlive every CUDA Tensor, buffer, task,
///          stream, and provider-resource lease.
class Runtime {
  public:
    Runtime(Runtime const&) = delete;
    Runtime& operator=(Runtime const&) = delete;
    Runtime(Runtime&&) = delete;
    Runtime& operator=(Runtime&&) = delete;
    ~Runtime() noexcept;

    /// \brief Return whether this runtime enrolled a CUDA device ordinal.
    [[nodiscard]] bool has_device(int device) const noexcept;

    /// \brief Return the canonical resources for an enrolled device.
    [[nodiscard]] DeviceResources& device_resources(int device) const;

    /// \brief Return the canonical resources for the configured default device.
    [[nodiscard]] DeviceResources& default_device_resources() const;

    /// \brief Return the configured default CUDA device ordinal, if any.
    [[nodiscard]] std::optional<int> default_device() const noexcept { return default_device_; }

  private:
    struct construction_key
    {};

    explicit Runtime(RuntimeConfig config, construction_key);

    std::vector<std::unique_ptr<DeviceResources>> device_resources_;
    std::optional<int> default_device_;

    friend Runtime initialize(RuntimeConfig);
};

/// \brief Install and return a scoped process-wide CUDA runtime owner.
/// \details Only one installation may be active. Retain the returned object at
///          the application startup boundary for as long as CUDA objects exist.
[[nodiscard]] Runtime initialize(RuntimeConfig config = {});

/// \brief Return whether a process-wide CUDA runtime is currently installed.
[[nodiscard]] bool is_initialized() noexcept;

/// \brief Return the installed runtime or fail when CUDA was not initialized.
[[nodiscard]] Runtime& runtime();

/// \brief Return canonical resources for an enrolled CUDA device.
[[nodiscard]] DeviceResources& device_resources(int device);

/// \brief Return canonical resources for the installed runtime's default device.
[[nodiscard]] DeviceResources& device_resources();

} // namespace uni20::cuda
