#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensor
{

class CudaDeviceContext {
  public:
    struct EventDependency
    {
        CudaDeviceContext* context = nullptr;
        cudaEvent_t event = nullptr;
        std::uint64_t sequence = 0;

        EventDependency(CudaDeviceContext& context, cudaEvent_t event, std::uint64_t sequence)
            : context(&context), event(event), sequence(sequence)
        {}
        EventDependency(EventDependency const&) = delete;
        EventDependency& operator=(EventDependency const&) = delete;
        ~EventDependency();
    };

    using EventDependencyRef = std::shared_ptr<EventDependency>;

    struct ScratchBuffer
    {
        void* ptr = nullptr;
        std::size_t bytes = 0;
        cudaEvent_t readyEvent = nullptr;
        cudaStream_t readyStream = nullptr;
        bool readyEventRecorded = false;
    };

    // Small leased device buffers for scalar scratch and future block-wise
    // reductions.  The cache permits multiple in-flight users and protects
    // reuse across streams with the buffer-local ready event.
    class ScratchLease {
      public:
        ScratchLease() = default;
        ScratchLease(CudaDeviceContext& context, std::shared_ptr<ScratchBuffer> buffer, cudaStream_t stream)
            : context_(&context), buffer_(std::move(buffer)), stream_(stream)
        {}
        ScratchLease(ScratchLease const&) = delete;
        ScratchLease& operator=(ScratchLease const&) = delete;
        ScratchLease(ScratchLease&& other) noexcept;
        ScratchLease& operator=(ScratchLease&& other) noexcept;
        ~ScratchLease();

        void* get() const noexcept { return buffer_ == nullptr ? nullptr : buffer_->ptr; }
        template <typename T> T* as() const noexcept { return static_cast<T*>(get()); }

      private:
        CudaDeviceContext* context_ = nullptr;
        std::shared_ptr<ScratchBuffer> buffer_;
        cudaStream_t stream_ = nullptr;

        void release();
    };

    class GpuEvent;
    using GpuEventRef = std::shared_ptr<GpuEvent>;

    class StreamLease {
      public:
        StreamLease() = default;
        StreamLease(StreamLease const&) = delete;
        StreamLease& operator=(StreamLease const&) = delete;
        StreamLease(StreamLease&& other) noexcept;
        StreamLease& operator=(StreamLease&& other) noexcept;
        ~StreamLease();

        cudaStream_t stream() const noexcept { return stream_; }
        explicit operator bool() const noexcept { return stream_ != nullptr; }
        void release();

      private:
        StreamLease(CudaDeviceContext& context, cudaStream_t stream) : context_(&context), stream_(stream) {}

        CudaDeviceContext* context_ = nullptr;
        cudaStream_t stream_ = nullptr;

        friend class CudaDeviceContext;
    };

    class GpuEvent {
      public:
        GpuEvent() = default;
        GpuEvent(CudaDeviceContext& context, cudaStream_t producerStream);
        GpuEvent(GpuEvent const&) = delete;
        GpuEvent& operator=(GpuEvent const&) = delete;
        GpuEvent(GpuEvent&& other) noexcept;
        GpuEvent& operator=(GpuEvent&& other) noexcept;
        ~GpuEvent() = default;

        cudaStream_t stream() const noexcept;
        std::uint64_t sequence() const noexcept { return publishSequence_; }
        void waitOn(cudaStream_t consumerStream);
        EventDependencyRef dependencyEvent();

      private:
        CudaDeviceContext* context_ = nullptr;
        EventDependencyRef event_;
        cudaStream_t producerStream_ = nullptr;
        std::uint64_t publishSequence_ = 0;
    };

    CudaDeviceContext(int deviceId, int workStreamCount, bool serialCuda);
    CudaDeviceContext(CudaDeviceContext const&) = delete;
    CudaDeviceContext& operator=(CudaDeviceContext const&) = delete;
    ~CudaDeviceContext();

    static std::shared_ptr<CudaDeviceContext> shared(int deviceId, int workStreamCount, bool serialCuda);

    int deviceId() const noexcept { return deviceId_; }
    bool serialCuda() const noexcept { return serialCuda_; }
    cudaStream_t memoryStream() const noexcept { return memoryStream_; }

    StreamLease leaseWorkStream(cudaStream_t preferredStream = nullptr);
    GpuEventRef recordCompletionEvent(cudaStream_t producerStream);
    cudaEvent_t acquireEvent();
    void retireEvent(cudaEvent_t event);
    cudaEvent_t recordEvent(cudaStream_t stream);
    EventDependencyRef recordDependencyEvent(cudaStream_t stream);
    void waitEvent(cudaStream_t stream, cudaEvent_t event);
    void enqueueAsyncFree(void* ptr, cudaStream_t stream, std::vector<EventDependencyRef> dependencies);
    ScratchLease acquireScratch(std::size_t bytes, cudaStream_t stream);
    cublasHandle_t cublasHandleForCurrentThread(cudaStream_t stream);
    void syncWorkStreams(const char* reason = "work_unspecified");
    void syncMemoryStream(const char* reason = "memory_unspecified");
    void release();

  private:
    struct Counters
    {
        std::uint64_t eventCreate = 0;
        std::uint64_t eventRecord = 0;
        std::uint64_t eventWait = 0;
        std::uint64_t eventDestroy = 0;
        std::uint64_t streamSync = 0;
        std::uint64_t asyncFree = 0;
        std::uint64_t asyncFreeReclaim = 0;
        std::uint64_t asyncFreePoll = 0;
        std::unordered_map<std::string, std::uint64_t> streamSyncByReason;
    };

    struct PendingFree
    {
        void* ptr = nullptr;
        cudaEvent_t completeEvent = nullptr;
        std::vector<EventDependencyRef> dependencies;
    };

    int deviceId_ = 0;
    bool serialCuda_ = false;
    bool logCounters_ = false;
    bool released_ = false;
    cudaStream_t memoryStream_ = nullptr;
    std::deque<cudaStream_t> availableWorkStreams_;
    std::size_t createdWorkStreamCount_ = 0;
    std::vector<cudaEvent_t> freeEvents_;
    std::vector<cudaEvent_t> retiredEvents_;
    std::vector<PendingFree> pendingFrees_;
    std::vector<std::shared_ptr<ScratchBuffer>> freeScratchBuffers_;
    std::vector<std::shared_ptr<ScratchBuffer>> allScratchBuffers_;
    std::mutex scratchMutex_;
    Counters counters_;
    std::uint64_t dependencySequence_ = 0;

    void reclaimRetiredEvents();
    void reclaimCompletedAsyncFrees();
    void countStreamSync(const char* reason);
    cudaStream_t acquireWorkStream(cudaStream_t preferredStream = nullptr);
    void returnWorkStream(cudaStream_t stream);
    void releaseScratch(std::shared_ptr<ScratchBuffer> buffer, cudaStream_t stream);
    void printCounters() const;

    friend class ScratchLease;
};

} // namespace tensor
