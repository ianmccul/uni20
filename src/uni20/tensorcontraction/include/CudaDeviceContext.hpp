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

namespace cuda
{
class Stream;
class CublasStream;
class Completion;
using CompletionRef = std::shared_ptr<Completion>;
} // namespace cuda

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

    struct DeviceAllocation
    {
        void* ptr = nullptr;
        cuda::CompletionRef completion;
        cudaError_t status = cudaSuccess;
#if DEBUG_LOG
        cudaStream_t stream = nullptr;
#endif

        explicit operator bool() const noexcept { return status == cudaSuccess && ptr != nullptr; }
    };

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

    CudaDeviceContext(int deviceId, int workStreamCount, bool serialCuda);
    CudaDeviceContext(CudaDeviceContext const&) = delete;
    CudaDeviceContext& operator=(CudaDeviceContext const&) = delete;
    ~CudaDeviceContext();

    static std::shared_ptr<CudaDeviceContext> shared(int deviceId, int workStreamCount, bool serialCuda);

    int deviceId() const noexcept { return deviceId_; }
    bool serialCuda() const noexcept { return serialCuda_; }

    cuda::Stream leaseWorkStream();
    cuda::CublasStream leaseBlasStream();
    std::size_t freeMemorySnapshot();
    DeviceAllocation allocateFromPool(cudaMemPool_t pool, std::size_t bytes);
    void preallocatePool(cudaMemPool_t pool, std::size_t bytes);
    cuda::CompletionRef recordCompletionEvent(cudaStream_t producerStream);
    cuda::CompletionRef recordCompletionEvent(cudaStream_t producerStream, std::vector<EventDependencyRef> antecedents);
    cudaEvent_t acquireEvent();
    void retireEvent(cudaEvent_t event);
    cudaEvent_t recordEvent(cudaStream_t stream);
    EventDependencyRef recordDependencyEvent(cudaStream_t stream);
    void waitEvent(cudaStream_t stream, cudaEvent_t event);
    void enqueueAsyncFree(void* ptr, cudaMemPool_t pool, std::size_t bytes,
                          std::vector<EventDependencyRef> dependencies);
    ScratchLease acquireScratch(std::size_t bytes, cudaStream_t stream);
    void syncWorkStreams(const char* reason = "work_unspecified");
    void syncMemoryStream(const char* reason = "memory_unspecified");
    void flushPoolCache();
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
        std::uint64_t poolCacheHit = 0;
        std::uint64_t poolCacheMiss = 0;
        std::uint64_t poolCacheStore = 0;
        std::uint64_t poolCacheBypass = 0;
        std::uint64_t poolCacheRelease = 0;
        std::unordered_map<std::string, std::uint64_t> streamSyncByReason;
    };

    struct PendingFree
    {
        void* ptr = nullptr;
        cudaEvent_t completeEvent = nullptr;
        std::vector<EventDependencyRef> dependencies;
    };

    struct BlasLane
    {
        cudaStream_t stream = nullptr;
        cublasHandle_t handle = nullptr;
        void* workspace = nullptr;
        std::size_t workspaceBytes = 0;
    };

    struct CachedPoolAllocation
    {
        void* ptr = nullptr;
        cudaMemPool_t pool = nullptr;
        std::size_t bytes = 0;
        cuda::CompletionRef ready;
    };

    int deviceId_ = 0;
    bool serialCuda_ = false;
    bool logCounters_ = false;
    bool released_ = false;
    bool freeMemorySnapshotValid_ = false;
    std::size_t freeMemorySnapshot_ = 0;
    std::size_t maxCachedPoolBytes_ = 0;
    std::size_t cachedPoolBytes_ = 0;
    std::size_t peakCachedPoolBytes_ = 0;
    cudaStream_t memoryStream_ = nullptr;
    std::deque<cudaStream_t> availableWorkStreams_;
    std::size_t createdWorkStreamCount_ = 0;
    std::vector<BlasLane> blasLanes_;
    std::deque<std::size_t> availableBlasLanes_;
    std::vector<cudaEvent_t> freeEvents_;
    std::vector<cudaEvent_t> retiredEvents_;
    std::vector<PendingFree> pendingFrees_;
    std::vector<CachedPoolAllocation> cachedPoolAllocations_;
    std::vector<std::shared_ptr<ScratchBuffer>> freeScratchBuffers_;
    std::vector<std::shared_ptr<ScratchBuffer>> allScratchBuffers_;
    std::mutex scratchMutex_;
    Counters counters_;
    std::uint64_t dependencySequence_ = 0;

    void reclaimRetiredEvents();
    void reclaimCompletedAsyncFrees();
    void countStreamSync(const char* reason);
    cudaStream_t acquireWorkStream();
    void returnWorkStream(cudaStream_t stream);
    std::size_t acquireBlasLane();
    void returnBlasLane(std::size_t laneIndex);
    cublasHandle_t prepareBlasHandle(std::size_t laneIndex);
    void releaseScratch(std::shared_ptr<ScratchBuffer> buffer, cudaStream_t stream);
    CachedPoolAllocation takeCachedPoolAllocation(cudaMemPool_t pool, std::size_t bytes);
    bool cachePoolAllocation(void* ptr, cudaMemPool_t pool, std::size_t bytes,
                             std::vector<EventDependencyRef>& dependencies);
    void printCounters() const;

    friend class ScratchLease;
    friend class cuda::Stream;
    friend class cuda::CublasStream;
    friend class cuda::Completion;
};

namespace cuda
{

class Completion {
  public:
    Completion() = default;
    Completion(Completion const&) = delete;
    Completion& operator=(Completion const&) = delete;
    Completion(Completion&& other) noexcept;
    Completion& operator=(Completion&& other) noexcept;
    ~Completion() = default;

    cudaStream_t stream() const noexcept;
    std::uint64_t sequence() const noexcept { return publishSequence_; }
    void waitOn(cudaStream_t consumerStream);
    CudaDeviceContext::EventDependencyRef dependencyEvent();

  private:
    Completion(CudaDeviceContext& context, cudaStream_t producerStream,
               std::vector<CudaDeviceContext::EventDependencyRef> antecedents = {});

    CudaDeviceContext* context_ = nullptr;
    CudaDeviceContext::EventDependencyRef event_;
    std::vector<CudaDeviceContext::EventDependencyRef> antecedents_;
    cudaStream_t producerStream_ = nullptr;
    std::uint64_t publishSequence_ = 0;

    friend class tensor::CudaDeviceContext;
};

class Stream {
  public:
    Stream() = default;
    Stream(Stream const&) = delete;
    Stream& operator=(Stream const&) = delete;
    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) noexcept;
    ~Stream();

    cudaStream_t stream() const noexcept { return stream_; }
    explicit operator bool() const noexcept { return stream_ != nullptr; }
    void setDevice() const;
    CudaDeviceContext::ScratchLease acquireScratch(std::size_t bytes) const;
    CompletionRef recordCompletion() const;
    void release();

  private:
    Stream(CudaDeviceContext& context, cudaStream_t stream) : context_(&context), stream_(stream) {}

    CudaDeviceContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;

    friend class tensor::CudaDeviceContext;
};

class CublasStream {
  public:
    CublasStream() = default;
    CublasStream(CublasStream const&) = delete;
    CublasStream& operator=(CublasStream const&) = delete;
    CublasStream(CublasStream&& other) noexcept;
    CublasStream& operator=(CublasStream&& other) noexcept;
    ~CublasStream();

    cudaStream_t stream() const noexcept;
    explicit operator bool() const noexcept { return context_ != nullptr; }
    void setDevice() const;
    cublasHandle_t prepare_handle() const;
    CudaDeviceContext::ScratchLease acquireScratch(std::size_t bytes) const;
    CompletionRef recordCompletion() const;
    void release();

  private:
    CublasStream(CudaDeviceContext& context, std::size_t laneIndex) : context_(&context), laneIndex_(laneIndex) {}

    CudaDeviceContext* context_ = nullptr;
    std::size_t laneIndex_ = 0;

    friend class tensor::CudaDeviceContext;
};

} // namespace cuda

} // namespace tensor
