#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
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

    // The current TensorContraction executor borrows slots from one host thread
    // per active device.  Each slot owns its cuBLAS handle so stream selection
    // is fixed at construction instead of mutating a shared handle.
    struct WorkSlot
    {
        cudaStream_t stream = nullptr;
        cublasHandle_t handle = nullptr;
        std::uint64_t lastUse = 0;
    };

    CudaDeviceContext(int deviceId, int workStreamCount, bool serialCuda);
    CudaDeviceContext(CudaDeviceContext const&) = delete;
    CudaDeviceContext& operator=(CudaDeviceContext const&) = delete;
    ~CudaDeviceContext();

    static std::shared_ptr<CudaDeviceContext> shared(int deviceId, int workStreamCount, bool serialCuda);

    int deviceId() const noexcept { return deviceId_; }
    bool serialCuda() const noexcept { return serialCuda_; }
    cudaStream_t memoryStream() const noexcept { return memoryStream_; }

    WorkSlot& nextWorkSlot();
    WorkSlot& nextWorkSlot(cudaStream_t preferredStream);
    cudaEvent_t acquireEvent();
    void retireEvent(cudaEvent_t event);
    cudaEvent_t recordEvent(cudaStream_t stream);
    EventDependencyRef recordDependencyEvent(cudaStream_t stream);
    void waitEvent(cudaStream_t stream, cudaEvent_t event);
    void enqueueAsyncFree(void* ptr, cudaStream_t stream, std::vector<EventDependencyRef> dependencies);
    ScratchLease acquireScratch(std::size_t bytes, cudaStream_t stream);
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
    std::vector<WorkSlot> workSlots_;
    std::vector<cudaEvent_t> freeEvents_;
    std::vector<cudaEvent_t> retiredEvents_;
    std::vector<PendingFree> pendingFrees_;
    std::vector<std::shared_ptr<ScratchBuffer>> freeScratchBuffers_;
    std::vector<std::shared_ptr<ScratchBuffer>> allScratchBuffers_;
    std::mutex scratchMutex_;
    Counters counters_;
    std::size_t nextWorkSlot_ = 0;
    std::uint64_t workSlotUseCounter_ = 0;
    std::uint64_t dependencySequence_ = 0;

    void reclaimRetiredEvents();
    void reclaimCompletedAsyncFrees();
    void countStreamSync(const char* reason);
    WorkSlot& markWorkSlotUsed(WorkSlot& slot);
    void releaseScratch(std::shared_ptr<ScratchBuffer> buffer, cudaStream_t stream);
    void printCounters() const;

    friend class ScratchLease;
};

} // namespace tensor
