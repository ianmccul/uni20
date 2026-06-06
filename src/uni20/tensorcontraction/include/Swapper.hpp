#pragma once
#include <cublas_v2.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <unordered_map>
#include <vector>

#include "CudaDeviceContext.hpp"
#include "Matrix.hpp"

namespace tensor
{

class Swapper;

class GpuBuffer {
    friend class Swapper;
    struct AccessGeneration
    {
        cuda::CompletionRef completion;
    };

    struct AccessState
    {
        // One memory block owns its CUDA dependency generations.  The writer
        // generation gates future reads/writes; reader generation gates writes.
        AccessGeneration readers;
        AccessGeneration writer;
    };

    struct AllocationGroup
    {
        void* basePtr = nullptr;
        size_t bytes = 0;
        size_t liveBuffers = 0;
        bool freeScheduled = false;
        std::vector<CudaDeviceContext::EventDependencyRef> dependencies;
    };

    void* ptr;
    int id;
    size_t dim1;
    size_t dim2;
    void* hostPtr;
    bool hasValidContent = false;
    bool dependencyEventsEnabled = true;
    CudaDeviceContext* deviceContext = nullptr;
    AccessState accessState;
    std::shared_ptr<AllocationGroup> allocationGroup;

  public:
    GpuBuffer() = delete;
    GpuBuffer(void* ptr, Matrix mat, bool dependencyEventsEnabled, CudaDeviceContext& deviceContext,
              std::shared_ptr<AllocationGroup> allocationGroup);
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    GpuBuffer(GpuBuffer&& other);
    GpuBuffer& operator=(GpuBuffer&& other);

    double* getPtr();
    int getId() const;
    size_t size() const { return dim1 * dim2; }
    size_t sizeInByte() const { return size() * sizeof(double); }
    bool contentValid() const { return hasValidContent; }
    void* allocationBasePtr() const { return allocationGroup == nullptr ? ptr : allocationGroup->basePtr; }
    size_t allocationSizeInByte() const
    {
      return allocationGroup == nullptr ? this->sizeInByte() : allocationGroup->bytes;
    }
    size_t allocationOffsetInByte() const
    {
      if (allocationGroup == nullptr || allocationGroup->basePtr == nullptr)
      {
        return 0;
      }
      auto const* base = static_cast<std::byte const*>(allocationGroup->basePtr);
      auto const* current = static_cast<std::byte const*>(ptr);
      return static_cast<size_t>(current - base);
    }
    DeviceMatrixView deviceView(int deviceId) const;
    void publishAllocation(cuda::CompletionRef completion);
    void waitBeforeRead(cudaStream_t stream);
    void waitBeforeWrite(cudaStream_t stream);
    void publishRead(cudaStream_t stream);
    void publishRead(cuda::CompletionRef completion);
    void publishWrite(cudaStream_t stream);
    void publishWrite(cuda::CompletionRef completion);
};

class Swapper {
  public:
    struct RuntimeCounters
    {
        std::uint64_t h2dCopies = 0;
        std::uint64_t h2dBytes = 0;
        std::uint64_t d2hCopies = 0;
        std::uint64_t d2hBytes = 0;
        std::uint64_t d2dCopies = 0;
        std::uint64_t d2dBytes = 0;
        std::uint64_t peerCopies = 0;
        std::uint64_t peerBytes = 0;
        std::uint64_t ensureLocalPeerCopies = 0;
        std::uint64_t ensureLocalPeerBytes = 0;
        std::uint64_t preStoreRelocateD2dCopies = 0;
        std::uint64_t preStoreRelocateD2dBytes = 0;
        std::uint64_t preStoreRelocatePeerCopies = 0;
        std::uint64_t preStoreRelocatePeerBytes = 0;
        std::uint64_t syncBufferPeerCopies = 0;
        std::uint64_t syncBufferPeerBytes = 0;
        std::uint64_t cudaEventCreate = 0;
        std::uint64_t cudaEventRecord = 0;
        std::uint64_t cudaEventWait = 0;
        std::uint64_t cudaEventQuery = 0;
        std::uint64_t cudaEventDestroy = 0;
        std::uint64_t cudaStreamSync = 0;
        std::uint64_t cudaAsyncFree = 0;
        std::uint64_t cudaAsyncFreeReclaim = 0;
        std::uint64_t cudaAsyncFreePoll = 0;
        std::uint64_t cudaPoolCacheHit = 0;
        std::uint64_t cudaPoolCacheMiss = 0;
        std::uint64_t cudaPoolCacheStore = 0;
        std::uint64_t cudaPoolCacheBypass = 0;
        std::uint64_t cudaPoolCacheRelease = 0;
    };

  private:
    using MapType = std::unordered_map<int, std::shared_ptr<GpuBuffer>>;
    std::vector<MapType> hostToGpuMaps;

    using preStoreMapType = std::unordered_map<int, std::pair<int, std::shared_ptr<GpuBuffer>>>;
    preStoreMapType preStoreMap;

    std::unordered_map<int, int> matToNCCLIdMap;

    std::vector<std::shared_ptr<CudaDeviceContext>> deviceContexts;
    std::vector<cudaMemPool_t> memPools;
    std::vector<bool> ownsMemPool;
    std::vector<std::set<int>> pinnedMatrix;

#if DEBUG_LOG
    std::unordered_map<int, std::string> matrixNames;
#endif

    int deviceCount;
    bool serialCuda = false;
    bool dependencyEventsEnabled = true;
    bool memoryPoolsInitialized = false;
    bool released = false;
    RuntimeCounters runtimeCounters_;

    bool isPinned(int id, int deviceId);
    void freeBuffer(std::shared_ptr<GpuBuffer>, int deviceId);
    void destroyBufferEvents(std::shared_ptr<GpuBuffer> const& buffer);
    void invalidateCopiesAfterWrite(std::shared_ptr<GpuBuffer> const& writer);
    std::vector<CudaDeviceContext::EventDependencyRef>
    collectBufferDependencies(std::shared_ptr<GpuBuffer> const& buffer) const;

  public:
    /// \brief Access direction for a slab operation.
    enum class SlabAccessKind
    {
      /// \brief The operation reads every sub-block in the slab.
      Read,
      /// \brief The operation writes every sub-block in the slab.
      Write,
    };

    class GpuAccessPlan {
        Swapper& swapper;
        int deviceId = 0;
        cuda::Stream streamOwner;
        std::vector<std::shared_ptr<GpuBuffer>> readBuffers;
        std::vector<std::shared_ptr<GpuBuffer>> writeBuffers;

      public:
        GpuAccessPlan(Swapper& swapper, int deviceId, std::vector<std::shared_ptr<GpuBuffer>> readBuffers,
                      std::vector<std::shared_ptr<GpuBuffer>> writeBuffers);
        GpuAccessPlan(GpuAccessPlan const&) = delete;
        GpuAccessPlan& operator=(GpuAccessPlan const&) = delete;
        GpuAccessPlan(GpuAccessPlan&&) = delete;
        GpuAccessPlan& operator=(GpuAccessPlan&&) = delete;
        ~GpuAccessPlan();

        cudaStream_t stream() const { return streamOwner.stream(); }
        cuda::CompletionRef recordCompletion() const;
        void publishCompletion(cuda::CompletionRef completion) const;
    };

    class BlasAccessPlan {
        Swapper& swapper;
        int deviceId = 0;
        cuda::CublasStream streamOwner;
        std::vector<std::shared_ptr<GpuBuffer>> readBuffers;
        std::vector<std::shared_ptr<GpuBuffer>> writeBuffers;

      public:
        BlasAccessPlan(Swapper& swapper, int deviceId, std::vector<std::shared_ptr<GpuBuffer>> readBuffers,
                       std::vector<std::shared_ptr<GpuBuffer>> writeBuffers);
        BlasAccessPlan(BlasAccessPlan const&) = delete;
        BlasAccessPlan& operator=(BlasAccessPlan const&) = delete;
        BlasAccessPlan(BlasAccessPlan&&) = delete;
        BlasAccessPlan& operator=(BlasAccessPlan&&) = delete;
        ~BlasAccessPlan();

        cudaStream_t stream() const { return streamOwner.stream(); }
        cublasHandle_t handle() const;
        cuda::CompletionRef recordCompletion() const;
        void publishCompletion(cuda::CompletionRef completion) const;
    };

    /// \brief RAII access token for a whole coalesced GPU allocation slab.
    /// \details The plan waits for all sub-block dependencies before exposing a
    ///          stream and publishes one completion back to all sub-blocks when
    ///          destroyed.
    class SlabAccessPlan {
        Swapper& swapper;
        int deviceId = 0;
        SlabAccessKind accessKind = SlabAccessKind::Read;
        cuda::Stream streamOwner;
        std::vector<std::shared_ptr<GpuBuffer>> buffers;
        void* basePtr = nullptr;
        size_t bytes = 0;
        bool published = false;

      public:
        /// \brief Acquire a whole coalesced allocation slab for one operation.
        /// \param swapper Owner of the device buffers and dependency state.
        /// \param deviceId CUDA device containing the slab.
        /// \param accessKind Whether the operation reads or writes the slab.
        /// \param buffers Ordered sub-block buffers covering the complete slab.
        SlabAccessPlan(Swapper& swapper, int deviceId, SlabAccessKind accessKind,
                       std::vector<std::shared_ptr<GpuBuffer>> buffers);
        SlabAccessPlan(SlabAccessPlan const&) = delete;
        SlabAccessPlan& operator=(SlabAccessPlan const&) = delete;
        SlabAccessPlan(SlabAccessPlan&&) = delete;
        SlabAccessPlan& operator=(SlabAccessPlan&&) = delete;
        ~SlabAccessPlan();

        /// \brief Return the CUDA stream prepared for the slab operation.
        /// \return CUDA stream that has waited on all required sub-block events.
        cudaStream_t stream() const { return streamOwner.stream(); }
        /// \brief Return the base pointer of the coalesced allocation slab.
        /// \return Device pointer to the first byte of the slab.
        void* data() const { return basePtr; }
        /// \brief Return the total byte count covered by the slab.
        /// \return Number of bytes in the complete coalesced allocation.
        size_t sizeInByte() const { return bytes; }
        /// \brief Record completion of the slab operation in the plan stream.
        /// \return Completion token that can be published to all sub-blocks.
        cuda::CompletionRef recordCompletion() const;
        /// \brief Publish an operation completion to every sub-block in the slab.
        /// \param completion Completion token recorded after the slab operation.
        void publishCompletion(cuda::CompletionRef completion);
    };

    Swapper();
    Swapper(const Swapper&) = delete;
    Swapper& operator=(const Swapper&) = delete;
    ~Swapper();

    std::shared_ptr<GpuBuffer> getGpuBufferOrNone(Matrix mat, int deviceId);
    std::shared_ptr<GpuBuffer> getForRead(Matrix mat, int deviceId, cudaStream_t stream);
    std::shared_ptr<GpuBuffer> getForWrite(Matrix mat, int deviceId, cudaStream_t stream);
    std::shared_ptr<GpuBuffer> getForReadNoWait(Matrix mat, int deviceId);
    std::shared_ptr<GpuBuffer> getForWriteNoWait(Matrix mat, int deviceId);
    GpuAccessPlan createAccessPlan(std::vector<std::shared_ptr<GpuBuffer>> readBuffers,
                                   std::vector<std::shared_ptr<GpuBuffer>> writeBuffers, int deviceId);
    BlasAccessPlan createBlasAccessPlan(std::vector<std::shared_ptr<GpuBuffer>> readBuffers,
                                        std::vector<std::shared_ptr<GpuBuffer>> writeBuffers, int deviceId);
    /// \brief Acquire a whole coalesced pre-store slab for a slab operation.
    /// \param mats Ordered matrices that must cover the complete slab.
    /// \param deviceId CUDA device containing the slab.
    /// \param accessKind Whether the operation reads or writes the slab.
    /// \return RAII plan exposing the synchronized slab pointer and stream.
    SlabAccessPlan createSlabAccessPlan(const std::vector<Matrix>& mats, int deviceId, SlabAccessKind accessKind);
    void waitForAccessDependencies(const std::vector<std::shared_ptr<GpuBuffer>>& readBuffers,
                                   const std::vector<std::shared_ptr<GpuBuffer>>& writeBuffers, cudaStream_t stream);
    void publishAccessCompletion(const std::vector<std::shared_ptr<GpuBuffer>>& readBuffers,
                                 const std::vector<std::shared_ptr<GpuBuffer>>& writeBuffers, cudaStream_t stream,
                                 cuda::CompletionRef completion);
    void syncBuffer(Matrix mat, int deviceId, cudaStream_t stream);
    void freeBuffer(Matrix mat, int deviceId, cudaStream_t stream);

    void freeAllBuffer(int deviceId);
    void freeAllEvents(int deviceId);
    void syncMemStream(int deviceId, const char* reason = "swapper_sync_mem_stream");
    void initMemPools();
    void dumpMemPoolStatus(int deviceId);
    cudaMemPool_t getMemPool(int deviceId) const { return memPools[deviceId]; }
    int getDeviceCount() const { return deviceCount; }
    bool dependencyEventsActive() const { return dependencyEventsEnabled; }
    CudaDeviceContext& deviceContext(int deviceId) { return *deviceContexts[deviceId]; }
    CudaDeviceContext const& deviceContext(int deviceId) const { return *deviceContexts[deviceId]; }
    [[nodiscard]] RuntimeCounters runtimeCounters() const;

    void freeAllUnpinMatrices(int deviceId);
    void pinMatrix(Matrix mat, int deviceId);
    void unpinMatrix(Matrix mat, int deviceId);
    std::shared_ptr<GpuBuffer> allocate(Matrix mat, int deviceId);
    std::shared_ptr<GpuBuffer> ensureLocalCopy(Matrix mat, int deviceId);
    std::pair<int, std::shared_ptr<GpuBuffer>> findLocalSourceBuffer(Matrix mat, int requesterDeviceId);
    void preStoreMatrix(Matrix mat, int deviceId);
    std::shared_ptr<GpuBuffer> uploadHostMatrix(HostMatrixView host, int deviceId);
    void uploadHostMatricesCoalesced(const std::vector<Matrix>& mats, std::span<double const> values, int deviceId);
    void copyHostToPreStoreMatrix(Matrix mat);
    void refreshHostMatrixToDevice(HostMatrixView host);
    bool refreshHostMatricesToDeviceCoalesced(const std::vector<Matrix>& mats, std::span<double const> values,
                                              int deviceId);
    void copyPreStoreMatrixToHost(Matrix mat);
    void downloadDeviceToHost(HostMatrixView host);
    bool downloadDeviceMatricesToHostCoalesced(const std::vector<Matrix>& mats, std::span<double> values);
    void registerGpuAllocation(Matrix mat, int deviceId);
    void registerGpuAllocationsCoalesced(const std::vector<Matrix>& mats, int deviceId);
    /// \brief Ensure a pre-store matrix buffer exists on the requested CUDA device.
    /// \param mat Matrix whose pre-store buffer should be placed.
    /// \param deviceId Target CUDA device.
    /// \param preserveExistingContent If true, copy existing resident contents to the new placement.
    void ensurePreStoreOnDevice(Matrix mat, int deviceId, bool preserveExistingContent);
    /// \brief Ensure a batch of pre-store matrix buffers forms one coalesced slab on a CUDA device.
    /// \param mats Ordered matrices that should share one contiguous allocation.
    /// \param deviceId Target CUDA device.
    /// \param preserveExistingContent If true, copy existing resident contents into the new slab.
    void ensurePreStoreCoalescedOnDevice(std::vector<Matrix> const& mats, int deviceId, bool preserveExistingContent);
    bool anyPreStoreBuffer(const std::vector<Matrix>& mats) const;
    std::optional<int> commonPreStoreDevice(const std::vector<Matrix>& mats) const;
    bool preStoreBuffersAreCoalesced(const std::vector<Matrix>& mats, int deviceId) const;
    /// \brief Return ordered buffers for a complete coalesced pre-store slab.
    /// \param mats Matrices that must cover the slab in storage order.
    /// \param deviceId Device containing the slab.
    /// \return Ordered sub-block buffers, or empty if the matrices do not form
    ///         a complete coalesced slab on the requested device.
    std::vector<std::shared_ptr<GpuBuffer>> collectCoalescedPreStoreBuffers(const std::vector<Matrix>& mats,
                                                                            int deviceId) const;
    std::pair<int, std::shared_ptr<GpuBuffer>> getPreStoreBufferOrNone(Matrix mat);
    void notifyMatrixRead(Matrix mat, int deviceId, cuda::CompletionRef completion);
    void notifyMatrixWrite(Matrix mat, int deviceId, cuda::CompletionRef completion);
    void copyMatrix(Matrix mat, std::shared_ptr<GpuBuffer> buffer, int deviceId, cudaStream_t stream);
    void copyHostToDevice(HostMatrixView host, std::shared_ptr<GpuBuffer> buffer, int deviceId, cudaStream_t stream);
    void exchangePreStoreMap();
    bool isOnRemoteGpu(int matId) const;
    int getRemoteNCCLId(int matId) const;
    void release();

#if DEBUG_LOG
    void setMatrixName(Matrix mat, const std::string& name);
    std::string getMatrixName(int matId) const;
#endif

    void clear();
    void clear(Matrix mat);
};

} // namespace tensor
