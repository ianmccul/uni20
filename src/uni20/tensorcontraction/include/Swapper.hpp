#pragma once
#include <cublas_v2.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
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

    void* ptr;
    int id;
    size_t dim1;
    size_t dim2;
    void* hostPtr;
    bool hasValidContent = false;
    bool dependencyEventsEnabled = true;
    CudaDeviceContext* deviceContext = nullptr;
    AccessState accessState;

  public:
    GpuBuffer() = delete;
    GpuBuffer(void* ptr, Matrix mat, bool dependencyEventsEnabled, CudaDeviceContext& deviceContext);
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    GpuBuffer(GpuBuffer&& other);
    GpuBuffer& operator=(GpuBuffer&& other);

    double* getPtr();
    int getId() const;
    size_t size() const { return dim1 * dim2; }
    size_t sizeInByte() const { return size() * sizeof(double); }
    bool contentValid() const { return hasValidContent; }
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
    bool released = false;

    bool isPinned(int id, int deviceId);
    void freeBuffer(std::shared_ptr<GpuBuffer>, int deviceId);
    void destroyBufferEvents(std::shared_ptr<GpuBuffer> const& buffer);

  public:
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

    void freeAllUnpinMatrices(int deviceId);
    void pinMatrix(Matrix mat, int deviceId);
    void unpinMatrix(Matrix mat, int deviceId);
    std::shared_ptr<GpuBuffer> allocate(Matrix mat, int deviceId);
    std::shared_ptr<GpuBuffer> ensureLocalCopy(Matrix mat, int deviceId);
    std::pair<int, std::shared_ptr<GpuBuffer>> findLocalSourceBuffer(Matrix mat, int requesterDeviceId);
    void preStoreMatrix(Matrix mat, int deviceId);
    std::shared_ptr<GpuBuffer> uploadHostMatrix(HostMatrixView host, int deviceId);
    void copyHostToPreStoreMatrix(Matrix mat);
    void refreshHostMatrixToDevice(HostMatrixView host);
    void copyPreStoreMatrixToHost(Matrix mat);
    void downloadDeviceToHost(HostMatrixView host);
    void registerGpuAllocation(Matrix mat, int deviceId);
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
