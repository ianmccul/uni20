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
class StreamManager;

class GpuBuffer {
    friend class Swapper;
    void* ptr;
    int id;
    size_t dim1;
    size_t dim2;
    void* hostPtr;
    bool dependencyEventsEnabled = true;
    CudaDeviceContext* deviceContext = nullptr;

    CudaDeviceContext::EventDependencyRef useFinishEvent;
    cudaStream_t useFinishStream = nullptr;
    CudaDeviceContext::EventDependencyRef writeFinishEvent;
    cudaStream_t writeFinishStream = nullptr;

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
    void waitForWriteFinish(cudaStream_t stream);
    void notifyWriteFinish(cudaStream_t stream);
    void notifyWriteFinish(cudaStream_t stream, CudaDeviceContext::EventDependencyRef event);
    void waitForReadFinish(cudaStream_t stream);
    void notifyReadFinish(cudaStream_t stream);
    void notifyReadFinish(cudaStream_t stream, CudaDeviceContext::EventDependencyRef event);
};

class Swapper {
    using MapType = std::unordered_map<int, std::shared_ptr<GpuBuffer>>;
    std::vector<MapType> hostToGpuMaps;

    using preStoreMapType = std::unordered_map<int, std::pair<int, std::shared_ptr<GpuBuffer>>>;
    preStoreMapType preStoreMap;

    std::unordered_map<int, int> matToNCCLIdMap;

    std::vector<std::unique_ptr<CudaDeviceContext>> deviceContexts;
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
    Swapper();
    Swapper(const Swapper&) = delete;
    Swapper& operator=(const Swapper&) = delete;
    ~Swapper();

    std::shared_ptr<GpuBuffer> getGpuBufferOrNone(Matrix mat, int deviceId);
    std::shared_ptr<GpuBuffer> getForRead(Matrix mat, int deviceId, cudaStream_t stream);
    std::shared_ptr<GpuBuffer> getForWrite(Matrix mat, int deviceId, cudaStream_t stream);
    std::shared_ptr<GpuBuffer> getForReadNoWait(Matrix mat, int deviceId);
    std::shared_ptr<GpuBuffer> getForWriteNoWait(Matrix mat, int deviceId);
    cudaStream_t preferredStreamForAccess(const std::vector<std::shared_ptr<GpuBuffer>>& readBuffers,
                                          const std::vector<std::shared_ptr<GpuBuffer>>& writeBuffers) const;
    void waitForAccessDependencies(const std::vector<std::shared_ptr<GpuBuffer>>& readBuffers,
                                   const std::vector<std::shared_ptr<GpuBuffer>>& writeBuffers, cudaStream_t stream);
    void syncBuffer(Matrix mat, int deviceId, cudaStream_t stream);
    void freeBuffer(Matrix mat, int deviceId, cudaStream_t stream);

    void freeAllBuffer(int deviceId);
    void freeAllEvents(int deviceId);
    void syncMemStream(int deviceId);
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
    void preStoreMatrix(Matrix mat, int deviceId);
    void copyHostToPreStoreMatrix(Matrix mat);
    void copyPreStoreMatrixToHost(Matrix mat);
    void registerGpuAllocation(Matrix mat, int deviceId);
    std::pair<int, std::shared_ptr<GpuBuffer>> getPreStoreBufferOrNone(Matrix mat);
    void notifyMatrixRead(Matrix mat, int deviceId, cudaStream_t stream, CudaDeviceContext::EventDependencyRef event);
    void notifyMatrixWrite(Matrix mat, int deviceId, cudaStream_t stream, CudaDeviceContext::EventDependencyRef event);
    void copyMatrix(Matrix mat, std::shared_ptr<GpuBuffer> buffer, int deviceId, cudaStream_t stream,
                    StreamManager& streamManager);
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
