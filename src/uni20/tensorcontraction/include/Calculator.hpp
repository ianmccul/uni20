#pragma once

#include <cublas_v2.h>
#include <nccl.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "Matrix.hpp"
#include "Swapper.hpp"

namespace tensor
{

class StreamManager;

class WorkBase {
  protected:
    StreamManager& streamManager;
    Swapper& swapper;

  public:
    WorkBase(StreamManager& streamManager, Swapper& swapper) : streamManager(streamManager), swapper(swapper) {}
    virtual void execute();
    virtual std::vector<Matrix> readMatrices() const { return {}; }
    virtual std::vector<Matrix> writeMatrices() const { return {}; }
#if DEBUG_LOG
    virtual void dump();
    virtual const char* getTypeName() const { return "UNKNOWN"; }
    virtual std::string getTypeSpecificInfo() const { return ""; }
    virtual std::string getMatrixInfo() const { return ""; }
#endif
    virtual ~WorkBase() = default;
};

class MatWorkBase : public WorkBase {
  protected:
    std::vector<Matrix> matrices;
    double alpha;
    cudaEvent_t syncFinishEvent;

  public:
    MatWorkBase(const std::vector<Matrix>& mats, double alpha, StreamManager& streamManager, Swapper& swapper,
                cudaEvent_t syncFinishEvent = nullptr)
        : WorkBase(streamManager, swapper), matrices(mats), alpha(alpha), syncFinishEvent(syncFinishEvent)
    {}

    const std::vector<Matrix>& getMatrices() const { return matrices; }
    double getAlpha() const { return alpha; }
    cudaEvent_t getSyncFinishEvent() const { return syncFinishEvent; }

#if DEBUG_LOG
    std::string getMatrixInfo() const override;
#endif
};

class MatMulWork : public MatWorkBase {
    using MatWorkBase::MatWorkBase;

  public:
    void execute() override;
    std::vector<Matrix> readMatrices() const override { return {matrices[1], matrices[2]}; }
    std::vector<Matrix> writeMatrices() const override { return {matrices[0]}; }
#if DEBUG_LOG
    const char* getTypeName() const override { return "MATMUL"; }
#endif
};

class MatMulAccuWork : public MatWorkBase {
    using MatWorkBase::MatWorkBase;

  public:
    void execute() override;
    std::vector<Matrix> readMatrices() const override { return {matrices[0], matrices[1], matrices[2]}; }
    std::vector<Matrix> writeMatrices() const override { return {matrices[0]}; }
#if DEBUG_LOG
    const char* getTypeName() const override { return "MATMUL_ACCU"; }
    std::string getTypeSpecificInfo() const override;
#endif
};

class AddAccuWork : public MatWorkBase {
    using MatWorkBase::MatWorkBase;

  public:
    void execute() override;
    std::vector<Matrix> readMatrices() const override { return {matrices[0], matrices[1]}; }
    std::vector<Matrix> writeMatrices() const override { return {matrices[0]}; }
#if DEBUG_LOG
    const char* getTypeName() const override { return "ADD_ACCU"; }
    std::string getTypeSpecificInfo() const override;
#endif
};

class SyncWork : public MatWorkBase {
    using MatWorkBase::MatWorkBase;

  public:
    void execute() override;
    std::vector<Matrix> readMatrices() const override { return {matrices[0]}; }
#if DEBUG_LOG
    const char* getTypeName() const override { return "SYNC"; }
#endif
};

class PinWork : public MatWorkBase {
    using MatWorkBase::MatWorkBase;

  public:
    void execute() override;
#if DEBUG_LOG
    const char* getTypeName() const override { return "PIN"; }
#endif
};

class UnpinWork : public MatWorkBase {
    using MatWorkBase::MatWorkBase;

  public:
    void execute() override;
#if DEBUG_LOG
    const char* getTypeName() const override { return "UNPIN"; }
#endif
};

class FreeWork : public MatWorkBase {
    using MatWorkBase::MatWorkBase;

  public:
    void execute() override;
#if DEBUG_LOG
    const char* getTypeName() const override { return "FREE"; }
#endif
};

class MemsetWork : public MatWorkBase {
    using MatWorkBase::MatWorkBase;

  public:
    void execute() override;
    std::vector<Matrix> writeMatrices() const override { return {matrices[0]}; }
#if DEBUG_LOG
    const char* getTypeName() const override { return "MEMSET"; }
#endif
};

class NCCLSendRecvWork : public WorkBase {
    std::vector<std::tuple<int, Matrix, cudaEvent_t>> dstMatPair;
    ncclComm_t comm;

  public:
    NCCLSendRecvWork(std::vector<std::tuple<int, Matrix, cudaEvent_t>> dstMatPair, ncclComm_t comm,
                     StreamManager& streamManager, Swapper& swapper)
        : WorkBase(streamManager, swapper), dstMatPair(std::move(dstMatPair)), comm(comm)
    {}

    void execute() override;
    std::vector<Matrix> readMatrices() const override;
    std::vector<Matrix> writeMatrices() const override;
#if DEBUG_LOG
    const char* getTypeName() const override { return "NCCL_SENDRECV"; }
    std::string getMatrixInfo() const override;
#endif
};

class InnerProductWork : public MatWorkBase {
    double* result;

  public:
    InnerProductWork(const std::vector<Matrix>& mats, double alpha, StreamManager& streamManager, Swapper& swapper,
                     double* result)
        : MatWorkBase(mats, alpha, streamManager, swapper), result(result)
    {}
    void execute() override;
    std::vector<Matrix> readMatrices() const override { return {matrices[0], matrices[1]}; }
#if DEBUG_LOG
    const char* getTypeName() const override { return "INNER_PRODUCT"; }
#endif
};

class ScalarMulWork : public MatWorkBase {
    double* coff;

  public:
    ScalarMulWork(const std::vector<Matrix>& mats, double alpha, StreamManager& streamManager, Swapper& swapper,
                  double* coff)
        : MatWorkBase(mats, alpha, streamManager, swapper), coff(coff)
    {}
    void execute() override;
    std::vector<Matrix> readMatrices() const override { return {matrices[0]}; }
    std::vector<Matrix> writeMatrices() const override { return {matrices[0]}; }
#if DEBUG_LOG
    const char* getTypeName() const override { return "SCALAR_MUL"; }
#endif
};

class BarrierWork : public WorkBase {
    std::shared_ptr<std::atomic_int> count;
    int target;

  public:
    BarrierWork(int target, std::shared_ptr<std::atomic_int> count, StreamManager& streamManager, Swapper& swapper)
        : WorkBase(streamManager, swapper), count(count), target(target)
    {}
    void execute() override;

#if DEBUG_LOG
    const char* getTypeName() const override { return "BARRIER"; }
    std::string getTypeSpecificInfo() const override;
#endif
};

class StreamManager {
    int deviceId;
    CudaDeviceContext& deviceContext;
    cudaStream_t currentStream = nullptr;
    cublasHandle_t currentHandle = nullptr;
    CudaDeviceContext::VirtualStreamRef currentVirtualStream;
    CudaDeviceContext::ConcreteStreamLease currentLease;
    bool fixedStreamActive = false;

  public:
    StreamManager(Swapper& swapper, int deviceId, int deviceCount);

    void syncAllStreams() const;
    void clear();

    // Accessors for WorkBase subclasses
    int getDeviceId() const { return deviceId; }
    cublasHandle_t getHandle() const { return currentHandle; }
    cudaStream_t setEnv();
    cudaStream_t setEnv(cudaStream_t preferredStream);
    cudaStream_t getStream();
    cudaStream_t getStream(cudaStream_t preferredStream);
    cudaStream_t beginFixedStream(cudaStream_t preferredStream);
    void endFixedStream();
    cudaStream_t currentStreamHandle() const { return currentStream; }
    CudaDeviceContext::ScratchLease acquireScratch(std::size_t bytes)
    {
      return deviceContext.acquireScratch(bytes, currentStream);
    }
    CudaDeviceContext::VirtualStreamRef recordCompletion()
    {
      if (currentVirtualStream != nullptr)
      {
        currentVirtualStream->markPublished();
      }
      return currentVirtualStream;
    }

  private:
    void activateStream(cudaStream_t preferredStream = nullptr);
};

// Generic template factory for creating Work objects
template <typename T, typename... Args> std::unique_ptr<WorkBase> createWork(Args&&... args)
{
  return std::make_unique<T>(std::forward<Args>(args)...);
}

} // namespace tensor
