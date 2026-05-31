#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensor
{

class CudaDeviceContext {
  public:
    // The current TensorContraction executor borrows slots from one host thread
    // per active device.  Each slot owns its cuBLAS handle so stream selection
    // is fixed at construction instead of mutating a shared handle.
    struct WorkSlot
    {
        cudaStream_t stream = nullptr;
        cublasHandle_t handle = nullptr;
    };

    CudaDeviceContext(int deviceId, int workStreamCount, bool serialCuda);
    CudaDeviceContext(CudaDeviceContext const&) = delete;
    CudaDeviceContext& operator=(CudaDeviceContext const&) = delete;
    ~CudaDeviceContext();

    int deviceId() const noexcept { return deviceId_; }
    bool serialCuda() const noexcept { return serialCuda_; }
    cudaStream_t memoryStream() const noexcept { return memoryStream_; }

    WorkSlot& nextWorkSlot();
    cudaEvent_t acquireEvent();
    void retireEvent(cudaEvent_t event);
    cudaEvent_t recordEvent(cudaStream_t stream);
    void waitEvent(cudaStream_t stream, cudaEvent_t event);
    void syncWorkStreams();
    void syncMemoryStream();
    void release();

  private:
    struct Counters
    {
        std::uint64_t eventCreate = 0;
        std::uint64_t eventRecord = 0;
        std::uint64_t eventWait = 0;
        std::uint64_t eventDestroy = 0;
        std::uint64_t streamSync = 0;
    };

    int deviceId_ = 0;
    bool serialCuda_ = false;
    bool logCounters_ = false;
    bool released_ = false;
    cudaStream_t memoryStream_ = nullptr;
    std::vector<WorkSlot> workSlots_;
    std::vector<cudaEvent_t> freeEvents_;
    std::vector<cudaEvent_t> retiredEvents_;
    Counters counters_;
    std::size_t nextWorkSlot_ = 0;

    void reclaimRetiredEvents();
    void printCounters() const;
};

} // namespace tensor
