#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
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
    void syncWorkStreams() const;
    void syncMemoryStream() const;
    void release();

  private:
    int deviceId_ = 0;
    bool serialCuda_ = false;
    bool released_ = false;
    cudaStream_t memoryStream_ = nullptr;
    std::vector<WorkSlot> workSlots_;
    std::size_t nextWorkSlot_ = 0;
};

} // namespace tensor
