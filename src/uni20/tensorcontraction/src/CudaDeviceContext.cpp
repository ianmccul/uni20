#include "CudaDeviceContext.hpp"

#include "Utils.h"

#include <algorithm>

namespace tensor
{

CudaDeviceContext::CudaDeviceContext(int deviceId, int workStreamCount, bool serialCuda)
    : deviceId_(deviceId), serialCuda_(serialCuda)
{
  CUDA_CALL(cudaSetDevice(deviceId_));

  if (serialCuda_)
  {
    memoryStream_ = cudaStreamLegacy;
    workStreamCount = 1;
  }
  else
  {
    CUDA_CALL(cudaStreamCreate(&memoryStream_));
    workStreamCount = std::max(1, workStreamCount);
  }

  workSlots_.reserve(static_cast<std::size_t>(workStreamCount));
  for (int i = 0; i < workStreamCount; ++i)
  {
    WorkSlot slot;
    slot.stream = serialCuda_ ? cudaStreamLegacy : nullptr;
    if (!serialCuda_)
    {
      CUDA_CALL(cudaStreamCreate(&slot.stream));
    }
    CUBLAS_CALL(cublasCreate(&slot.handle));
    CUBLAS_CALL(cublasSetStream(slot.handle, slot.stream));
    workSlots_.push_back(slot);
  }
}

CudaDeviceContext::~CudaDeviceContext() { release(); }

auto CudaDeviceContext::nextWorkSlot() -> WorkSlot&
{
  auto& slot = workSlots_[nextWorkSlot_];
  nextWorkSlot_ = (nextWorkSlot_ + 1) % workSlots_.size();
  CUDA_CALL(cudaSetDevice(deviceId_));
  return slot;
}

void CudaDeviceContext::syncWorkStreams() const
{
  CUDA_CALL(cudaSetDevice(deviceId_));
  if (serialCuda_)
  {
    CUDA_CALL(cudaStreamSynchronize(cudaStreamLegacy));
    return;
  }
  for (auto const& slot : workSlots_)
  {
    CUDA_CALL(cudaStreamSynchronize(slot.stream));
  }
}

void CudaDeviceContext::syncMemoryStream() const
{
  CUDA_CALL(cudaSetDevice(deviceId_));
  CUDA_CALL(cudaStreamSynchronize(memoryStream_));
}

void CudaDeviceContext::release()
{
  if (released_)
  {
    return;
  }
  released_ = true;

  CUDA_CALL(cudaSetDevice(deviceId_));
  syncWorkStreams();
  syncMemoryStream();

  for (auto& slot : workSlots_)
  {
    if (slot.handle != nullptr)
    {
      CUBLAS_CALL(cublasDestroy(slot.handle));
      slot.handle = nullptr;
    }
    if (!serialCuda_ && slot.stream != nullptr)
    {
      CUDA_CALL(cudaStreamDestroy(slot.stream));
      slot.stream = nullptr;
    }
  }
  workSlots_.clear();

  if (!serialCuda_ && memoryStream_ != nullptr)
  {
    CUDA_CALL(cudaStreamDestroy(memoryStream_));
  }
  memoryStream_ = nullptr;
}

} // namespace tensor
