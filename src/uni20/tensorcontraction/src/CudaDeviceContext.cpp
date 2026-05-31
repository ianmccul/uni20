#include "CudaDeviceContext.hpp"

#include "Utils.h"

#include <algorithm>
#include <cstdio>

namespace tensor
{

CudaDeviceContext::EventDependency::~EventDependency()
{
  if (context != nullptr)
  {
    context->retireEvent(event);
  }
}

CudaDeviceContext::CudaDeviceContext(int deviceId, int workStreamCount, bool serialCuda)
    : deviceId_(deviceId), serialCuda_(serialCuda),
      logCounters_(envFlagEnabled("UNI20_TENSORCONTRACTION_CUDA_COUNTERS") ||
                   envFlagEnabled("TENSORCONTRACTION_CUDA_COUNTERS"))
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

cudaEvent_t CudaDeviceContext::acquireEvent()
{
  CUDA_CALL(cudaSetDevice(deviceId_));
  if (!freeEvents_.empty())
  {
    auto event = freeEvents_.back();
    freeEvents_.pop_back();
    return event;
  }

  cudaEvent_t event;
  CUDA_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  ++counters_.eventCreate;
  return event;
}

void CudaDeviceContext::retireEvent(cudaEvent_t event)
{
  if (event != nullptr)
  {
    retiredEvents_.push_back(event);
  }
}

cudaEvent_t CudaDeviceContext::recordEvent(cudaStream_t stream)
{
  auto event = acquireEvent();
  CUDA_CALL(cudaEventRecord(event, stream));
  ++counters_.eventRecord;
  return event;
}

CudaDeviceContext::EventDependencyRef CudaDeviceContext::recordDependencyEvent(cudaStream_t stream)
{
  return std::make_shared<EventDependency>(*this, recordEvent(stream));
}

void CudaDeviceContext::waitEvent(cudaStream_t stream, cudaEvent_t event)
{
  if (event == nullptr)
  {
    return;
  }
  CUDA_CALL(cudaStreamWaitEvent(stream, event));
  ++counters_.eventWait;
}

void CudaDeviceContext::syncWorkStreams()
{
  CUDA_CALL(cudaSetDevice(deviceId_));
  if (serialCuda_)
  {
    CUDA_CALL(cudaStreamSynchronize(cudaStreamLegacy));
    ++counters_.streamSync;
    reclaimRetiredEvents();
    return;
  }
  for (auto const& slot : workSlots_)
  {
    CUDA_CALL(cudaStreamSynchronize(slot.stream));
    ++counters_.streamSync;
  }
  reclaimRetiredEvents();
}

void CudaDeviceContext::syncMemoryStream()
{
  CUDA_CALL(cudaSetDevice(deviceId_));
  CUDA_CALL(cudaStreamSynchronize(memoryStream_));
  ++counters_.streamSync;
  reclaimRetiredEvents();
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

  printCounters();

  for (auto event : retiredEvents_)
  {
    CUDA_CALL(cudaEventDestroy(event));
    ++counters_.eventDestroy;
  }
  retiredEvents_.clear();

  for (auto event : freeEvents_)
  {
    CUDA_CALL(cudaEventDestroy(event));
    ++counters_.eventDestroy;
  }
  freeEvents_.clear();

  if (!serialCuda_ && memoryStream_ != nullptr)
  {
    CUDA_CALL(cudaStreamDestroy(memoryStream_));
  }
  memoryStream_ = nullptr;
}

void CudaDeviceContext::reclaimRetiredEvents()
{
  freeEvents_.insert(freeEvents_.end(), retiredEvents_.begin(), retiredEvents_.end());
  retiredEvents_.clear();
}

void CudaDeviceContext::printCounters() const
{
  if (!logCounters_)
  {
    return;
  }
  std::fprintf(stderr,
               "[TENSORCONTRACTION][CUDA_COUNTERS] Device=%d EventCreate=%llu EventRecord=%llu EventWait=%llu "
               "EventDestroy=%llu StreamSync=%llu EventPoolFree=%zu\n",
               deviceId_, static_cast<unsigned long long>(counters_.eventCreate),
               static_cast<unsigned long long>(counters_.eventRecord),
               static_cast<unsigned long long>(counters_.eventWait),
               static_cast<unsigned long long>(counters_.eventDestroy),
               static_cast<unsigned long long>(counters_.streamSync), freeEvents_.size());
}

} // namespace tensor
