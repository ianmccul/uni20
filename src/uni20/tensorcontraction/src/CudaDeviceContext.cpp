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

CudaDeviceContext::ScratchLease::ScratchLease(ScratchLease&& other) noexcept
    : context_(other.context_), buffer_(std::move(other.buffer_)), stream_(other.stream_)
{
  other.context_ = nullptr;
  other.stream_ = nullptr;
}

CudaDeviceContext::ScratchLease& CudaDeviceContext::ScratchLease::operator=(ScratchLease&& other) noexcept
{
  if (this != &other)
  {
    release();
    context_ = other.context_;
    buffer_ = std::move(other.buffer_);
    stream_ = other.stream_;
    other.context_ = nullptr;
    other.stream_ = nullptr;
  }
  return *this;
}

CudaDeviceContext::ScratchLease::~ScratchLease() { release(); }

void CudaDeviceContext::ScratchLease::release()
{
  if (context_ != nullptr && buffer_ != nullptr)
  {
    context_->releaseScratch(std::move(buffer_), stream_);
  }
  context_ = nullptr;
  stream_ = nullptr;
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

CudaDeviceContext::ScratchLease CudaDeviceContext::acquireScratch(std::size_t bytes, cudaStream_t stream)
{
  CUDA_CALL(cudaSetDevice(deviceId_));
  std::shared_ptr<ScratchBuffer> buffer;
  {
    std::lock_guard<std::mutex> lock(scratchMutex_);
    auto candidate =
        std::find_if(freeScratchBuffers_.begin(), freeScratchBuffers_.end(), [bytes, stream](auto const& item) {
          return item->bytes >= bytes && (!item->readyEventRecorded || item->readyStream == stream);
        });
    if (candidate == freeScratchBuffers_.end())
    {
      candidate = std::find_if(freeScratchBuffers_.begin(), freeScratchBuffers_.end(),
                               [bytes](auto const& item) { return item->bytes >= bytes; });
    }
    if (candidate != freeScratchBuffers_.end())
    {
      buffer = std::move(*candidate);
      freeScratchBuffers_.erase(candidate);
    }
  }

  if (buffer == nullptr)
  {
    buffer = std::make_shared<ScratchBuffer>();
    buffer->bytes = bytes;
    CUDA_CALL(cudaMalloc(&buffer->ptr, bytes));
    CUDA_CALL(cudaEventCreateWithFlags(&buffer->readyEvent, cudaEventDisableTiming));
    std::lock_guard<std::mutex> lock(scratchMutex_);
    allScratchBuffers_.push_back(buffer);
  }

  if (buffer->readyEventRecorded && buffer->readyStream != stream)
  {
    waitEvent(stream, buffer->readyEvent);
  }
  buffer->readyEventRecorded = false;
  buffer->readyStream = nullptr;
  return ScratchLease(*this, std::move(buffer), stream);
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

  for (auto& buffer : allScratchBuffers_)
  {
    if (buffer->readyEvent != nullptr)
    {
      CUDA_CALL(cudaEventDestroy(buffer->readyEvent));
      buffer->readyEvent = nullptr;
    }
    if (buffer->ptr != nullptr)
    {
      CUDA_CALL(cudaFree(buffer->ptr));
      buffer->ptr = nullptr;
    }
  }
  freeScratchBuffers_.clear();
  allScratchBuffers_.clear();

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

void CudaDeviceContext::releaseScratch(std::shared_ptr<ScratchBuffer> buffer, cudaStream_t stream)
{
  if (buffer == nullptr)
  {
    return;
  }
  CUDA_CALL(cudaSetDevice(deviceId_));
  if (!serialCuda_)
  {
    CUDA_CALL(cudaEventRecord(buffer->readyEvent, stream));
    buffer->readyEventRecorded = true;
    buffer->readyStream = stream;
  }
  std::lock_guard<std::mutex> lock(scratchMutex_);
  freeScratchBuffers_.push_back(std::move(buffer));
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
