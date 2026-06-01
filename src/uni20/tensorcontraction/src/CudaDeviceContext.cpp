#include "CudaDeviceContext.hpp"

#include "Utils.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <map>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensor
{

static_assert(!std::is_copy_constructible_v<CudaDeviceContext::StreamLease>);
static_assert(!std::is_copy_assignable_v<CudaDeviceContext::StreamLease>);
static_assert(std::is_move_constructible_v<CudaDeviceContext::StreamLease>);
static_assert(std::is_move_assignable_v<CudaDeviceContext::StreamLease>);

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

CudaDeviceContext::StreamLease::StreamLease(StreamLease&& other) noexcept
    : context_(other.context_), stream_(other.stream_)
{
  other.context_ = nullptr;
  other.stream_ = nullptr;
}

CudaDeviceContext::StreamLease& CudaDeviceContext::StreamLease::operator=(StreamLease&& other) noexcept
{
  if (this != &other)
  {
    release();
    context_ = other.context_;
    stream_ = other.stream_;
    other.context_ = nullptr;
    other.stream_ = nullptr;
  }
  return *this;
}

CudaDeviceContext::StreamLease::~StreamLease() { release(); }

void CudaDeviceContext::StreamLease::release()
{
  if (context_ != nullptr && stream_ != nullptr)
  {
    context_->returnWorkStream(stream_);
  }
  context_ = nullptr;
  stream_ = nullptr;
}

CudaDeviceContext::GpuEvent::GpuEvent(CudaDeviceContext& context, cudaStream_t producerStream)
    : context_(&context), event_(context.recordDependencyEvent(producerStream)), producerStream_(producerStream),
      publishSequence_(event_->sequence)
{}

CudaDeviceContext::GpuEvent::GpuEvent(GpuEvent&& other) noexcept
    : context_(other.context_), event_(std::move(other.event_)), producerStream_(other.producerStream_),
      publishSequence_(other.publishSequence_)
{
  other.context_ = nullptr;
  other.producerStream_ = nullptr;
  other.publishSequence_ = 0;
}

CudaDeviceContext::GpuEvent& CudaDeviceContext::GpuEvent::operator=(GpuEvent&& other) noexcept
{
  if (this != &other)
  {
    context_ = other.context_;
    event_ = std::move(other.event_);
    producerStream_ = other.producerStream_;
    publishSequence_ = other.publishSequence_;
    other.context_ = nullptr;
    other.producerStream_ = nullptr;
    other.publishSequence_ = 0;
  }
  return *this;
}

cudaStream_t CudaDeviceContext::GpuEvent::stream() const noexcept { return producerStream_; }

void CudaDeviceContext::GpuEvent::waitOn(cudaStream_t consumerStream)
{
  if (context_ != nullptr && event_ != nullptr) context_->waitEvent(consumerStream, event_->event);
}

CudaDeviceContext::EventDependencyRef CudaDeviceContext::GpuEvent::dependencyEvent() { return event_; }

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

  for (int i = 0; i < workStreamCount; ++i)
  {
    cudaStream_t stream = serialCuda_ ? cudaStreamLegacy : nullptr;
    if (!serialCuda_)
    {
      CUDA_CALL(cudaStreamCreate(&stream));
      ++createdWorkStreamCount_;
    }
    availableWorkStreams_.push_back(stream);
  }
}

std::shared_ptr<CudaDeviceContext> CudaDeviceContext::shared(int deviceId, int workStreamCount, bool serialCuda)
{
  using Key = std::tuple<int, int, bool>;
  static std::mutex mutex;
  static std::map<Key, std::shared_ptr<CudaDeviceContext>> contexts;

  Key key{deviceId, serialCuda ? 1 : std::max(1, workStreamCount), serialCuda};
  std::lock_guard<std::mutex> lock(mutex);
  if (auto const it = contexts.find(key); it != contexts.end())
  {
    return it->second;
  }

  auto created = std::make_shared<CudaDeviceContext>(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  contexts[key] = created;
  return created;
}

CudaDeviceContext::~CudaDeviceContext() { release(); }

CudaDeviceContext::StreamLease CudaDeviceContext::leaseWorkStream(cudaStream_t preferredStream)
{
  return StreamLease(*this, acquireWorkStream(preferredStream));
}

CudaDeviceContext::GpuEventRef CudaDeviceContext::recordCompletionEvent(cudaStream_t producerStream)
{
  assert(producerStream != nullptr);
  return std::make_shared<GpuEvent>(*this, producerStream);
}

cudaStream_t CudaDeviceContext::acquireWorkStream(cudaStream_t preferredStream)
{
  CUDA_CALL(cudaSetDevice(deviceId_));

  if (preferredStream == cudaStreamLegacy)
  {
    return cudaStreamLegacy;
  }

  if (preferredStream != nullptr)
  {
    auto it = std::find(availableWorkStreams_.begin(), availableWorkStreams_.end(), preferredStream);
    if (it != availableWorkStreams_.end())
    {
      auto stream = *it;
      availableWorkStreams_.erase(it);
      return stream;
    }
  }

  if (!availableWorkStreams_.empty())
  {
    auto stream = availableWorkStreams_.front();
    availableWorkStreams_.pop_front();
    return stream;
  }

  // This first prototype does not repossess active leases. A future
  // idle-aware scheduler should wait for or co_await an idle stream slot.
  assert(false && "no unleased CUDA work stream available");
  return cudaStreamLegacy;
}

void CudaDeviceContext::returnWorkStream(cudaStream_t stream)
{
  if (stream != nullptr && stream != cudaStreamLegacy)
  {
    availableWorkStreams_.push_back(stream);
  }
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
  return std::make_shared<EventDependency>(*this, recordEvent(stream), ++dependencySequence_);
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

void CudaDeviceContext::enqueueAsyncFree(void* ptr, cudaStream_t stream, std::vector<EventDependencyRef> dependencies)
{
  if (ptr == nullptr)
  {
    return;
  }
  CUDA_CALL(cudaSetDevice(deviceId_));
  for (auto const& dependency : dependencies)
  {
    if (dependency != nullptr)
    {
      waitEvent(stream, dependency->event);
    }
  }
  CUDA_CALL(cudaFreeAsync(ptr, stream));
  ++counters_.asyncFree;
  if (dependencies.empty())
  {
    return;
  }
  auto completeEvent = recordEvent(stream);
  pendingFrees_.push_back(PendingFree{ptr, completeEvent, std::move(dependencies)});
  reclaimCompletedAsyncFrees();
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

cublasHandle_t CudaDeviceContext::cublasHandleForCurrentThread(cudaStream_t stream)
{
  struct ThreadLocalHandle
  {
      int deviceId = -1;
      cublasHandle_t handle = nullptr;

      ThreadLocalHandle() = default;
      ThreadLocalHandle(ThreadLocalHandle const&) = delete;
      ThreadLocalHandle& operator=(ThreadLocalHandle const&) = delete;
      ThreadLocalHandle(ThreadLocalHandle&& other) noexcept : deviceId(other.deviceId), handle(other.handle)
      {
        other.deviceId = -1;
        other.handle = nullptr;
      }
      ThreadLocalHandle& operator=(ThreadLocalHandle&& other) noexcept
      {
        if (this != &other)
        {
          destroy();
          deviceId = other.deviceId;
          handle = other.handle;
          other.deviceId = -1;
          other.handle = nullptr;
        }
        return *this;
      }
      ~ThreadLocalHandle() { destroy(); }

      void destroy()
      {
        if (handle != nullptr)
        {
          CUDA_CALL(cudaSetDevice(deviceId));
          CUBLAS_CALL(cublasDestroy(handle));
          handle = nullptr;
        }
      }
  };

  thread_local std::vector<ThreadLocalHandle> handles;
  auto const index = static_cast<std::size_t>(deviceId_);
  if (handles.size() <= index)
  {
    handles.resize(index + 1);
  }
  auto& cached = handles[index];
  if (cached.handle == nullptr)
  {
    CUDA_CALL(cudaSetDevice(deviceId_));
    cached.deviceId = deviceId_;
    CUBLAS_CALL(cublasCreate(&cached.handle));
  }
  else
  {
    CUDA_CALL(cudaSetDevice(deviceId_));
  }
  CUBLAS_CALL(cublasSetStream(cached.handle, stream));
  return cached.handle;
}

void CudaDeviceContext::syncWorkStreams(const char* reason)
{
  CUDA_CALL(cudaSetDevice(deviceId_));
  if (serialCuda_)
  {
    CUDA_CALL(cudaStreamSynchronize(cudaStreamLegacy));
    countStreamSync(reason);
    reclaimRetiredEvents();
    return;
  }
  for (auto const stream : availableWorkStreams_)
  {
    CUDA_CALL(cudaStreamSynchronize(stream));
    countStreamSync(reason);
  }
  reclaimCompletedAsyncFrees();
  reclaimRetiredEvents();
}

void CudaDeviceContext::syncMemoryStream(const char* reason)
{
  CUDA_CALL(cudaSetDevice(deviceId_));
  CUDA_CALL(cudaStreamSynchronize(memoryStream_));
  countStreamSync(reason);
  reclaimCompletedAsyncFrees();
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
  syncWorkStreams("device_context_release_work_streams");
  syncMemoryStream("device_context_release_memory_stream");

  assert(serialCuda_ || availableWorkStreams_.size() == createdWorkStreamCount_);
  while (!availableWorkStreams_.empty())
  {
    auto stream = availableWorkStreams_.front();
    availableWorkStreams_.pop_front();
    if (!serialCuda_ && stream != nullptr)
    {
      CUDA_CALL(cudaStreamDestroy(stream));
    }
  }
  createdWorkStreamCount_ = 0;

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

void CudaDeviceContext::reclaimCompletedAsyncFrees()
{
  CUDA_CALL(cudaSetDevice(deviceId_));
  std::vector<PendingFree> stillPending;
  stillPending.reserve(pendingFrees_.size());
  for (auto& pending : pendingFrees_)
  {
    cudaError_t status = cudaEventQuery(pending.completeEvent);
    if (status == cudaSuccess)
    {
      retireEvent(pending.completeEvent);
      pending.completeEvent = nullptr;
      pending.dependencies.clear();
      ++counters_.asyncFreeReclaim;
    }
    else if (status == cudaErrorNotReady)
    {
      stillPending.push_back(std::move(pending));
      ++counters_.asyncFreePoll;
    }
    else
    {
      CUDA_CALL(status);
    }
  }
  pendingFrees_ = std::move(stillPending);
}

void CudaDeviceContext::countStreamSync(const char* reason)
{
  ++counters_.streamSync;
  if (reason != nullptr)
  {
    ++counters_.streamSyncByReason[reason];
  }
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
  std::fprintf(
      stderr,
      "[TENSORCONTRACTION][CUDA_COUNTERS] Device=%d EventCreate=%llu EventRecord=%llu EventWait=%llu "
      "EventDestroy=%llu StreamSync=%llu EventPoolFree=%zu AsyncFree=%llu AsyncFreeReclaim=%llu "
      "AsyncFreePoll=%llu\n",
      deviceId_, static_cast<unsigned long long>(counters_.eventCreate),
      static_cast<unsigned long long>(counters_.eventRecord), static_cast<unsigned long long>(counters_.eventWait),
      static_cast<unsigned long long>(counters_.eventDestroy), static_cast<unsigned long long>(counters_.streamSync),
      freeEvents_.size(), static_cast<unsigned long long>(counters_.asyncFree),
      static_cast<unsigned long long>(counters_.asyncFreeReclaim),
      static_cast<unsigned long long>(counters_.asyncFreePoll));
  for (auto const& [reason, count] : counters_.streamSyncByReason)
  {
    std::fprintf(stderr, "[TENSORCONTRACTION][CUDA_SYNC_REASON] Device=%d Reason=%s Count=%llu\n", deviceId_,
                 reason.c_str(), static_cast<unsigned long long>(count));
  }
}

} // namespace tensor
