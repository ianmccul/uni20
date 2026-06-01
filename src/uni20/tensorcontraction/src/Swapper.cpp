#include <mpi.h>

#include <cassert>
#include <cstdlib>
#include <functional>
#include <unordered_set>

#include "Calculator.hpp"
#include "Debug.hpp"
#include "Swapper.hpp"
#include "Utils.h"

namespace tensor
{

GpuBuffer::GpuBuffer(GpuBuffer&& other)
    : ptr(other.ptr), id(other.id), dim1(other.dim1), dim2(other.dim2), hostPtr(other.hostPtr),
      hasValidContent(other.hasValidContent), dependencyEventsEnabled(other.dependencyEventsEnabled),
      deviceContext(other.deviceContext), accessState(std::move(other.accessState))
{
  other.ptr = nullptr;
  other.dim1 = 0;
  other.dim2 = 0;
  other.hostPtr = nullptr;
  other.hasValidContent = false;
  other.dependencyEventsEnabled = false;
  other.deviceContext = nullptr;
  other.accessState = {};
}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other)
{
  if (this != &other)
  {
    ptr = other.ptr;
    id = other.id;
    dim1 = other.dim1;
    dim2 = other.dim2;
    hostPtr = other.hostPtr;
    hasValidContent = other.hasValidContent;
    dependencyEventsEnabled = other.dependencyEventsEnabled;
    deviceContext = other.deviceContext;
    accessState = std::move(other.accessState);

    // Reset the moved-from object to a safe state
    other.ptr = nullptr;
    other.dim1 = 0;
    other.dim2 = 0;
    other.hostPtr = nullptr;
    other.hasValidContent = false;
    other.dependencyEventsEnabled = false;
    other.deviceContext = nullptr;
    other.accessState = {};
  }
  return *this;
}

GpuBuffer::GpuBuffer(void* ptr, Matrix mat, bool dependencyEventsEnabled, CudaDeviceContext& deviceContext)
    : ptr(ptr), id(mat.getId()), dim1(mat.getFirstDim()), dim2(mat.getSecondDim()), hostPtr(mat.getPtr()),
      dependencyEventsEnabled(dependencyEventsEnabled), deviceContext(&deviceContext)
{}

double* GpuBuffer::getPtr() { return static_cast<double*>(ptr); }
int GpuBuffer::getId() const { return id; }

void GpuBuffer::publishAllocation(cuda::CompletionRef completion)
{
  if (!dependencyEventsEnabled)
  {
    return;
  }
  accessState.writer.completion = std::move(completion);
}

void GpuBuffer::waitBeforeWrite(cudaStream_t stream)
{
  if (!dependencyEventsEnabled)
  {
    return;
  }
  if (accessState.readers.completion != nullptr)
  {
    accessState.readers.completion->waitOn(stream);
  }
}

void GpuBuffer::publishRead(cuda::CompletionRef completion)
{
  if (!dependencyEventsEnabled)
  {
    return;
  }
  accessState.readers.completion = std::move(completion);
}

void GpuBuffer::publishRead(cudaStream_t stream)
{
  if (!dependencyEventsEnabled)
  {
    return;
  }
  publishRead(deviceContext->recordCompletionEvent(stream));
}

void GpuBuffer::waitBeforeRead(cudaStream_t stream)
{
  if (!dependencyEventsEnabled)
  {
    return;
  }
  if (accessState.writer.completion != nullptr)
  {
    accessState.writer.completion->waitOn(stream);
  }
}

void GpuBuffer::publishWrite(cuda::CompletionRef completion)
{
  hasValidContent = true;
  if (!dependencyEventsEnabled)
  {
    return;
  }
  accessState.writer.completion = completion;
  accessState.readers.completion = std::move(completion);
}

void GpuBuffer::publishWrite(cudaStream_t stream)
{
  if (!dependencyEventsEnabled)
  {
    return;
  }
  publishWrite(deviceContext->recordCompletionEvent(stream));
}

Swapper::Swapper()
{
  int visibleDeviceCount = 0;
  cudaGetDeviceCount(&visibleDeviceCount);
  deviceCount = resolveActiveCudaDeviceCount(visibleDeviceCount);
  serialCuda = envFlagEnabled("UNI20_TENSORCONTRACTION_SERIAL_CUDA");
  dependencyEventsEnabled = !serialCuda;
  const int workStreamCount = resolveTensorContractionStreamCount();
  deviceContexts.reserve(deviceCount);
  for (int i = 0; i < deviceCount; i++)
  {
    deviceContexts.push_back(CudaDeviceContext::shared(i, workStreamCount, serialCuda));
    hostToGpuMaps.emplace_back();
    pinnedMatrix.emplace_back();
  }
  CUDA_CALL(cudaSetDevice(0));
}

Swapper::~Swapper() { release(); }

void Swapper::pinMatrix(Matrix mat, int deviceId) { pinnedMatrix[deviceId].insert(mat.getId()); }

void Swapper::unpinMatrix(Matrix mat, int deviceId) { pinnedMatrix[deviceId].erase(mat.getId()); }

bool Swapper::isPinned(int id, int deviceId) { return pinnedMatrix[deviceId].find(id) != pinnedMatrix[deviceId].end(); }

#if DEBUG_LOG
void Swapper::setMatrixName(Matrix mat, const std::string& name) { matrixNames[mat.getId()] = name; }

std::string Swapper::getMatrixName(int matId) const
{
  auto it = matrixNames.find(matId);
  if (it != matrixNames.end())
  {
    return it->second;
  }
  return std::to_string(matId); // Fall back to ID if no name set
}

#endif

std::pair<int, std::shared_ptr<GpuBuffer>> Swapper::getPreStoreBufferOrNone(Matrix mat)
{
  const auto it = preStoreMap.find(mat.getId());
  if (it == preStoreMap.end()) return {-1, nullptr};

  return it->second;
}

void Swapper::notifyMatrixRead(Matrix mat, int deviceId, cuda::CompletionRef completion)
{
  auto buffer = getGpuBufferOrNone(mat, deviceId);
  if (buffer != nullptr)
  {
    buffer->publishRead(completion);
  }
}

void Swapper::notifyMatrixWrite(Matrix mat, int deviceId, cuda::CompletionRef completion)
{
  auto buffer = getGpuBufferOrNone(mat, deviceId);
  if (buffer != nullptr)
  {
    buffer->publishWrite(completion);
  }
}

void Swapper::exchangePreStoreMap()
{
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  int device_count;
  CUDA_CALL(cudaGetDeviceCount(&device_count));

  std::vector<std::pair<int, int>> nccl_mat_id_pair;

  for (const auto& item : preStoreMap)
  {
    const int mat_id = item.first;
    const int device_id = item.second.first;
    const int nccl_comm_id = mpi_rank * device_count + device_id;
    nccl_mat_id_pair.push_back({nccl_comm_id, mat_id});
  }

  // 1. use MPI_Allgather to share nccl_mat_id_pair across all nodes.
  int local_count = nccl_mat_id_pair.size();

  std::vector<int> all_counts(mpi_size);
  MPI_Allgather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> recvcounts(mpi_size);
  std::vector<int> displs(mpi_size);
  int total_pairs = 0;
  for (int i = 0; i < mpi_size; i++)
  {
    recvcounts[i] = all_counts[i] * 2;
    displs[i] = total_pairs * 2;
    total_pairs += all_counts[i];
  }

  std::vector<int> sendbuf(local_count * 2);
  for (int i = 0; i < local_count; i++)
  {
    sendbuf[i * 2] = nccl_mat_id_pair[i].first;
    sendbuf[i * 2 + 1] = nccl_mat_id_pair[i].second;
  }

  std::vector<int> recvbuf(total_pairs * 2);
  MPI_Allgatherv(sendbuf.data(), local_count * 2, MPI_INT, recvbuf.data(), recvcounts.data(), displs.data(), MPI_INT,
                 MPI_COMM_WORLD);

  // 2. Update Swapper::matToNCCLIdMap. Note: only update with mat on the other
  // nodes.
  for (int rank = 0; rank < mpi_size; rank++)
  {
    if (rank == mpi_rank) continue;
    int base = displs[rank];
    for (int i = 0; i < all_counts[rank]; i++)
    {
      int nccl_comm_id = recvbuf[base + i * 2];
      int mat_id = recvbuf[base + i * 2 + 1];
      matToNCCLIdMap[mat_id] = nccl_comm_id;
    }
  }
}

bool Swapper::isOnRemoteGpu(int matId) const { return matToNCCLIdMap.find(matId) != matToNCCLIdMap.end(); }

int Swapper::getRemoteNCCLId(int matId) const
{
  assert(isOnRemoteGpu(matId));
  return matToNCCLIdMap.at(matId);
}

std::shared_ptr<GpuBuffer> Swapper::getGpuBufferOrNone(Matrix mat, int deviceId)
{
  auto [id, buffer] = getPreStoreBufferOrNone(mat);
  if (buffer != nullptr && id == deviceId)
  {
    return buffer;
  }

  auto& hostToGpuMap = hostToGpuMaps[deviceId];

  auto it = hostToGpuMap.find(mat.getId());
  if (it == hostToGpuMap.end()) return nullptr;
  return it->second;
}

std::pair<int, std::shared_ptr<GpuBuffer>> Swapper::findLocalSourceBuffer(Matrix mat, int requesterDeviceId)
{
  auto [preStoreDeviceId, preStoreBuffer] = getPreStoreBufferOrNone(mat);
  if (preStoreBuffer != nullptr && preStoreBuffer->contentValid())
  {
    return {preStoreDeviceId, preStoreBuffer};
  }

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
  {
    if (deviceId == requesterDeviceId)
    {
      continue;
    }
    auto const& hostToGpuMap = hostToGpuMaps[deviceId];
    auto it = hostToGpuMap.find(mat.getId());
    if (it != hostToGpuMap.end() && it->second->contentValid())
    {
      return {deviceId, it->second};
    }
  }

  return {-1, nullptr};
}

void Swapper::freeAllUnpinMatrices(int deviceId)
{
  std::vector<std::shared_ptr<GpuBuffer>> buffers;

  for (auto [id, buffer] : hostToGpuMaps[deviceId])
  {
    if (!isPinned(id, deviceId))
    {
      buffers.push_back(buffer);
    }
  }

  for (auto buffer : buffers)
  {
    hostToGpuMaps[deviceId].erase(buffer->getId());
    freeBuffer(buffer, deviceId);
  }
}

void Swapper::release()
{
  if (released)
  {
    return;
  }
  released = true;

  clear();

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
  {
    CUDA_CALL(cudaSetDevice(deviceId));
    if (deviceId < static_cast<int>(memPools.size()))
    {
      deviceContexts[deviceId]->syncMemoryStream("swapper_release_before_mempool_trim");
      CUDA_CALL(cudaMemPoolTrimTo(memPools[deviceId], 0));
      if (ownsMemPool[deviceId])
      {
        CUDA_CALL(cudaMemPoolDestroy(memPools[deviceId]));
      }
    }
  }
  memPools.clear();
  ownsMemPool.clear();
  deviceContexts.clear();
  CUDA_CALL(cudaSetDevice(0));
}

std::shared_ptr<GpuBuffer> Swapper::allocate(Matrix mat, int deviceId)
{
  auto memPool = memPools[deviceId];

  auto allocation = deviceContexts[deviceId]->allocateFromPool(memPool, mat.sizeInByte());

  if (allocation.status != cudaSuccess)
  {
    if (allocation.status != cudaErrorMemoryAllocation)
    {
      fprintf(stderr, "Unexpected CUDA Error at %s:%d\n", __FILE__, __LINE__);
      fprintf(stderr, "  Error code: %d\n", allocation.status);
      fprintf(stderr, "  Error text: %s\n", cudaGetErrorString(allocation.status));
      assert(false);
    }

    return nullptr;
  }

  DEBUG_GPU_ALLOC(*this, mat.getId(), deviceId, allocation.stream, mat.sizeInByte());
  auto buffer = std::make_shared<GpuBuffer>(allocation.ptr, mat, dependencyEventsEnabled, *deviceContexts[deviceId]);
  auto& hostToGpuMap = hostToGpuMaps[deviceId];
  hostToGpuMap[buffer->getId()] = buffer;

  buffer->publishAllocation(std::move(allocation.completion));
  return buffer;
}

void Swapper::copyMatrix(Matrix mat, std::shared_ptr<GpuBuffer> buffer, int deviceId, cudaStream_t stream)
{
  assert(mat.getPtr() != nullptr);
  DEBUG_GPU_COPY_H2D(*this, mat.getId(), deviceId, stream, mat.sizeInByte());
  CUDA_CALL(cudaMemcpyAsync(buffer->getPtr(), mat.getPtr(), mat.sizeInByte(), cudaMemcpyHostToDevice, stream));
  buffer->publishWrite(stream);
}

void Swapper::preStoreMatrix(Matrix mat, int deviceId)
{
  DEBUG_PRESTORE(*this, mat.getId(), 0, deviceId);
  auto allocation = deviceContexts[deviceId]->allocateFromPool(memPools[deviceId], mat.sizeInByte());
  CUDA_CALL(allocation.status);
  auto buffer = std::make_shared<GpuBuffer>(allocation.ptr, mat, dependencyEventsEnabled, *deviceContexts[deviceId]);
  buffer->publishAllocation(std::move(allocation.completion));

  auto stream = deviceContexts[deviceId]->leaseWorkStream();
  buffer->waitBeforeWrite(stream.stream());
  DEBUG_GPU_COPY_H2D(*this, mat.getId(), deviceId, stream.stream(), mat.sizeInByte());
  CUDA_CALL(cudaMemcpyAsync(buffer->getPtr(), mat.getPtr(), mat.sizeInByte(), cudaMemcpyHostToDevice, stream.stream()));
  buffer->publishWrite(stream.recordCompletion());
  assert(preStoreMap.find(mat.getId()) == preStoreMap.end());
  preStoreMap[mat.getId()] = std::make_pair(deviceId, buffer);
}

void Swapper::copyHostToPreStoreMatrix(Matrix mat)
{
  // Refresh an existing resident buffer from the host-side MatrixFamily block.
  // This is used only at explicit host/GPU authority boundaries.
  auto [deviceId, buffer] = getPreStoreBufferOrNone(mat);
  assert(buffer != nullptr);
  assert(mat.getPtr() != nullptr);

  CUDA_CALL(cudaSetDevice(deviceId));
  auto stream = deviceContexts[deviceId]->leaseWorkStream();
  buffer->waitBeforeWrite(stream.stream());
  DEBUG_GPU_COPY_H2D(*this, mat.getId(), deviceId, stream.stream(), mat.sizeInByte());
  CUDA_CALL(cudaMemcpyAsync(buffer->getPtr(), mat.getPtr(), mat.sizeInByte(), cudaMemcpyHostToDevice, stream.stream()));
  buffer->publishWrite(stream.recordCompletion());
}

void Swapper::copyPreStoreMatrixToHost(Matrix mat)
{
  // Materialize resident GPU data back into the MatrixFamily host block without
  // going through a worklist SyncWork for every vector operation.
  auto [deviceId, buffer] = getPreStoreBufferOrNone(mat);
  assert(buffer != nullptr);
  assert(mat.getPtr() != nullptr);

  CUDA_CALL(cudaSetDevice(deviceId));
  auto stream = deviceContexts[deviceId]->leaseWorkStream();
  buffer->waitBeforeRead(stream.stream());
  DEBUG_GPU_COPY_D2H(*this, mat.getId(), deviceId, stream.stream(), mat.sizeInByte());
  CUDA_CALL(cudaMemcpyAsync(mat.getPtr(), buffer->getPtr(), mat.sizeInByte(), cudaMemcpyDeviceToHost, stream.stream()));
  buffer->publishRead(stream.recordCompletion());
}

void Swapper::registerGpuAllocation(Matrix mat, int deviceId)
{
  CUDA_CALL(cudaSetDevice(deviceId));
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  DEBUG_PRESTORE(*this, mat.getId(), mpi_rank, deviceId);
  auto allocation = deviceContexts[deviceId]->allocateFromPool(memPools[deviceId], mat.sizeInByte());
  CUDA_CALL(allocation.status);
  auto buffer = std::make_shared<GpuBuffer>(allocation.ptr, mat, dependencyEventsEnabled, *deviceContexts[deviceId]);
  buffer->publishAllocation(std::move(allocation.completion));
  assert(preStoreMap.find(mat.getId()) == preStoreMap.end());
  preStoreMap[mat.getId()] = std::make_pair(deviceId, buffer);
}

std::shared_ptr<GpuBuffer> Swapper::getForRead(Matrix mat, int deviceId, cudaStream_t stream)
{
  auto bufferPtr = getForReadNoWait(mat, deviceId);
  if (bufferPtr != nullptr)
  {
    bufferPtr->waitBeforeRead(stream);
    return bufferPtr;
  }

  return nullptr;
}

std::shared_ptr<GpuBuffer> Swapper::getForWrite(Matrix mat, int deviceId, cudaStream_t stream)
{
  auto bufferPtr = getForWriteNoWait(mat, deviceId);
  if (bufferPtr != nullptr)
  {
    bufferPtr->waitBeforeRead(stream);
    bufferPtr->waitBeforeWrite(stream);
    return bufferPtr;
  }

  return nullptr;
}

std::shared_ptr<GpuBuffer> Swapper::getForReadNoWait(Matrix mat, int deviceId)
{
  return getGpuBufferOrNone(mat, deviceId);
}

std::shared_ptr<GpuBuffer> Swapper::getForWriteNoWait(Matrix mat, int deviceId)
{
  return getGpuBufferOrNone(mat, deviceId);
}

Swapper::GpuAccessPlan::GpuAccessPlan(Swapper& swapper, int deviceId,
                                      std::vector<std::shared_ptr<GpuBuffer>> readBuffers,
                                      std::vector<std::shared_ptr<GpuBuffer>> writeBuffers)
    : swapper(swapper), deviceId(deviceId), streamOwner(swapper.deviceContext(deviceId).leaseWorkStream()),
      readBuffers(std::move(readBuffers)), writeBuffers(std::move(writeBuffers))
{
  swapper.waitForAccessDependencies(this->readBuffers, this->writeBuffers, this->stream());
}

Swapper::GpuAccessPlan::~GpuAccessPlan()
{
  if (swapper.dependencyEventsActive())
  {
    this->publishCompletion(this->recordCompletion());
  }
}

cublasHandle_t Swapper::GpuAccessPlan::handle() const { return streamOwner.prepare_handle(); }

cuda::CompletionRef Swapper::GpuAccessPlan::recordCompletion() const { return streamOwner.recordCompletion(); }

void Swapper::GpuAccessPlan::publishCompletion(cuda::CompletionRef completion) const
{
  swapper.publishAccessCompletion(readBuffers, writeBuffers, this->stream(), std::move(completion));
}

auto Swapper::createAccessPlan(std::vector<std::shared_ptr<GpuBuffer>> readBuffers,
                               std::vector<std::shared_ptr<GpuBuffer>> writeBuffers, int deviceId) -> GpuAccessPlan
{
  return GpuAccessPlan(*this, deviceId, std::move(readBuffers), std::move(writeBuffers));
}

void Swapper::waitForAccessDependencies(const std::vector<std::shared_ptr<GpuBuffer>>& readBuffers,
                                        const std::vector<std::shared_ptr<GpuBuffer>>& writeBuffers,
                                        cudaStream_t stream)
{
  if (!dependencyEventsEnabled)
  {
    return;
  }

  struct ProducerDependency
  {
      cudaStream_t stream = nullptr;
      cuda::CompletionRef completion;
  };

  std::vector<ProducerDependency> dependencies;
  auto addDependency = [&](cuda::CompletionRef completion) {
    if (completion == nullptr || completion->stream() == stream)
    {
      return;
    }
    auto producerStream = completion->stream();
    for (auto& dependency : dependencies)
    {
      if (dependency.stream == producerStream)
      {
        // One wait on the newest completion from a producer stream covers
        // earlier completions by CUDA stream order.
        if (completion->sequence() > dependency.completion->sequence())
        {
          dependency.completion = std::move(completion);
        }
        return;
      }
    }
    dependencies.push_back(ProducerDependency{producerStream, std::move(completion)});
  };

  for (auto const& buffer : readBuffers)
  {
    if (buffer == nullptr)
    {
      continue;
    }
    addDependency(buffer->accessState.writer.completion);
  }

  for (auto const& buffer : writeBuffers)
  {
    if (buffer == nullptr)
    {
      continue;
    }
    addDependency(buffer->accessState.writer.completion);
    addDependency(buffer->accessState.readers.completion);
  }

  std::unordered_set<cudaEvent_t> waitedEvents;
  for (auto const& dependency : dependencies)
  {
    auto event = dependency.completion->dependencyEvent();
    if (event != nullptr && waitedEvents.insert(event->event).second)
    {
      event->context->waitEvent(stream, event->event);
    }
  }
}

void Swapper::publishAccessCompletion(const std::vector<std::shared_ptr<GpuBuffer>>& readBuffers,
                                      const std::vector<std::shared_ptr<GpuBuffer>>& writeBuffers, cudaStream_t stream,
                                      cuda::CompletionRef completion)
{
  if (!dependencyEventsEnabled)
  {
    return;
  }
  (void)stream;
  for (auto const& buffer : readBuffers)
  {
    if (buffer != nullptr)
    {
      buffer->publishRead(completion);
    }
  }
  for (auto const& buffer : writeBuffers)
  {
    if (buffer != nullptr)
    {
      buffer->publishWrite(completion);
    }
  }
}

void Swapper::freeAllBuffer(int deviceId)
{
  auto& map = hostToGpuMaps[deviceId];
  for (auto [id, buffer] : map)
  {
    freeBuffer(buffer, deviceId);
  }
  map.clear();
}

void Swapper::freeBuffer(std::shared_ptr<GpuBuffer> buffer, int deviceId)
{
  std::vector<CudaDeviceContext::EventDependencyRef> dependencies;
  dependencies.reserve(2);
  if (buffer->accessState.writer.completion != nullptr)
  {
    dependencies.push_back(buffer->accessState.writer.completion->dependencyEvent());
  }
  if (buffer->accessState.readers.completion != nullptr &&
      buffer->accessState.readers.completion != buffer->accessState.writer.completion)
  {
    dependencies.push_back(buffer->accessState.readers.completion->dependencyEvent());
  }
  destroyBufferEvents(buffer);
  DEBUG_GPU_FREE(*this, buffer->getId(), deviceId, nullptr);
  deviceContexts[deviceId]->enqueueAsyncFree(buffer->getPtr(), std::move(dependencies));
  buffer->ptr = nullptr;
}

void Swapper::destroyBufferEvents(std::shared_ptr<GpuBuffer> const& buffer)
{
  if (buffer->deviceContext == nullptr)
  {
    return;
  }
  buffer->accessState = {};
}

void Swapper::freeAllEvents(int deviceId) { (void)deviceId; }

void Swapper::syncMemStream(int deviceId, const char* reason)
{
  deviceContexts[deviceId]->syncMemoryStream(reason);
  deviceContexts[deviceId]->syncWorkStreams(reason);
}

void Swapper::initMemPools()
{
  // Use CUDA's default pool unless the legacy custom-pool path is explicitly
  // requested. The custom path preallocates most free GPU memory, which is too
  // expensive for repeated small DMRG local solves.
  const char* useDefaultPoolEnv = std::getenv("USE_DEFAULT_POOL");
  bool useDefaultPool = (useDefaultPoolEnv == nullptr || std::string(useDefaultPoolEnv) != "OFF");
  const bool memoryLog = std::getenv("TENSORCONTRACTION_MEMORY_LOG") != nullptr;

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if (memoryLog)
  {
    fprintf(stderr, "[MEMORY][POOL_MODE] Node=%d Mode=%s\n", mpi_rank, useDefaultPool ? "DEFAULT" : "CUSTOM");
  }

  if (useDefaultPool)
  {
    // Use default memory pool for each device
    for (int i = 0; i < deviceCount; i++)
    {
      CUDA_CALL(cudaSetDevice(i));
      cudaMemPool_t pool;
      CUDA_CALL(cudaDeviceGetDefaultMemPool(&pool, i));
      memPools.push_back(pool);
      ownsMemPool.push_back(false);
    }
    // Note: When using default pool, we don't set release threshold or maxSize
    // NCCL_HEADROOM is ignored in this mode
  }
  else
  {
    // Use custom memory pools with NCCL_HEADROOM
    // Parse NCCL_HEADROOM (format: "0" or "[0-9]+G", default: 3GB)
    size_t ncclHeadroom = 3ULL * 1024 * 1024 * 1024; // Default 3GB
    const char* ncclHeadroomEnv = std::getenv("NCCL_HEADROOM");
    if (ncclHeadroomEnv != nullptr)
    {
      std::string headroomStr(ncclHeadroomEnv);
      if (headroomStr == "0")
      {
        ncclHeadroom = 0;
      }
      else if (headroomStr.back() == 'G')
      {
        // Parse number before 'G'
        size_t gbValue = std::stoull(headroomStr.substr(0, headroomStr.size() - 1));
        ncclHeadroom = gbValue * 1024ULL * 1024 * 1024;
      }
    }

    if (memoryLog)
    {
      fprintf(stderr, "[MEMORY][NCCL_HEADROOM] Node=%d Size=%.2fGB\n", mpi_rank,
              ncclHeadroom / (1024.0 * 1024.0 * 1024.0));
    }

    // Step 1: Create memory pool for each device with maxSize limit
    for (int i = 0; i < deviceCount; i++)
    {
      CUDA_CALL(cudaSetDevice(i));

      size_t freeMem, totalMem;
      CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));
      size_t maxSize = freeMem > ncclHeadroom ? freeMem - ncclHeadroom : 0;

      cudaMemPoolProps poolProps = {};
      poolProps.allocType = cudaMemAllocationTypePinned;
      poolProps.handleTypes = cudaMemHandleTypeNone;
      poolProps.location.type = cudaMemLocationTypeDevice;
      poolProps.location.id = i;
      poolProps.maxSize = maxSize;

      cudaMemPool_t pool;
      CUDA_CALL(cudaMemPoolCreate(&pool, &poolProps));
      uint64_t threshold = UINT64_MAX;
      cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
      memPools.push_back(pool);
      ownsMemPool.push_back(true);

      // pre-allocate maxSize memory and free it to prevent memory frag.
      // if failed then let it fail.
      deviceContexts[i]->preallocatePool(pool, maxSize);
    }
  }

  for (int i = 0; i < deviceCount; i++)
  {
    for (int j = 0; j < deviceCount; j++)
    {
      if (i == j) continue;

      int canAccess;
      CUDA_CALL(cudaDeviceCanAccessPeer(&canAccess, i, j));

      if (canAccess)
      {
        cudaMemAccessDesc accessDesc = {};
        accessDesc.location.type = cudaMemLocationTypeDevice;
        accessDesc.location.id = j;
        accessDesc.flags = cudaMemAccessFlagsProtReadWrite;
        cudaMemPoolSetAccess(memPools[i], &accessDesc, 1);
      }
    }
  }

  CUDA_CALL(cudaSetDevice(0));
}

void Swapper::dumpMemPoolStatus(int deviceId)
{
  CUDA_CALL(cudaSetDevice(deviceId));

  uint64_t highestReserved = 0;
  uint64_t currentReserved = 0;
  uint64_t currentUsed = 0;

  CUDA_CALL(cudaMemPoolGetAttribute(memPools[deviceId], cudaMemPoolAttrReservedMemHigh, &highestReserved));
  CUDA_CALL(cudaMemPoolGetAttribute(memPools[deviceId], cudaMemPoolAttrReservedMemCurrent, &currentReserved));
  CUDA_CALL(cudaMemPoolGetAttribute(memPools[deviceId], cudaMemPoolAttrUsedMemCurrent, &currentUsed));

  size_t freeMem, totalMem;
  CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  printf("Node=%d Device=%d: total memory: %.3fGB\n", mpi_rank, deviceId, totalMem / (1024.0 * 1024.0 * 1024.0));
  printf("Node=%d Device=%d: highest reserved memory: %.3fGB\n", mpi_rank, deviceId,
         highestReserved / (1024.0 * 1024.0 * 1024.0));
  printf("Node=%d Device=%d: current reserved memory: %.3fGB\n", mpi_rank, deviceId,
         currentReserved / (1024.0 * 1024.0 * 1024.0));
  printf("Node=%d Device=%d: current used memory: %.3fGB\n", mpi_rank, deviceId,
         currentUsed / (1024.0 * 1024.0 * 1024.0));
  printf("Node=%d Device=%d: free memory: %.3fGB\n", mpi_rank, deviceId, freeMem / (1024.0 * 1024.0 * 1024.0));
}

void Swapper::freeBuffer(Matrix mat, int deviceId, cudaStream_t stream)
{
  if (isPinned(mat.getId(), deviceId))
  {
    return;
  }

  auto& hostToGpuMap = hostToGpuMaps[deviceId];
  auto it = hostToGpuMap.find(mat.getId());
  if (it == hostToGpuMap.end())
  {
    return;
  }

  auto buffer = it->second;
  hostToGpuMap.erase(it);
  freeBuffer(buffer, deviceId);
}

void Swapper::syncBuffer(Matrix mat, int deviceId, cudaStream_t stream)
{
  auto& hostToGpuMap = hostToGpuMaps[deviceId];
  auto it = hostToGpuMap.find(mat.getId());
  assert(it != hostToGpuMap.end());
  std::shared_ptr<GpuBuffer> buffer = it->second;

  buffer->waitBeforeRead(stream);

  if (const auto it = preStoreMap.find(mat.getId()); it != preStoreMap.end())
  {
    auto [dstDeviceId, dstBuffer] = it->second;
    assert(dstDeviceId != deviceId);
    dstBuffer->waitBeforeWrite(stream);
    DEBUG_GPU_COPY_D2D(*this, mat.getId(), deviceId, buffer->getPtr(), dstDeviceId, dstBuffer->getPtr(), stream,
                       mat.sizeInByte());
    CUDA_CALL(
        cudaMemcpyPeerAsync(dstBuffer->getPtr(), dstDeviceId, buffer->getPtr(), deviceId, mat.sizeInByte(), stream));
    auto completion = deviceContext(deviceId).recordCompletionEvent(stream);
    buffer->publishRead(completion);
    dstBuffer->publishWrite(std::move(completion));
    return;
  }
  else
  {
    DEBUG_GPU_COPY_D2H(*this, buffer->getId(), deviceId, stream, buffer->sizeInByte());
    CUDA_CALL(cudaMemcpyAsync(buffer->hostPtr, buffer->getPtr(), buffer->sizeInByte(), cudaMemcpyDeviceToHost, stream));
  }
  buffer->publishRead(stream);
}

void Swapper::clear()
{
  for (auto& hostToGpuMap : hostToGpuMaps)
  {
    for (auto& [id, buffer] : hostToGpuMap)
    {
      destroyBufferEvents(buffer);
      cudaFree(buffer->getPtr());
    }

    hostToGpuMap.clear();
  }

  for (auto [id, pair] : preStoreMap)
  {
    auto buffer = pair.second;
    destroyBufferEvents(buffer);
    CUDA_CALL(cudaFree(buffer->getPtr()));
  }

  preStoreMap.clear();
}

void Swapper::clear(Matrix mat)
{
  for (auto& hostToGpuMap : hostToGpuMaps)
  {
    auto it = hostToGpuMap.find(mat.getId());
    if (it == hostToGpuMap.end())
    {
      continue;
    }
    CUDA_CALL(cudaFree(it->second->getPtr()));
    hostToGpuMap.erase(it);
  }
}

} // namespace tensor
