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

thread_local bool accessDependencyWaitsSuppressed = false;

Swapper::ScopedAccessDependencyWaitSuppression::ScopedAccessDependencyWaitSuppression()
    : previous(accessDependencyWaitsSuppressed)
{
  accessDependencyWaitsSuppressed = true;
}

Swapper::ScopedAccessDependencyWaitSuppression::~ScopedAccessDependencyWaitSuppression()
{
  accessDependencyWaitsSuppressed = previous;
}

GpuBuffer::GpuBuffer(GpuBuffer&& other)
    : ptr(other.ptr), id(other.id), dim1(other.dim1), dim2(other.dim2), hostPtr(other.hostPtr),
      dependencyEventsEnabled(other.dependencyEventsEnabled), deviceContext(other.deviceContext),
      useFinishEvent(std::move(other.useFinishEvent)), useFinishStream(other.useFinishStream),
      writeFinishEvent(std::move(other.writeFinishEvent)), writeFinishStream(other.writeFinishStream)
{
  other.ptr = nullptr;
  other.dim1 = 0;
  other.dim2 = 0;
  other.hostPtr = nullptr;
  other.dependencyEventsEnabled = false;
  other.deviceContext = nullptr;
  other.useFinishStream = nullptr;
  other.writeFinishStream = nullptr;
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
    dependencyEventsEnabled = other.dependencyEventsEnabled;
    deviceContext = other.deviceContext;
    useFinishEvent = std::move(other.useFinishEvent);
    useFinishStream = other.useFinishStream;
    writeFinishEvent = std::move(other.writeFinishEvent);
    writeFinishStream = other.writeFinishStream;

    // Reset the moved-from object to a safe state
    other.ptr = nullptr;
    other.dim1 = 0;
    other.dim2 = 0;
    other.hostPtr = nullptr;
    other.dependencyEventsEnabled = false;
    other.deviceContext = nullptr;
    other.useFinishStream = nullptr;
    other.writeFinishStream = nullptr;
  }
  return *this;
}

GpuBuffer::GpuBuffer(void* ptr, Matrix mat, bool dependencyEventsEnabled, CudaDeviceContext& deviceContext)
    : ptr(ptr), id(mat.getId()), dim1(mat.getFirstDim()), dim2(mat.getSecondDim()), hostPtr(mat.getPtr()),
      dependencyEventsEnabled(dependencyEventsEnabled), deviceContext(&deviceContext)
{}

double* GpuBuffer::getPtr() { return static_cast<double*>(ptr); }
int GpuBuffer::getId() const { return id; }

void GpuBuffer::waitForReadFinish(cudaStream_t stream)
{
  if (!dependencyEventsEnabled || accessDependencyWaitsSuppressed)
  {
    return;
  }
  if (useFinishEvent != nullptr && useFinishStream != stream)
  {
    deviceContext->waitEvent(stream, useFinishEvent->event);
  }
}

void GpuBuffer::notifyReadFinish(cudaStream_t stream, CudaDeviceContext::EventDependencyRef event)
{
  if (!dependencyEventsEnabled)
  {
    return;
  }
  useFinishEvent = event;
  useFinishStream = stream;
}

void GpuBuffer::notifyReadFinish(cudaStream_t stream)
{
  notifyReadFinish(stream, deviceContext->recordDependencyEvent(stream));
}

void GpuBuffer::waitForWriteFinish(cudaStream_t stream)
{
  if (!dependencyEventsEnabled || accessDependencyWaitsSuppressed)
  {
    return;
  }
  if (writeFinishEvent != nullptr && writeFinishStream != stream)
  {
    deviceContext->waitEvent(stream, writeFinishEvent->event);
  }
}

void GpuBuffer::notifyWriteFinish(cudaStream_t stream, CudaDeviceContext::EventDependencyRef event)
{
  if (!dependencyEventsEnabled)
  {
    return;
  }
  writeFinishEvent = event;
  writeFinishStream = stream;
  useFinishEvent = event;
  useFinishStream = stream;
}

void GpuBuffer::notifyWriteFinish(cudaStream_t stream)
{
  notifyWriteFinish(stream, deviceContext->recordDependencyEvent(stream));
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

void Swapper::notifyMatrixRead(Matrix mat, int deviceId, cudaStream_t stream,
                               CudaDeviceContext::EventDependencyRef event)
{
  auto buffer = getGpuBufferOrNone(mat, deviceId);
  if (buffer != nullptr)
  {
    buffer->notifyReadFinish(stream, event);
  }
}

void Swapper::notifyMatrixWrite(Matrix mat, int deviceId, cudaStream_t stream,
                                CudaDeviceContext::EventDependencyRef event)
{
  auto buffer = getGpuBufferOrNone(mat, deviceId);
  if (buffer != nullptr)
  {
    buffer->notifyWriteFinish(stream, event);
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
  void* ptr;
  auto memStream = deviceContexts[deviceId]->memoryStream();
  auto memPool = memPools[deviceId];

  cudaError_t errCode = cudaMallocFromPoolAsync(&ptr, mat.sizeInByte(), memPool, memStream);

  if (errCode != cudaSuccess)
  {
    if (errCode != cudaErrorMemoryAllocation)
    {
      fprintf(stderr, "Unexpected CUDA Error at %s:%d\n", __FILE__, __LINE__);
      fprintf(stderr, "  Error code: %d\n", errCode);
      fprintf(stderr, "  Error text: %s\n", cudaGetErrorString(errCode));
      assert(false);
    }

    return nullptr;
  }

  DEBUG_GPU_ALLOC(*this, mat.getId(), deviceId, memStream, mat.sizeInByte());
  auto buffer = std::make_shared<GpuBuffer>(ptr, mat, dependencyEventsEnabled, *deviceContexts[deviceId]);
  auto& hostToGpuMap = hostToGpuMaps[deviceId];
  hostToGpuMap[buffer->getId()] = buffer;

  buffer->notifyWriteFinish(memStream);
  return buffer;
}

void Swapper::copyMatrix(Matrix mat, std::shared_ptr<GpuBuffer> buffer, int deviceId, cudaStream_t stream,
                         StreamManager& streamManager)
{
  assert(mat.getPtr() != nullptr);
  DEBUG_GPU_COPY_H2D(*this, mat.getId(), deviceId, stream, mat.sizeInByte());
  CUDA_CALL(cudaMemcpyAsync(buffer->getPtr(), mat.getPtr(), mat.sizeInByte(), cudaMemcpyHostToDevice, stream));
  buffer->notifyWriteFinish(stream);
}

void Swapper::preStoreMatrix(Matrix mat, int deviceId)
{
  DEBUG_PRESTORE(*this, mat.getId(), 0, deviceId);
  auto memStream = deviceContexts[deviceId]->memoryStream();
  double* gpuPtr;
  CUDA_CALL(cudaMallocFromPoolAsync(&gpuPtr, mat.sizeInByte(), memPools[deviceId], memStream));
  CUDA_CALL(cudaMemcpyAsync(gpuPtr, mat.getPtr(), mat.sizeInByte(), cudaMemcpyHostToDevice, memStream));
  auto buffer = std::make_shared<GpuBuffer>(gpuPtr, mat, dependencyEventsEnabled, *deviceContexts[deviceId]);
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
  auto memStream = deviceContexts[deviceId]->memoryStream();
  buffer->waitForReadFinish(memStream);
  DEBUG_GPU_COPY_H2D(*this, mat.getId(), deviceId, memStream, mat.sizeInByte());
  CUDA_CALL(cudaMemcpyAsync(buffer->getPtr(), mat.getPtr(), mat.sizeInByte(), cudaMemcpyHostToDevice, memStream));
  buffer->notifyWriteFinish(memStream);
}

void Swapper::copyPreStoreMatrixToHost(Matrix mat)
{
  // Materialize resident GPU data back into the MatrixFamily host block without
  // going through a worklist SyncWork for every vector operation.
  auto [deviceId, buffer] = getPreStoreBufferOrNone(mat);
  assert(buffer != nullptr);
  assert(mat.getPtr() != nullptr);

  CUDA_CALL(cudaSetDevice(deviceId));
  auto memStream = deviceContexts[deviceId]->memoryStream();
  buffer->waitForWriteFinish(memStream);
  DEBUG_GPU_COPY_D2H(*this, mat.getId(), deviceId, memStream, mat.sizeInByte());
  CUDA_CALL(cudaMemcpyAsync(mat.getPtr(), buffer->getPtr(), mat.sizeInByte(), cudaMemcpyDeviceToHost, memStream));
  buffer->notifyReadFinish(memStream);
}

void Swapper::registerGpuAllocation(Matrix mat, int deviceId)
{
  CUDA_CALL(cudaSetDevice(deviceId));
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  DEBUG_PRESTORE(*this, mat.getId(), mpi_rank, deviceId);
  auto memStream = deviceContexts[deviceId]->memoryStream();
  double* gpuPtr;
  CUDA_CALL(cudaMallocFromPoolAsync(&gpuPtr, mat.sizeInByte(), memPools[deviceId], memStream));
  auto buffer = std::make_shared<GpuBuffer>(gpuPtr, mat, dependencyEventsEnabled, *deviceContexts[deviceId]);
  assert(preStoreMap.find(mat.getId()) == preStoreMap.end());
  preStoreMap[mat.getId()] = std::make_pair(deviceId, buffer);
}

std::shared_ptr<GpuBuffer> Swapper::getForRead(Matrix mat, int deviceId, cudaStream_t stream)
{
  auto bufferPtr = getForReadNoWait(mat, deviceId);
  if (bufferPtr != nullptr)
  {
    bufferPtr->waitForWriteFinish(stream);
    return bufferPtr;
  }

  return nullptr;
}

std::shared_ptr<GpuBuffer> Swapper::getForWrite(Matrix mat, int deviceId, cudaStream_t stream)
{
  auto bufferPtr = getForWriteNoWait(mat, deviceId);
  if (bufferPtr != nullptr)
  {
    bufferPtr->waitForWriteFinish(stream);
    bufferPtr->waitForReadFinish(stream);
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

cudaStream_t Swapper::preferredStreamForAccess(const std::vector<std::shared_ptr<GpuBuffer>>& readBuffers,
                                               const std::vector<std::shared_ptr<GpuBuffer>>& writeBuffers) const
{
  if (!dependencyEventsEnabled)
  {
    return nullptr;
  }

  std::unordered_map<cudaStream_t, int> scores;
  auto addDependencyStream = [&scores](CudaDeviceContext::EventDependencyRef const& event, cudaStream_t stream) {
    if (event != nullptr && stream != nullptr)
    {
      ++scores[stream];
    }
  };

  for (auto const& buffer : readBuffers)
  {
    if (buffer != nullptr)
    {
      addDependencyStream(buffer->writeFinishEvent, buffer->writeFinishStream);
    }
  }

  for (auto const& buffer : writeBuffers)
  {
    if (buffer != nullptr)
    {
      addDependencyStream(buffer->writeFinishEvent, buffer->writeFinishStream);
      addDependencyStream(buffer->useFinishEvent, buffer->useFinishStream);
    }
  }

  cudaStream_t bestStream = nullptr;
  int bestScore = 0;
  for (auto const& [stream, score] : scores)
  {
    if (score > bestScore)
    {
      bestStream = stream;
      bestScore = score;
    }
  }
  return bestStream;
}

void Swapper::waitForAccessDependencies(const std::vector<std::shared_ptr<GpuBuffer>>& readBuffers,
                                        const std::vector<std::shared_ptr<GpuBuffer>>& writeBuffers,
                                        cudaStream_t stream)
{
  if (!dependencyEventsEnabled || accessDependencyWaitsSuppressed)
  {
    return;
  }

  std::unordered_set<cudaEvent_t> waitedEvents;
  auto waitEvent = [&](CudaDeviceContext* context, cudaEvent_t event, cudaStream_t producerStream) {
    if (event == nullptr || context == nullptr || producerStream == stream || !waitedEvents.insert(event).second)
    {
      return;
    }
    context->waitEvent(stream, event);
  };

  for (auto const& buffer : readBuffers)
  {
    if (buffer == nullptr)
    {
      continue;
    }
    if (buffer->writeFinishEvent != nullptr)
    {
      waitEvent(buffer->deviceContext, buffer->writeFinishEvent->event, buffer->writeFinishStream);
    }
  }

  for (auto const& buffer : writeBuffers)
  {
    if (buffer == nullptr)
    {
      continue;
    }
    if (buffer->writeFinishEvent != nullptr)
    {
      waitEvent(buffer->deviceContext, buffer->writeFinishEvent->event, buffer->writeFinishStream);
    }
    if (buffer->useFinishEvent != nullptr)
    {
      waitEvent(buffer->deviceContext, buffer->useFinishEvent->event, buffer->useFinishStream);
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
  auto memStream = deviceContexts[deviceId]->memoryStream();
  buffer->waitForReadFinish(memStream);
  buffer->waitForWriteFinish(memStream);
  destroyBufferEvents(buffer);
  DEBUG_GPU_FREE(*this, buffer->getId(), deviceId, memStream);
  CUDA_CALL(cudaFreeAsync(buffer->getPtr(), memStream));
}

void Swapper::destroyBufferEvents(std::shared_ptr<GpuBuffer> const& buffer)
{
  if (buffer->deviceContext == nullptr)
  {
    return;
  }
  buffer->useFinishEvent.reset();
  buffer->useFinishStream = nullptr;
  buffer->writeFinishEvent.reset();
  buffer->writeFinishStream = nullptr;
}

void Swapper::freeAllEvents(int deviceId) { (void)deviceId; }

void Swapper::syncMemStream(int deviceId, const char* reason) { deviceContexts[deviceId]->syncMemoryStream(reason); }

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
      void* preAllocPtr;
      auto memStream = deviceContexts[i]->memoryStream();
      cudaError_t preAllocErr = cudaMallocFromPoolAsync(&preAllocPtr, maxSize, pool, memStream);
      if (preAllocErr == cudaSuccess)
      {
        cudaFreeAsync(preAllocPtr, memStream);
      }
      CUDA_CALL(cudaStreamSynchronize(memStream));
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

  buffer->waitForWriteFinish(stream);

  if (const auto it = preStoreMap.find(mat.getId()); it != preStoreMap.end())
  {
    auto [dstDeviceId, dstBuffer] = it->second;
    assert(dstDeviceId != deviceId);
    DEBUG_GPU_COPY_D2D(*this, mat.getId(), deviceId, buffer->getPtr(), dstDeviceId, dstBuffer->getPtr(), stream,
                       mat.sizeInByte());
    CUDA_CALL(
        cudaMemcpyPeerAsync(dstBuffer->getPtr(), dstDeviceId, buffer->getPtr(), deviceId, mat.sizeInByte(), stream));
  }
  else
  {
    DEBUG_GPU_COPY_D2H(*this, buffer->getId(), deviceId, stream, buffer->sizeInByte());
    CUDA_CALL(cudaMemcpyAsync(buffer->hostPtr, buffer->getPtr(), buffer->sizeInByte(), cudaMemcpyDeviceToHost, stream));
  }
  buffer->notifyReadFinish(stream);
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
