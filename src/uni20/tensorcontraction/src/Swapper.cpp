#include <mpi.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "Calculator.hpp"
#include "Debug.hpp"
#include "Swapper.hpp"
#include "Utils.h"

namespace tensor
{
namespace
{

enum class CopyStatsSite : std::size_t
{
  CopyMatrixH2D = 0,
  PreStoreH2DIndividual,
  PreStoreH2DCoalesced,
  RefreshPreStoreH2DIndividual,
  RefreshPreStoreH2DCoalesced,
  PreStoreD2HIndividual,
  PreStoreD2HCoalesced,
  SyncBufferD2H,
  Count,
};

struct CopyStatsEntry
{
    const char* label;
    std::atomic<unsigned long long> calls = 0;
    std::atomic<unsigned long long> bytes = 0;
};

class CopyStats {
    bool enabled_ = std::getenv("UNI20_TENSORCONTRACTION_COPY_STATS") != nullptr;
    std::array<CopyStatsEntry, static_cast<std::size_t>(CopyStatsSite::Count)> entries_ = {
        CopyStatsEntry{"copyMatrix H2D"},
        CopyStatsEntry{"preStoreMatrix H2D individual"},
        CopyStatsEntry{"preStoreMatrix H2D coalesced"},
        CopyStatsEntry{"copyHostToPreStoreMatrix H2D individual"},
        CopyStatsEntry{"copyHostToPreStoreMatrix H2D coalesced"},
        CopyStatsEntry{"copyPreStoreMatrixToHost D2H individual"},
        CopyStatsEntry{"copyPreStoreMatrixToHost D2H coalesced"},
        CopyStatsEntry{"syncBuffer D2H"},
    };

  public:
    ~CopyStats()
    {
      if (!enabled_)
      {
        return;
      }

      std::fprintf(stderr, "[TENSORCONTRACTION][COPY_STATS] Swapper CUDA copy summary\n");
      for (auto const& entry : entries_)
      {
        auto const calls = entry.calls.load(std::memory_order_relaxed);
        auto const bytes = entry.bytes.load(std::memory_order_relaxed);
        if (calls == 0)
        {
          continue;
        }
        auto const gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        std::fprintf(stderr, "[TENSORCONTRACTION][COPY_STATS] %-32s calls=%llu bytes=%llu gib=%.6g\n", entry.label,
                     calls, bytes, gib);
      }
    }

    void record(CopyStatsSite site, std::size_t bytes)
    {
      if (!enabled_)
      {
        return;
      }

      auto& entry = entries_[static_cast<std::size_t>(site)];
      entry.calls.fetch_add(1, std::memory_order_relaxed);
      entry.bytes.fetch_add(static_cast<unsigned long long>(bytes), std::memory_order_relaxed);
    }
};

CopyStats& copyStats()
{
  static CopyStats stats;
  return stats;
}

void recordCopy(CopyStatsSite site, std::size_t bytes) { copyStats().record(site, bytes); }

std::size_t totalMatrixBytes(const std::vector<Matrix>& mats)
{
  std::size_t total = 0;
  for (auto const& mat : mats)
  {
    if (mat.sizeInByte() > std::numeric_limits<std::size_t>::max() - total)
    {
      throw std::length_error("TensorContraction matrix batch size overflows size_t");
    }
    total += mat.sizeInByte();
  }
  return total;
}

void validateCoalescedHostStorage(const std::vector<Matrix>& mats, std::span<double const> values)
{
  std::size_t offset = 0;
  for (auto const& mat : mats)
  {
    if (offset > values.size() || mat.size() > values.size() - offset)
    {
      throw std::invalid_argument("TensorContraction coalesced host span is too small for Matrix batch");
    }
    if (mat.size() != 0 && mat.hostView().data() != values.data() + offset)
    {
      throw std::invalid_argument("TensorContraction Matrix batch is not backed by the supplied coalesced host span");
    }
    offset += mat.size();
  }
  if (offset != values.size())
  {
    throw std::invalid_argument("TensorContraction coalesced host span has trailing values not covered by matrices");
  }
}

} // namespace

GpuBuffer::GpuBuffer(GpuBuffer&& other)
    : ptr(other.ptr), id(other.id), dim1(other.dim1), dim2(other.dim2), hostPtr(other.hostPtr),
      hasValidContent(other.hasValidContent), dependencyEventsEnabled(other.dependencyEventsEnabled),
      deviceContext(other.deviceContext), accessState(std::move(other.accessState)),
      allocationGroup(std::move(other.allocationGroup))
{
  other.ptr = nullptr;
  other.dim1 = 0;
  other.dim2 = 0;
  other.hostPtr = nullptr;
  other.hasValidContent = false;
  other.dependencyEventsEnabled = false;
  other.deviceContext = nullptr;
  other.accessState = {};
  other.allocationGroup = nullptr;
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
    allocationGroup = std::move(other.allocationGroup);

    // Reset the moved-from object to a safe state
    other.ptr = nullptr;
    other.dim1 = 0;
    other.dim2 = 0;
    other.hostPtr = nullptr;
    other.hasValidContent = false;
    other.dependencyEventsEnabled = false;
    other.deviceContext = nullptr;
    other.accessState = {};
    other.allocationGroup = nullptr;
  }
  return *this;
}

GpuBuffer::GpuBuffer(void* ptr, Matrix mat, bool dependencyEventsEnabled, CudaDeviceContext& deviceContext,
                     std::shared_ptr<AllocationGroup> allocationGroup)
    : ptr(ptr), id(mat.getId()), dim1(mat.getFirstDim()), dim2(mat.getSecondDim()), hostPtr(mat.getPtr()),
      dependencyEventsEnabled(dependencyEventsEnabled), deviceContext(&deviceContext),
      allocationGroup(std::move(allocationGroup))
{}

double* GpuBuffer::getPtr() { return static_cast<double*>(ptr); }
int GpuBuffer::getId() const { return id; }
DeviceMatrixView GpuBuffer::deviceView(int deviceId) const
{
  return DeviceMatrixView(MatrixHandle(id, dim1, dim2), deviceId, static_cast<double*>(ptr), *deviceContext,
                          hasValidContent);
}

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

bool Swapper::anyPreStoreBuffer(const std::vector<Matrix>& mats) const
{
  for (auto const& mat : mats)
  {
    if (preStoreMap.find(mat.getId()) != preStoreMap.end())
    {
      return true;
    }
  }
  return false;
}

std::optional<int> Swapper::commonPreStoreDevice(const std::vector<Matrix>& mats) const
{
  if (mats.empty())
  {
    return std::nullopt;
  }

  std::optional<int> deviceId;
  for (auto const& mat : mats)
  {
    auto const it = preStoreMap.find(mat.getId());
    if (it == preStoreMap.end())
    {
      return std::nullopt;
    }
    if (!deviceId.has_value())
    {
      deviceId = it->second.first;
      continue;
    }
    if (*deviceId != it->second.first)
    {
      return std::nullopt;
    }
  }
  return deviceId;
}

bool Swapper::preStoreBuffersAreCoalesced(const std::vector<Matrix>& mats, int deviceId) const
{
  if (mats.empty())
  {
    return true;
  }

  std::shared_ptr<GpuBuffer::AllocationGroup> group;
  std::size_t offsetBytes = 0;
  for (auto const& mat : mats)
  {
    auto const it = preStoreMap.find(mat.getId());
    if (it == preStoreMap.end() || it->second.first != deviceId)
    {
      return false;
    }
    auto const& buffer = it->second.second;
    if (buffer == nullptr || buffer->allocationGroup == nullptr || buffer->allocationGroup->basePtr == nullptr)
    {
      return false;
    }
    if (group == nullptr)
    {
      group = buffer->allocationGroup;
    }
    else if (group != buffer->allocationGroup)
    {
      return false;
    }
    auto* expected = static_cast<std::byte*>(group->basePtr) + offsetBytes;
    if (buffer->ptr != expected)
    {
      return false;
    }
    offsetBytes += mat.sizeInByte();
  }
  return group != nullptr && offsetBytes == group->bytes;
}

std::vector<std::shared_ptr<GpuBuffer>> Swapper::collectCoalescedPreStoreBuffers(const std::vector<Matrix>& mats,
                                                                                 int deviceId) const
{
  if (!this->preStoreBuffersAreCoalesced(mats, deviceId))
  {
    return {};
  }

  std::vector<std::shared_ptr<GpuBuffer>> buffers;
  buffers.reserve(mats.size());
  for (auto const& mat : mats)
  {
    auto const it = preStoreMap.find(mat.getId());
    if (it == preStoreMap.end() || it->second.first != deviceId || it->second.second == nullptr)
    {
      return {};
    }
    buffers.push_back(it->second.second);
  }
  return buffers;
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
      deviceContexts[deviceId]->flushPoolCache();
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
  if (deviceId < 0 || deviceId >= static_cast<int>(memPools.size()) ||
      deviceId >= static_cast<int>(deviceContexts.size()))
  {
    throw std::logic_error("TensorContraction attempted GPU allocation on an inactive device");
  }

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
  auto allocationGroup = std::make_shared<GpuBuffer::AllocationGroup>();
  allocationGroup->basePtr = allocation.ptr;
  allocationGroup->bytes = mat.sizeInByte();
  allocationGroup->liveBuffers = 1;
  auto buffer = std::make_shared<GpuBuffer>(allocation.ptr, mat, dependencyEventsEnabled, *deviceContexts[deviceId],
                                            std::move(allocationGroup));
  auto& hostToGpuMap = hostToGpuMaps[deviceId];
  hostToGpuMap[buffer->getId()] = buffer;

  buffer->publishAllocation(std::move(allocation.completion));
  return buffer;
}

std::shared_ptr<GpuBuffer> Swapper::ensureLocalCopy(Matrix mat, int deviceId)
{
  if (deviceId < 0 || deviceId >= deviceCount)
  {
    throw std::logic_error("TensorContraction requested a local matrix copy on an inactive device");
  }

  if (auto buffer = getGpuBufferOrNone(mat, deviceId); buffer != nullptr && buffer->contentValid())
  {
    return buffer;
  }

  auto destination = getGpuBufferOrNone(mat, deviceId);
  if (destination == nullptr)
  {
    destination = allocate(mat, deviceId);
  }
  if (destination == nullptr)
  {
    throw std::runtime_error("TensorContraction failed to allocate a local matrix staging buffer");
  }

  auto [sourceDeviceId, sourceBuffer] = findLocalSourceBuffer(mat, deviceId);
  if (sourceBuffer != nullptr)
  {
    if (sourceDeviceId == deviceId)
    {
      return sourceBuffer;
    }

    auto streamOwner = deviceContext(deviceId).leaseWorkStream();
    auto stream = streamOwner.stream();
    waitForAccessDependencies({sourceBuffer}, {destination}, stream);
    CUDA_CALL(cudaMemcpyPeerAsync(destination->getPtr(), deviceId, sourceBuffer->getPtr(), sourceDeviceId,
                                  mat.sizeInByte(), stream));
    auto completion = streamOwner.recordCompletion();
    publishAccessCompletion({sourceBuffer}, {destination}, stream, std::move(completion));
    DEBUG_SECOND_HIT(*this, mat.getId(), sourceDeviceId, deviceId, mat.sizeInByte());
    return destination;
  }

  if (mat.hasHostStorage())
  {
    auto streamOwner = deviceContext(deviceId).leaseWorkStream();
    auto stream = streamOwner.stream();
    waitForAccessDependencies({}, {destination}, stream);
    copyHostToDevice(mat.hostView(), destination, deviceId, stream);
    return destination;
  }

  if (isOnRemoteGpu(mat.getId()))
  {
    throw std::logic_error(
        "TensorContraction deterministic RABC executor does not yet stage matrices from remote MPI ranks");
  }
  throw std::logic_error("TensorContraction tried to stage a hostless matrix with no local GPU source");
}

void Swapper::copyMatrix(Matrix mat, std::shared_ptr<GpuBuffer> buffer, int deviceId, cudaStream_t stream)
{
  this->copyHostToDevice(mat.hostView(), std::move(buffer), deviceId, stream);
}

void Swapper::copyHostToDevice(HostMatrixView host, std::shared_ptr<GpuBuffer> buffer, int deviceId,
                               cudaStream_t stream)
{
  assert(host.valid());
  DEBUG_GPU_COPY_H2D(*this, host.handle().id(), deviceId, stream, host.sizeInByte());
  recordCopy(CopyStatsSite::CopyMatrixH2D, host.sizeInByte());
  CUDA_CALL(cudaMemcpyAsync(buffer->getPtr(), host.requireData(), host.sizeInByte(), cudaMemcpyHostToDevice, stream));
  buffer->publishWrite(stream);
}

void Swapper::preStoreMatrix(Matrix mat, int deviceId) { (void)this->uploadHostMatrix(mat.hostView(), deviceId); }

std::shared_ptr<GpuBuffer> Swapper::uploadHostMatrix(HostMatrixView host, int deviceId)
{
  assert(host.valid());
  DEBUG_PRESTORE(*this, host.handle().id(), 0, deviceId);
  auto allocation = deviceContexts[deviceId]->allocateFromPool(memPools[deviceId], host.sizeInByte());
  CUDA_CALL(allocation.status);
  Matrix mat(host.handle().toHeader());
  mat.setPtr(host.data());
  mat.setHostMemoryKind(host.memoryKind());
  auto allocationGroup = std::make_shared<GpuBuffer::AllocationGroup>();
  allocationGroup->basePtr = allocation.ptr;
  allocationGroup->bytes = host.sizeInByte();
  allocationGroup->liveBuffers = 1;
  auto buffer = std::make_shared<GpuBuffer>(allocation.ptr, mat, dependencyEventsEnabled, *deviceContexts[deviceId],
                                            std::move(allocationGroup));
  buffer->publishAllocation(std::move(allocation.completion));

  auto stream = deviceContexts[deviceId]->leaseWorkStream();
  buffer->waitBeforeWrite(stream.stream());
  DEBUG_GPU_COPY_H2D(*this, host.handle().id(), deviceId, stream.stream(), host.sizeInByte());
  recordCopy(CopyStatsSite::PreStoreH2DIndividual, host.sizeInByte());
  CUDA_CALL(cudaMemcpyAsync(buffer->getPtr(), host.requireData(), host.sizeInByte(), cudaMemcpyHostToDevice,
                            stream.stream()));
  buffer->publishWrite(stream.recordCompletion());
  assert(preStoreMap.find(host.handle().id()) == preStoreMap.end());
  preStoreMap[host.handle().id()] = std::make_pair(deviceId, buffer);
  return buffer;
}

void Swapper::registerGpuAllocationsCoalesced(const std::vector<Matrix>& mats, int deviceId)
{
  if (mats.empty())
  {
    return;
  }
  std::size_t const totalBytes = totalMatrixBytes(mats);
  if (totalBytes == 0)
  {
    throw std::invalid_argument("TensorContraction cannot coalesce an empty Matrix batch allocation");
  }
  std::unordered_set<int> ids;
  ids.reserve(mats.size());
  for (auto const& mat : mats)
  {
    if (!ids.insert(mat.getId()).second)
    {
      throw std::logic_error("TensorContraction cannot coalesce a Matrix batch with duplicate matrix ids");
    }
    if (preStoreMap.find(mat.getId()) != preStoreMap.end())
    {
      throw std::logic_error("TensorContraction cannot coalesce a Matrix batch with existing GPU buffers");
    }
  }

  CUDA_CALL(cudaSetDevice(deviceId));
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  auto allocation = deviceContexts[deviceId]->allocateFromPool(memPools[deviceId], totalBytes);
  CUDA_CALL(allocation.status);

  auto allocationGroup = std::make_shared<GpuBuffer::AllocationGroup>();
  allocationGroup->basePtr = allocation.ptr;
  allocationGroup->bytes = totalBytes;
  allocationGroup->liveBuffers = mats.size();

  auto* base = static_cast<std::byte*>(allocation.ptr);
  std::size_t offsetBytes = 0;
  for (auto const& mat : mats)
  {
    DEBUG_PRESTORE(*this, mat.getId(), mpi_rank, deviceId);
    auto* ptr = base + offsetBytes;
    auto buffer =
        std::make_shared<GpuBuffer>(ptr, mat, dependencyEventsEnabled, *deviceContexts[deviceId], allocationGroup);
    buffer->publishAllocation(allocation.completion);
    preStoreMap[mat.getId()] = std::make_pair(deviceId, std::move(buffer));
    offsetBytes += mat.sizeInByte();
  }
}

void Swapper::uploadHostMatricesCoalesced(const std::vector<Matrix>& mats, std::span<double const> values, int deviceId)
{
  if (mats.empty())
  {
    return;
  }
  validateCoalescedHostStorage(mats, values);
  std::size_t const totalBytes = values.size_bytes();
  if (totalBytes == 0)
  {
    throw std::invalid_argument("TensorContraction cannot coalesce an empty Matrix batch upload");
  }

  this->registerGpuAllocationsCoalesced(mats, deviceId);

  auto access = this->createSlabAccessPlan(mats, deviceId, SlabAccessKind::Write);
  DEBUG_GPU_COPY_H2D(*this, mats.front().getId(), deviceId, access.stream(), totalBytes);
  recordCopy(CopyStatsSite::PreStoreH2DCoalesced, totalBytes);
  CUDA_CALL(cudaMemcpyAsync(access.data(), values.data(), totalBytes, cudaMemcpyHostToDevice, access.stream()));
}

void Swapper::copyHostToPreStoreMatrix(Matrix mat) { this->refreshHostMatrixToDevice(mat.hostView()); }

void Swapper::refreshHostMatrixToDevice(HostMatrixView host)
{
  // Refresh an existing resident buffer from the host-side MatrixFamily block.
  // This is used only at explicit host/GPU authority boundaries.
  Matrix mat(host.handle().toHeader());
  auto [deviceId, buffer] = getPreStoreBufferOrNone(mat);
  assert(buffer != nullptr);
  assert(host.valid());

  CUDA_CALL(cudaSetDevice(deviceId));
  auto stream = deviceContexts[deviceId]->leaseWorkStream();
  buffer->waitBeforeWrite(stream.stream());
  DEBUG_GPU_COPY_H2D(*this, host.handle().id(), deviceId, stream.stream(), host.sizeInByte());
  recordCopy(CopyStatsSite::RefreshPreStoreH2DIndividual, host.sizeInByte());
  CUDA_CALL(cudaMemcpyAsync(buffer->getPtr(), host.requireData(), host.sizeInByte(), cudaMemcpyHostToDevice,
                            stream.stream()));
  buffer->publishWrite(stream.recordCompletion());
}

bool Swapper::refreshHostMatricesToDeviceCoalesced(const std::vector<Matrix>& mats, std::span<double const> values,
                                                   int deviceId)
{
  if (mats.empty())
  {
    return true;
  }
  validateCoalescedHostStorage(mats, values);
  if (!this->preStoreBuffersAreCoalesced(mats, deviceId))
  {
    return false;
  }

  auto access = this->createSlabAccessPlan(mats, deviceId, SlabAccessKind::Write);
  DEBUG_GPU_COPY_H2D(*this, mats.front().getId(), deviceId, access.stream(), values.size_bytes());
  recordCopy(CopyStatsSite::RefreshPreStoreH2DCoalesced, values.size_bytes());
  CUDA_CALL(
      cudaMemcpyAsync(access.data(), values.data(), values.size_bytes(), cudaMemcpyHostToDevice, access.stream()));
  return true;
}

void Swapper::copyPreStoreMatrixToHost(Matrix mat) { this->downloadDeviceToHost(mat.hostView()); }

void Swapper::downloadDeviceToHost(HostMatrixView host)
{
  // Materialize resident GPU data back into the MatrixFamily host block without
  // going through a worklist SyncWork for every vector operation.
  Matrix mat(host.handle().toHeader());
  auto [deviceId, buffer] = getPreStoreBufferOrNone(mat);
  assert(buffer != nullptr);
  assert(host.valid());

  CUDA_CALL(cudaSetDevice(deviceId));
  auto stream = deviceContexts[deviceId]->leaseWorkStream();
  buffer->waitBeforeRead(stream.stream());
  DEBUG_GPU_COPY_D2H(*this, host.handle().id(), deviceId, stream.stream(), host.sizeInByte());
  recordCopy(CopyStatsSite::PreStoreD2HIndividual, host.sizeInByte());
  CUDA_CALL(cudaMemcpyAsync(host.requireData(), buffer->getPtr(), host.sizeInByte(), cudaMemcpyDeviceToHost,
                            stream.stream()));
  buffer->publishRead(stream.recordCompletion());
}

bool Swapper::downloadDeviceMatricesToHostCoalesced(const std::vector<Matrix>& mats, std::span<double> values)
{
  if (mats.empty())
  {
    return true;
  }
  validateCoalescedHostStorage(mats, std::span<double const>{values.data(), values.size()});
  auto const deviceId = this->commonPreStoreDevice(mats);
  if (!deviceId.has_value() || !this->preStoreBuffersAreCoalesced(mats, *deviceId))
  {
    return false;
  }

  auto access = this->createSlabAccessPlan(mats, *deviceId, SlabAccessKind::Read);
  DEBUG_GPU_COPY_D2H(*this, mats.front().getId(), *deviceId, access.stream(), values.size_bytes());
  recordCopy(CopyStatsSite::PreStoreD2HCoalesced, values.size_bytes());
  CUDA_CALL(
      cudaMemcpyAsync(values.data(), access.data(), values.size_bytes(), cudaMemcpyDeviceToHost, access.stream()));
  return true;
}

void Swapper::registerGpuAllocation(Matrix mat, int deviceId)
{
  CUDA_CALL(cudaSetDevice(deviceId));
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  DEBUG_PRESTORE(*this, mat.getId(), mpi_rank, deviceId);
  auto allocation = deviceContexts[deviceId]->allocateFromPool(memPools[deviceId], mat.sizeInByte());
  CUDA_CALL(allocation.status);
  auto allocationGroup = std::make_shared<GpuBuffer::AllocationGroup>();
  allocationGroup->basePtr = allocation.ptr;
  allocationGroup->bytes = mat.sizeInByte();
  allocationGroup->liveBuffers = 1;
  auto buffer = std::make_shared<GpuBuffer>(allocation.ptr, mat, dependencyEventsEnabled, *deviceContexts[deviceId],
                                            std::move(allocationGroup));
  buffer->publishAllocation(std::move(allocation.completion));
  assert(preStoreMap.find(mat.getId()) == preStoreMap.end());
  preStoreMap[mat.getId()] = std::make_pair(deviceId, buffer);
}

void Swapper::ensurePreStoreOnDevice(Matrix mat, int deviceId, bool preserveExistingContent)
{
  auto [existingDeviceId, existingBuffer] = this->getPreStoreBufferOrNone(mat);
  if (existingBuffer == nullptr)
  {
    if (preserveExistingContent)
    {
      throw std::logic_error("TensorContraction cannot relocate a pre-store matrix with no resident source buffer");
    }
    this->registerGpuAllocation(mat, deviceId);
    return;
  }
  if (preserveExistingContent && !existingBuffer->contentValid())
  {
    throw std::logic_error("TensorContraction cannot relocate a pre-store matrix with invalid resident contents");
  }
  if (existingDeviceId == deviceId)
  {
    return;
  }

  if (!preserveExistingContent)
  {
    this->clear(mat);
    this->registerGpuAllocation(mat, deviceId);
    return;
  }

  CUDA_CALL(cudaSetDevice(deviceId));
  auto allocation = deviceContexts[deviceId]->allocateFromPool(memPools[deviceId], mat.sizeInByte());
  CUDA_CALL(allocation.status);

  auto allocationGroup = std::make_shared<GpuBuffer::AllocationGroup>();
  allocationGroup->basePtr = allocation.ptr;
  allocationGroup->bytes = mat.sizeInByte();
  allocationGroup->liveBuffers = 1;
  auto replacement = std::make_shared<GpuBuffer>(allocation.ptr, mat, dependencyEventsEnabled,
                                                 *deviceContexts[deviceId], std::move(allocationGroup));
  replacement->publishAllocation(std::move(allocation.completion));

  {
    auto access = this->createAccessPlan({existingBuffer}, {replacement}, deviceId);
    DEBUG_GPU_COPY_D2D(*this, mat.getId(), existingDeviceId, existingBuffer->getPtr(), deviceId, replacement->getPtr(),
                       access.stream(), mat.sizeInByte());
    if (existingDeviceId == deviceId)
    {
      CUDA_CALL(cudaMemcpyAsync(replacement->getPtr(), existingBuffer->getPtr(), mat.sizeInByte(),
                                cudaMemcpyDeviceToDevice, access.stream()));
    }
    else
    {
      CUDA_CALL(cudaMemcpyPeerAsync(replacement->getPtr(), deviceId, existingBuffer->getPtr(), existingDeviceId,
                                    mat.sizeInByte(), access.stream()));
    }
  }

  preStoreMap[mat.getId()] = std::make_pair(deviceId, replacement);
  this->freeBuffer(std::move(existingBuffer), existingDeviceId);
}

void Swapper::ensurePreStoreCoalescedOnDevice(std::vector<Matrix> const& mats, int deviceId,
                                              bool preserveExistingContent)
{
  if (mats.empty())
  {
    return;
  }
  if (this->preStoreBuffersAreCoalesced(mats, deviceId))
  {
    return;
  }
  if (!this->anyPreStoreBuffer(mats))
  {
    if (preserveExistingContent)
    {
      throw std::logic_error("TensorContraction cannot relocate a coalesced pre-store slab with no resident sources");
    }
    this->registerGpuAllocationsCoalesced(mats, deviceId);
    return;
  }
  if (!preserveExistingContent)
  {
    for (auto const& mat : mats)
    {
      this->clear(mat);
    }
    this->registerGpuAllocationsCoalesced(mats, deviceId);
    return;
  }

  std::size_t const totalBytes = totalMatrixBytes(mats);
  if (totalBytes == 0)
  {
    throw std::invalid_argument("TensorContraction cannot coalesce an empty Matrix batch allocation");
  }

  CUDA_CALL(cudaSetDevice(deviceId));
  auto allocation = deviceContexts[deviceId]->allocateFromPool(memPools[deviceId], totalBytes);
  CUDA_CALL(allocation.status);

  auto allocationGroup = std::make_shared<GpuBuffer::AllocationGroup>();
  allocationGroup->basePtr = allocation.ptr;
  allocationGroup->bytes = totalBytes;
  allocationGroup->liveBuffers = mats.size();

  std::vector<std::pair<int, std::shared_ptr<GpuBuffer>>> oldBuffers;
  std::vector<std::shared_ptr<GpuBuffer>> readBuffers;
  std::vector<std::shared_ptr<GpuBuffer>> writeBuffers;
  std::vector<std::shared_ptr<GpuBuffer>> replacements;
  oldBuffers.reserve(mats.size());
  readBuffers.reserve(mats.size());
  writeBuffers.reserve(mats.size());
  replacements.reserve(mats.size());

  auto* base = static_cast<std::byte*>(allocation.ptr);
  std::size_t offsetBytes = 0;
  for (auto const& mat : mats)
  {
    auto* ptr = base + offsetBytes;
    auto replacement =
        std::make_shared<GpuBuffer>(ptr, mat, dependencyEventsEnabled, *deviceContexts[deviceId], allocationGroup);
    replacement->publishAllocation(allocation.completion);
    auto [oldDeviceId, oldBuffer] = this->getPreStoreBufferOrNone(mat);
    if (oldBuffer != nullptr)
    {
      if (!oldBuffer->contentValid())
      {
        throw std::logic_error(
            "TensorContraction cannot relocate a coalesced pre-store slab with invalid resident contents");
      }
      oldBuffers.emplace_back(oldDeviceId, oldBuffer);
      readBuffers.push_back(oldBuffer);
      writeBuffers.push_back(replacement);
    }
    else
    {
      throw std::logic_error(
          "TensorContraction cannot relocate a coalesced pre-store slab with a missing resident source");
    }
    replacements.push_back(std::move(replacement));
    offsetBytes += mat.sizeInByte();
  }

  if (!readBuffers.empty())
  {
    auto access = this->createAccessPlan(readBuffers, writeBuffers, deviceId);
    for (std::size_t copy = 0; copy < readBuffers.size(); ++copy)
    {
      auto const oldDeviceId = oldBuffers[copy].first;
      auto const& oldBuffer = readBuffers[copy];
      auto const& replacement = writeBuffers[copy];
      DEBUG_GPU_COPY_D2D(*this, oldBuffer->getId(), oldDeviceId, oldBuffer->getPtr(), deviceId, replacement->getPtr(),
                         access.stream(), oldBuffer->sizeInByte());
      if (oldDeviceId == deviceId)
      {
        CUDA_CALL(cudaMemcpyAsync(replacement->getPtr(), oldBuffer->getPtr(), oldBuffer->sizeInByte(),
                                  cudaMemcpyDeviceToDevice, access.stream()));
      }
      else
      {
        CUDA_CALL(cudaMemcpyPeerAsync(replacement->getPtr(), deviceId, oldBuffer->getPtr(), oldDeviceId,
                                      oldBuffer->sizeInByte(), access.stream()));
      }
    }
  }

  for (std::size_t index = 0; index < mats.size(); ++index)
  {
    preStoreMap[mats[index].getId()] = std::make_pair(deviceId, replacements[index]);
  }
  for (auto& [oldDeviceId, oldBuffer] : oldBuffers)
  {
    this->freeBuffer(std::move(oldBuffer), oldDeviceId);
  }
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

cuda::CompletionRef Swapper::GpuAccessPlan::recordCompletion() const { return streamOwner.recordCompletion(); }

void Swapper::GpuAccessPlan::publishCompletion(cuda::CompletionRef completion) const
{
  swapper.publishAccessCompletion(readBuffers, writeBuffers, this->stream(), std::move(completion));
}

Swapper::BlasAccessPlan::BlasAccessPlan(Swapper& swapper, int deviceId,
                                        std::vector<std::shared_ptr<GpuBuffer>> readBuffers,
                                        std::vector<std::shared_ptr<GpuBuffer>> writeBuffers)
    : swapper(swapper), deviceId(deviceId), streamOwner(swapper.deviceContext(deviceId).leaseBlasStream()),
      readBuffers(std::move(readBuffers)), writeBuffers(std::move(writeBuffers))
{
  swapper.waitForAccessDependencies(this->readBuffers, this->writeBuffers, this->stream());
}

Swapper::BlasAccessPlan::~BlasAccessPlan()
{
  if (swapper.dependencyEventsActive())
  {
    this->publishCompletion(this->recordCompletion());
  }
}

cublasHandle_t Swapper::BlasAccessPlan::handle() const { return streamOwner.prepare_handle(); }

cuda::CompletionRef Swapper::BlasAccessPlan::recordCompletion() const { return streamOwner.recordCompletion(); }

void Swapper::BlasAccessPlan::publishCompletion(cuda::CompletionRef completion) const
{
  swapper.publishAccessCompletion(readBuffers, writeBuffers, this->stream(), std::move(completion));
}

Swapper::SlabAccessPlan::SlabAccessPlan(Swapper& swapper, int deviceId, SlabAccessKind accessKind,
                                        std::vector<std::shared_ptr<GpuBuffer>> buffers)
    : swapper(swapper), deviceId(deviceId), accessKind(accessKind),
      streamOwner(swapper.deviceContext(deviceId).leaseWorkStream()), buffers(std::move(buffers))
{
  if (this->buffers.empty())
  {
    throw std::logic_error("TensorContraction slab access requires at least one GPU buffer");
  }

  auto const group = this->buffers.front()->allocationGroup;
  if (group == nullptr || group->basePtr == nullptr || group->bytes == 0)
  {
    throw std::logic_error("TensorContraction slab access requires a live coalesced allocation group");
  }

  basePtr = group->basePtr;
  bytes = group->bytes;
  std::size_t offsetBytes = 0;
  for (auto const& buffer : this->buffers)
  {
    if (buffer == nullptr || buffer->allocationGroup != group)
    {
      throw std::logic_error("TensorContraction slab access received buffers from different allocation groups");
    }
    auto* expected = static_cast<std::byte*>(basePtr) + offsetBytes;
    if (buffer->ptr != expected)
    {
      throw std::logic_error("TensorContraction slab access received non-contiguous sub-block buffers");
    }
    offsetBytes += buffer->sizeInByte();
  }
  if (offsetBytes != bytes)
  {
    throw std::logic_error("TensorContraction slab access received an incomplete allocation group");
  }

  if (accessKind == SlabAccessKind::Read)
  {
    swapper.waitForAccessDependencies(this->buffers, {}, this->stream());
  }
  else
  {
    swapper.waitForAccessDependencies({}, this->buffers, this->stream());
  }
}

Swapper::SlabAccessPlan::~SlabAccessPlan()
{
  if (swapper.dependencyEventsActive() && !published)
  {
    this->publishCompletion(this->recordCompletion());
  }
}

cuda::CompletionRef Swapper::SlabAccessPlan::recordCompletion() const { return streamOwner.recordCompletion(); }

void Swapper::SlabAccessPlan::publishCompletion(cuda::CompletionRef completion)
{
  if (published)
  {
    return;
  }
  if (accessKind == SlabAccessKind::Read)
  {
    swapper.publishAccessCompletion(buffers, {}, this->stream(), std::move(completion));
  }
  else
  {
    swapper.publishAccessCompletion({}, buffers, this->stream(), std::move(completion));
  }
  published = true;
}

auto Swapper::createAccessPlan(std::vector<std::shared_ptr<GpuBuffer>> readBuffers,
                               std::vector<std::shared_ptr<GpuBuffer>> writeBuffers, int deviceId) -> GpuAccessPlan
{
  return GpuAccessPlan(*this, deviceId, std::move(readBuffers), std::move(writeBuffers));
}

auto Swapper::createBlasAccessPlan(std::vector<std::shared_ptr<GpuBuffer>> readBuffers,
                                   std::vector<std::shared_ptr<GpuBuffer>> writeBuffers, int deviceId) -> BlasAccessPlan
{
  return BlasAccessPlan(*this, deviceId, std::move(readBuffers), std::move(writeBuffers));
}

auto Swapper::createSlabAccessPlan(const std::vector<Matrix>& mats, int deviceId,
                                   SlabAccessKind accessKind) -> SlabAccessPlan
{
  auto buffers = this->collectCoalescedPreStoreBuffers(mats, deviceId);
  if (buffers.empty())
  {
    throw std::logic_error("TensorContraction slab access requires a complete coalesced Matrix batch");
  }
  return SlabAccessPlan(*this, deviceId, accessKind, std::move(buffers));
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

std::vector<CudaDeviceContext::EventDependencyRef>
Swapper::collectBufferDependencies(std::shared_ptr<GpuBuffer> const& buffer) const
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
  return dependencies;
}

void Swapper::freeBuffer(std::shared_ptr<GpuBuffer> buffer, int deviceId)
{
  if (buffer == nullptr || buffer->ptr == nullptr)
  {
    return;
  }

  auto dependencies = this->collectBufferDependencies(buffer);
  destroyBufferEvents(buffer);

  auto group = buffer->allocationGroup;
  if (group != nullptr)
  {
    group->dependencies.insert(group->dependencies.end(), dependencies.begin(), dependencies.end());
    if (group->liveBuffers == 0)
    {
      throw std::logic_error("TensorContraction GPU allocation group was released too many times");
    }
    --group->liveBuffers;
    if (group->liveBuffers == 0 && !group->freeScheduled)
    {
      DEBUG_GPU_FREE(*this, buffer->getId(), deviceId, nullptr);
      deviceContexts[deviceId]->enqueueAsyncFree(group->basePtr, memPools[deviceId], group->bytes,
                                                 std::move(group->dependencies));
      group->freeScheduled = true;
      group->basePtr = nullptr;
      group->bytes = 0;
    }
  }
  else
  {
    DEBUG_GPU_FREE(*this, buffer->getId(), deviceId, nullptr);
    deviceContexts[deviceId]->enqueueAsyncFree(buffer->getPtr(), memPools[deviceId], buffer->sizeInByte(),
                                               std::move(dependencies));
  }

  buffer->ptr = nullptr;
  buffer->allocationGroup.reset();
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
    auto host = mat.hostView();
    DEBUG_GPU_COPY_D2H(*this, host.handle().id(), deviceId, stream, host.sizeInByte());
    recordCopy(CopyStatsSite::SyncBufferD2H, host.sizeInByte());
    CUDA_CALL(cudaMemcpyAsync(host.requireData(), buffer->getPtr(), host.sizeInByte(), cudaMemcpyDeviceToHost, stream));
  }
  buffer->publishRead(stream);
}

void Swapper::clear()
{
  for (int deviceId = 0; deviceId < static_cast<int>(hostToGpuMaps.size()); ++deviceId)
  {
    auto& hostToGpuMap = hostToGpuMaps[deviceId];
    for (auto& [id, buffer] : hostToGpuMap)
    {
      freeBuffer(buffer, deviceId);
    }

    hostToGpuMap.clear();
  }

  for (auto [id, pair] : preStoreMap)
  {
    auto deviceId = pair.first;
    auto buffer = pair.second;
    freeBuffer(buffer, deviceId);
  }

  preStoreMap.clear();
}

void Swapper::clear(Matrix mat)
{
  for (int deviceId = 0; deviceId < static_cast<int>(hostToGpuMaps.size()); ++deviceId)
  {
    auto& hostToGpuMap = hostToGpuMaps[deviceId];
    auto it = hostToGpuMap.find(mat.getId());
    if (it == hostToGpuMap.end())
    {
      continue;
    }
    freeBuffer(it->second, deviceId);
    hostToGpuMap.erase(it);
  }

  auto preStoreIt = preStoreMap.find(mat.getId());
  if (preStoreIt != preStoreMap.end())
  {
    auto [deviceId, buffer] = preStoreIt->second;
    preStoreMap.erase(preStoreIt);
    freeBuffer(std::move(buffer), deviceId);
  }
}

} // namespace tensor
