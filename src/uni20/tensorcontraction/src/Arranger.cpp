#include "Arranger.hpp"

#include <mpi.h>
#include <nccl.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "Calculator.hpp"
#include "Debug.hpp"
#include "MatrixAllocator.hpp"
#include "Swapper.hpp"

static int getLeastBusyDevice(const std::vector<double>& flopsPerDevice)
{
  if (flopsPerDevice.empty())
  {
    throw std::logic_error("TensorContraction has no active CUDA devices for work scheduling");
  }

  int result = -1;
  double flops = std::numeric_limits<double>::max();
  int const deviceCount = static_cast<int>(flopsPerDevice.size());

  for (int deviceId = 0; deviceId < deviceCount; deviceId++)
  {
    if (flopsPerDevice[static_cast<std::size_t>(deviceId)] < flops)
    {
      flops = flopsPerDevice[static_cast<std::size_t>(deviceId)];
      result = deviceId;
    }
  }

  return result;
}

namespace tensor
{
namespace
{

struct CoalescedMatrixRange
{
    int deviceId = 0;
    std::size_t begin = 0;
    std::size_t end = 0;
    std::size_t valueOffset = 0;
    std::size_t valueCount = 0;
};

std::size_t matrixElementCount(std::vector<Matrix> const& mats, std::size_t begin, std::size_t end)
{
  std::size_t count = 0;
  for (std::size_t index = begin; index < end; ++index)
  {
    count += mats[index].size();
  }
  return count;
}

std::vector<Matrix> matrixRange(std::vector<Matrix> const& mats, CoalescedMatrixRange range)
{
  return {mats.begin() + static_cast<std::ptrdiff_t>(range.begin),
          mats.begin() + static_cast<std::ptrdiff_t>(range.end)};
}

std::span<double const> valueRange(std::span<double const> values, CoalescedMatrixRange range)
{
  return values.subspan(range.valueOffset, range.valueCount);
}

std::span<double> valueRange(std::span<double> values, CoalescedMatrixRange range)
{
  return values.subspan(range.valueOffset, range.valueCount);
}

std::vector<CoalescedMatrixRange> contiguousCoalescedPlacement(std::vector<Matrix> const& mats, int activeDeviceCount)
{
  if (mats.empty())
  {
    return {};
  }

  std::size_t totalBytes = 0;
  for (auto mat : mats)
  {
    totalBytes += mat.sizeInByte();
  }

  int const rangeCount = std::max(1, std::min(activeDeviceCount, static_cast<int>(mats.size())));
  std::vector<CoalescedMatrixRange> ranges;
  ranges.reserve(static_cast<std::size_t>(rangeCount));

  std::size_t begin = 0;
  std::size_t valueOffset = 0;
  for (int rangeIndex = 0; rangeIndex < rangeCount; ++rangeIndex)
  {
    std::size_t end = begin;
    if (rangeIndex == rangeCount - 1)
    {
      end = mats.size();
    }
    else
    {
      std::size_t const targetBytes =
          (totalBytes * static_cast<std::size_t>(rangeIndex + 1)) / static_cast<std::size_t>(rangeCount);
      std::size_t prefixBytes = 0;
      for (std::size_t index = 0; index < begin; ++index)
      {
        prefixBytes += mats[index].sizeInByte();
      }
      while (end < mats.size() && (end == begin || prefixBytes + mats[end].sizeInByte() <= targetBytes))
      {
        prefixBytes += mats[end].sizeInByte();
        ++end;
      }
      std::size_t const remainingBlocks = mats.size() - end;
      std::size_t const remainingRanges = static_cast<std::size_t>(rangeCount - rangeIndex - 1);
      if (remainingBlocks < remainingRanges)
      {
        end -= remainingRanges - remainingBlocks;
      }
    }

    auto const valueCount = matrixElementCount(mats, begin, end);
    ranges.push_back(CoalescedMatrixRange{
        .deviceId = rangeIndex, .begin = begin, .end = end, .valueOffset = valueOffset, .valueCount = valueCount});
    begin = end;
    valueOffset += valueCount;
  }

  return ranges;
}

bool localizeCoalescedRange(Swapper& swapper, CoalescedMatrixRange range, std::vector<Matrix> const& mats,
                            std::span<double const> values, bool uploadFromHost, bool refreshExisting)
{
  auto const rangeMats = matrixRange(mats, range);
  auto const rangeValues = valueRange(values, range);
  bool const anyExisting = swapper.anyPreStoreBuffer(rangeMats);
  auto const existingDevice = swapper.commonPreStoreDevice(rangeMats);

  if (!anyExisting)
  {
    if (uploadFromHost)
    {
      swapper.uploadHostMatricesCoalesced(rangeMats, rangeValues, range.deviceId);
    }
    else
    {
      swapper.registerGpuAllocationsCoalesced(rangeMats, range.deviceId);
    }
    return true;
  }

  if (!existingDevice.has_value() || *existingDevice != range.deviceId ||
      !swapper.preStoreBuffersAreCoalesced(rangeMats, range.deviceId))
  {
    return false;
  }

  if (uploadFromHost && refreshExisting)
  {
    return swapper.refreshHostMatricesToDeviceCoalesced(rangeMats, rangeValues, range.deviceId);
  }
  return true;
}

} // namespace

Arranger::Arranger(Swapper& swapper) : swapper(swapper)
{
  deviceCount = swapper.getDeviceCount();
  for (int i = 0; i < deviceCount; i++)
  {
    worklistsForInterMat.emplace_back();
    worklistsForTheRest.emplace_back();
    linearAlgebraWorklists.emplace_back();
  }
  linearAlgebraFlopsPerDevice.resize(deviceCount, 0.0);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
}

Arranger::~Arranger() { releaseResources(); }

void Arranger::ensureNcclCommsInitialized()
{
  if (ncclCommsInitialized)
  {
    return;
  }

  int total_ranks = deviceCount * mpi_size;
  if (total_ranks == 1)
  {
    ncclCommsInitialized = true;
    return;
  }

  if (mpi_rank == 0)
  {
    NCCL_CALL(ncclGetUniqueId(&ncclAllDeviceId));
  }

  MPI_Bcast(&ncclAllDeviceId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

  DEBUG_NCCL_COMM_CREATE_START(mpi_rank, total_ranks, &ncclAllDeviceId);
  NCCL_CALL(ncclGroupStart());
  for (int i = 0; i < deviceCount; i++)
  {
    ncclComm_t comm;
    cudaSetDevice(i);
    int global_nccl_rank = mpi_rank * deviceCount + i;
    NCCL_CALL(ncclCommInitRank(&comm, total_ranks, ncclAllDeviceId, global_nccl_rank));
    DEBUG_NCCL_COMM_INIT(mpi_rank, i, global_nccl_rank, total_ranks, comm, &ncclAllDeviceId);
    allDeviceComms.push_back(comm);
  }
  NCCL_CALL(ncclGroupEnd());
  DEBUG_NCCL_COMM_CREATE_END(mpi_rank, total_ranks);
  ncclCommsInitialized = true;
}

void Arranger::ensureMemoryPoolsInitialized()
{
  if (memoryPoolsInitialized)
  {
    return;
  }
  swapper.initMemPools();
  memoryPoolsInitialized = true;
}

void Arranger::preprocess(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                          const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats,
                          const std::vector<TermTy>& fTerms, std::vector<bool>& shouldReuseInter,
                          std::vector<bool>& shouldCombineInter, std::vector<bool>& shouldFinalizeInter,
                          bool shouldAlloc)
{
  shouldReuseInter.assign(fTerms.size(), false);
  shouldCombineInter.assign(fTerms.size(), false);
  shouldFinalizeInter.assign(fTerms.size(), false);

  // First pass: identify which matrices need to be allocated
  std::vector<std::pair<int, int>> interDims;   // dimensions for interMats
  std::vector<int> interIndices;                // indices in interMats vector
  std::vector<std::pair<int, int>> combineDims; // dimensions for combineMats
  std::vector<int> combineIndices;              // indices in combineMats vector

  // Collect interMat dimensions
  for (std::size_t i = 0; i < fTerms.size(); i++)
  {
    if (shouldReuseInter[i])
    {
      continue;
    }

    int bIdx1 = std::get<2>(fTerms[i]);
    int cIdx1 = std::get<3>(fTerms[i]);

    Matrix bMat = bMats[bIdx1];
    Matrix cMat = cMats[cIdx1];

    for (std::size_t j = i + 1; j < fTerms.size(); j++)
    {
      int bIdx2 = std::get<2>(fTerms[j]);
      int cIdx2 = std::get<3>(fTerms[j]);
      if (bIdx1 == bIdx2 && cIdx1 == cIdx2)
      {
        shouldReuseInter[i] = shouldReuseInter[j] = true;
      }
    }

    if (shouldReuseInter[i])
    {
      interDims.push_back({bMat.getFirstDim(), cMat.getSecondDim()});
      interIndices.push_back(static_cast<int>(i));
    }
  }

  // Collect combineMat dimensions
  for (int i = fTerms.size() - 1; i >= 0; i--)
  {
    if (shouldCombineInter[i])
    {
      continue;
    }

    auto [rIdx1, aIdx1, bIdx, cIdx, fval] = fTerms[i];

    for (int j = i - 1; j >= 0; j--)
    {
      int rIdx2 = std::get<0>(fTerms[j]);
      int aIdx2 = std::get<1>(fTerms[j]);

      if (rIdx1 == rIdx2 && aIdx1 == aIdx2)
      {
        shouldCombineInter[j] = shouldCombineInter[i] = shouldFinalizeInter[i] = true;
      }
    }

    if (shouldFinalizeInter[i])
    {
      Matrix bMat = bMats[bIdx];
      Matrix cMat = cMats[cIdx];
      combineDims.push_back({bMat.getFirstDim(), cMat.getSecondDim()});
      combineIndices.push_back(static_cast<int>(i));
    }
  }

  // Second pass: allocate matrices in chunks if needed
  if (shouldAlloc)
  {
    // Allocate and initialize interMats
    if (!interDims.empty())
    {
      for (size_t idx = 0; idx < interIndices.size(); ++idx)
      {
        int termIdx = interIndices[idx];
        int bIdx = std::get<2>(fTerms[termIdx]);
        int cIdx = std::get<3>(fTerms[termIdx]);
        Matrix mat(nullptr, bMats[bIdx].getFirstDim(), cMats[cIdx].getSecondDim());
        mat.setNodeId(mpi_rank);
        interMats.push_back(mat);
        interMatsIdx.push_back({bIdx, cIdx});
      }
    }

    // Allocate and initialize combineMats
    if (!combineDims.empty())
    {
      for (size_t idx = 0; idx < combineIndices.size(); ++idx)
      {
        int termIdx = combineIndices[idx];
        auto [rIdx, aIdx, bIdx, cIdx, fval] = fTerms[termIdx];
        combineMats.push_back(
            std::make_tuple(rIdx, aIdx, Matrix(nullptr, bMats[bIdx].getFirstDim(), cMats[cIdx].getSecondDim())));
      }
    }
  }
}

std::vector<Matrix>& Arranger::getInterMats() { return interMats; }

void Arranger::preStoreToDevice(const std::vector<Matrix>& aMats, const std::vector<Matrix>& bMats,
                                const std::vector<Matrix>& cMats)
{
  int const preStoreDeviceCount = deviceCount;

  // Get available GPU memory for each device
  std::vector<size_t> availableMemory(deviceCount);
  for (int i = 0; i < preStoreDeviceCount; i++)
  {
    // This is a planning budget only. The actual async allocation still
    // reports failure if another process consumes memory later.
    availableMemory[i] = swapper.deviceContext(i).freeMemorySnapshot() / 2;
  }

  // Track memory usage per device
  std::vector<size_t> usedMemory(deviceCount, 0);

  // Create a mapping from matrix ID to new location (device ID)
  // -1 means not assigned to GPU
  std::unordered_map<int, int> matrixToDevice;

  // Simple policy: Round-robin assignment across devices
  int currentDeviceIdx = 0;

  auto assignMatrixToDevice = [&](const std::vector<Matrix>& mats) {
    for (auto mat : mats)
    {
      size_t matSize = mat.sizeInByte();

      // Find a device with enough available memory
      int assignedDevice = -1;
      for (int attempt = 0; attempt < preStoreDeviceCount; attempt++)
      {
        int deviceId = (currentDeviceIdx + attempt) % preStoreDeviceCount;
        if (usedMemory[deviceId] + matSize <= availableMemory[deviceId])
        {
          assignedDevice = deviceId;
          currentDeviceIdx = (deviceId + 1) % preStoreDeviceCount;
          break;
        }
      }

      // If no device has space, don't assign this matrix to GPU
      if (assignedDevice == -1)
      {
        matrixToDevice[mat.getId()] = -1;
      }
      else
      {
        matrixToDevice[mat.getId()] = assignedDevice;
        usedMemory[assignedDevice] += matSize;
      }
    }
  };

  assignMatrixToDevice(interMats);

  assignMatrixToDevice(aMats);
  assignMatrixToDevice(bMats);
  assignMatrixToDevice(cMats);

  auto copyMatrix = [&](const std::vector<Matrix>& mats) {
    for (auto& mat : mats)
    {
      if (int deviceId = matrixToDevice[mat.getId()]; deviceId != -1)
      {
        cudaSetDevice(deviceId);
        swapper.preStoreMatrix(mat, deviceId);
      }
    }
  };

  copyMatrix(interMats);
  copyMatrix(aMats);
  copyMatrix(bMats);
  copyMatrix(cMats);

  for (int i = 0; i < preStoreDeviceCount; i++)
  {
    swapper.syncMemStream(i, "preprocess_upload_inputs");
  }
  CUDA_CALL(cudaSetDevice(0)); // Reset to device 0
}

void Arranger::calculateRFlops(const std::vector<TermTy>& fTerms, const std::vector<Matrix>& rMats,
                               const std::vector<Matrix>& aMats, const std::vector<Matrix>& bMats,
                               const std::vector<Matrix>& cMats, const std::vector<bool>& shouldReuseInter,
                               const std::vector<bool>& shouldCombineInter,
                               const std::vector<bool>& shouldFinalizeInter)
{
  rFlops.assign(rMats.size(), 0.0);

  for (std::size_t i = 0; i < fTerms.size(); i++)
  {
    double flops = 0.0;
    auto [rIdx, aIdx, bIdx, cIdx, fval] = fTerms[i];
    bool shouldReuse = shouldReuseInter[i];
    bool shouldCombine = shouldCombineInter[i];
    bool shouldFinalize = shouldFinalizeInter[i];

    auto rMat = rMats[rIdx];
    auto aMat = aMats[aIdx];
    auto bMat = bMats[bIdx];
    auto cMat = cMats[cIdx];

    if (!shouldReuse)
    {
      flops += bMat.getFirstDim() * cMat.getSecondDim() * bMat.getSecondDim();
    }

    if (shouldCombine)
    {
      flops += 2.0 * bMat.getFirstDim() * cMat.getSecondDim();

      if (shouldFinalize)
      {
        flops += rMat.size() + aMat.getFirstDim() * cMat.getSecondDim() * aMat.getSecondDim();
      }
    }
    else
    {
      flops += 2.0 * rMat.size() + aMat.getFirstDim() * cMat.getSecondDim() * aMat.getSecondDim();
    }

    rFlops[rIdx] += flops;
  }
}

void Arranger::enableP2PPeerAccess()
{
  // Enable P2P access between all device pairs for direct GPU-to-GPU
  // communication
  for (int i = 0; i < deviceCount; i++)
  {
    for (int j = 0; j < deviceCount; j++)
    {
      if (i != j)
      {
        int canAccess;
        CUDA_CALL(cudaDeviceCanAccessPeer(&canAccess, i, j));
        DEBUG_P2P_AVAILABLE(i, j, canAccess);
        if (canAccess)
        {
          CUDA_CALL(cudaSetDevice(i));
          // Ignore error if P2P is already enabled
          [[maybe_unused]] cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
          DEBUG_P2P_ENABLED(i, j, (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled));
          cudaGetLastError(); // Clear any error state
        }
      }
    }
  }

  // Configure memory pool access for P2P transfers
  for (int i = 0; i < deviceCount; i++)
  {
    cudaSetDevice(i);

    // Get the default memory pool for this device
    cudaMemPool_t pool;
    CUDA_CALL(cudaDeviceGetDefaultMemPool(&pool, i));

    for (int j = 0; j < deviceCount; j++)
    {
      if (i == j) continue; // Skip self

      int canAccess;
      CUDA_CALL(cudaDeviceCanAccessPeer(&canAccess, i, j));

      if (canAccess)
      {
        DEBUG_P2P_AVAILABLE(i, j, canAccess);
        cudaMemAccessDesc desc = {};
        desc.location.type = cudaMemLocationTypeDevice;
        desc.location.id = j;
        desc.flags = cudaMemAccessFlagsProtReadWrite;
        [[maybe_unused]] cudaError_t err = cudaMemPoolSetAccess(pool, &desc, 1);
        DEBUG_P2P_ENABLED(i, j, (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled));
      }
    }
  }
}

void Arranger::analyzeComputation(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                                  const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats,
                                  const std::vector<TermTy>& fTerms)
{
  CUDA_CALL(cudaSetDevice(0)); // Reset to device 0

  sortedFTerms = fTerms;

  preprocess(rMats, aMats, bMats, cMats, fTerms, shouldReuseInter, shouldCombineInter, shouldFinalizeInter,
             /*shouldAlloc=*/false);

  calculateRFlops(fTerms, rMats, aMats, bMats, cMats, shouldReuseInter, shouldCombineInter, shouldFinalizeInter);

  std::sort(sortedFTerms.begin(), sortedFTerms.end(), [&](const TermTy& term1, const TermTy& term2) {
    int rIdx1 = std::get<0>(term1);
    int aIdx1 = std::get<1>(term1);
    int rIdx2 = std::get<0>(term2);
    int aIdx2 = std::get<1>(term2);
    if (rFlops[rIdx1] != rFlops[rIdx2])
    {
      return rFlops[rIdx1] > rFlops[rIdx2];
    }

    if (rIdx1 != rIdx2)
    {
      return rIdx1 < rIdx2;
    }

    return aIdx1 < aIdx2;
  });

  preprocess(rMats, aMats, bMats, cMats, sortedFTerms, shouldReuseInter, shouldCombineInter, shouldFinalizeInter,
             /*shouldAlloc=*/true);

  // Sort interMats and interMatsIdx together by flops (descending)
  std::vector<int> sortOrder(interMats.size());
  std::iota(sortOrder.begin(), sortOrder.end(), 0);
  std::sort(sortOrder.begin(), sortOrder.end(), [&](int i, int j) {
    auto [bIdx1, cIdx1] = interMatsIdx[i];
    auto [bIdx2, cIdx2] = interMatsIdx[j];

    double flops1 = (double)bMats[bIdx1].getFirstDim() * bMats[bIdx1].getSecondDim() * cMats[cIdx1].getSecondDim();

    double flops2 = (double)bMats[bIdx2].getFirstDim() * bMats[bIdx2].getSecondDim() * cMats[cIdx2].getSecondDim();
    return flops1 > flops2;
  });
  {
    std::vector<Matrix> sortedMats(interMats.size());
    std::vector<std::pair<int, int>> sortedIdx(interMatsIdx.size());
    for (std::size_t i = 0; i < sortOrder.size(); i++)
    {
      sortedMats[i] = interMats[sortOrder[i]];
      sortedIdx[i] = interMatsIdx[sortOrder[i]];
    }
    interMats = std::move(sortedMats);
    interMatsIdx = std::move(sortedIdx);
  }

#if DEBUG_LOG
  initializeDebugInfo(rMats, aMats, bMats, cMats);
#endif
  ensureMemoryPoolsInitialized();
}

void Arranger::compileWorklists(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                                const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats,
                                bool syncResultsToHost)
{
  std::vector<double> flopsPerDevice(deviceCount, 0.0);

  // The contraction is split into two parts:
  //   1. Calculate the repeated B*C result.
  //   2. Calculate the rest of the contraction.
  compileWorklistsForInterMat(rMats, aMats, bMats, cMats, flopsPerDevice);
  compileWorklistsForTheRest(rMats, aMats, bMats, cMats, flopsPerDevice, syncResultsToHost);
  buildLiveInterval(worklistsForInterMat, liveIntervalsForInterMat);
  buildLiveInterval(worklistsForTheRest, liveIntervalsForTheRest);
}

void Arranger::buildLiveInterval(std::vector<WorklistTy>& worklists, std::vector<LiveIntervalMap>& liveIntervals)
{
  liveIntervals.resize(worklists.size());

  for (int deviceId = 0; deviceId < deviceCount; deviceId++)
  {
    std::unordered_map<Matrix, std::pair<int, int>> matToIntervalMap;
    auto& worklist = worklists[deviceId];
    auto& intervals = liveIntervals[deviceId];
    intervals.clear();
    for (int i = 0; i < (int)worklist.size(); i++)
    {
      auto matWork = std::dynamic_pointer_cast<MatWorkBase>(worklist[i]);
      if (!matWork) continue;
      for (const auto& mat : matWork->getMatrices())
      {
        auto it = matToIntervalMap.find(mat);
        if (it == matToIntervalMap.end())
        {
          matToIntervalMap[mat] = {i, i};
        }
        else
        {
          it->second.second = i;
        }
      }
    }

    for (auto [mat, interval] : matToIntervalMap)
    {
      intervals.push_back({interval.first, interval.second, mat});
    }

    std::sort(intervals.begin(), intervals.end());
  }
}

int Arranger::leastBusyContractionDevice(const std::vector<double>& flopsPerDevice) const
{
  return getLeastBusyDevice(flopsPerDevice);
}

int Arranger::leastBusyLinearAlgebraDevice() const { return getLeastBusyDevice(linearAlgebraFlopsPerDevice); }

void Arranger::compileWorklistsForInterMat(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                                           const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats,
                                           std::vector<double>& flopsPerDevice)
{
  for (std::size_t i = 0; i < interMats.size(); i++)
  {
    auto [bIdx, cIdx] = interMatsIdx[i];
    Matrix interMat = interMats[i];
    Matrix bMat = bMats[bIdx];
    Matrix cMat = cMats[cIdx];

    int leastBusyDeviceId = this->leastBusyContractionDevice(flopsPerDevice);
    cudaSetDevice(leastBusyDeviceId);
    auto& worklist = worklistsForInterMat[leastBusyDeviceId];

    cudaEvent_t syncFinishEvent = swapper.dependencyEventsActive() ? createSyncFinishEvent() : nullptr;
    matToSyncFinishEventMap[interMat.getId()] = syncFinishEvent;

    worklist.push_back(
        createWork<MatMulWork>(std::vector<Matrix>{interMat, bMat, cMat}, 1.0, leastBusyDeviceId, swapper));
    worklist.push_back(
        createWork<SyncWork>(std::vector<Matrix>{interMat}, 0.0, leastBusyDeviceId, swapper, syncFinishEvent));
    flopsPerDevice[leastBusyDeviceId] +=
        (double)bMat.getFirstDim() * (double)cMat.getSecondDim() * (double)bMat.getSecondDim();
  }
}

#if DEBUG_LOG
void Arranger::initializeDebugInfo(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                                   const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats)
{
  for (int i = 0; i < rMats.size(); i++)
  {
    swapper.setMatrixName(rMats[i], "R" + std::to_string(i));
  }
  for (int i = 0; i < aMats.size(); i++)
  {
    swapper.setMatrixName(aMats[i], "A" + std::to_string(i));
  }
  for (int i = 0; i < bMats.size(); i++)
  {
    swapper.setMatrixName(bMats[i], "B" + std::to_string(i));
  }
  for (int i = 0; i < cMats.size(); i++)
  {
    swapper.setMatrixName(cMats[i], "C" + std::to_string(i));
  }
  for (int i = 0; i < interMats.size(); i++)
  {
    swapper.setMatrixName(interMats[i], "inter" + std::to_string(i));
  }
}
#endif

void Arranger::compileForSingleR(int fTermsStart, int fTermsEnd, const Matrix rMat, const std::vector<Matrix>& aMats,
                                 const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats,
                                 std::vector<double>& flopsPerDevice, bool syncResultToHost)
{
  Matrix combineMat;
  bool shouldMemsetCombineMat = true;
  bool shouldAddAccuRMat = false;

  // Resident no-host-sync outputs must be produced on their pre-store device;
  // otherwise the transient result is freed before the next vector operation.
  auto [residentDeviceId, _] = swapper.getPreStoreBufferOrNone(rMat);
  int currentDeviceId = residentDeviceId == -1 ? this->leastBusyContractionDevice(flopsPerDevice) : residentDeviceId;
  WorklistTy& worklist = worklistsForTheRest[currentDeviceId];

  for (int i = fTermsStart; i < fTermsEnd; i++)
  {
    double flops = 0.0;
    int aIdx = std::get<1>(sortedFTerms[i]);
    int bIdx = std::get<2>(sortedFTerms[i]);
    int cIdx = std::get<3>(sortedFTerms[i]);
    double fval = std::get<4>(sortedFTerms[i]);

    bool shouldReuse = shouldReuseInter[i];
    bool shouldCombine = shouldCombineInter[i];
    bool shouldFinalize = shouldFinalizeInter[i];

    auto aMat = aMats[aIdx];
    auto bMat = bMats[bIdx];
    auto cMat = cMats[cIdx];

    Matrix interMat(nullptr, bMat.getFirstDim(), cMat.getSecondDim());
    cudaEvent_t syncFinishEvent = nullptr;

    if (shouldReuse)
    {
      auto it = std::find_if(interMatsIdx.begin(), interMatsIdx.end(), [bIdx, cIdx](const std::pair<int, int>& elem) {
        return elem.first == bIdx && elem.second == cIdx;
      });
      assert(it != interMatsIdx.end() && "No intermediate result found");
      interMat = interMats[it - interMatsIdx.begin()];
      syncFinishEvent = matToSyncFinishEventMap.find(interMat.getId())->second;
    }
    else
    {
      flops += bMat.getFirstDim() * cMat.getSecondDim() * bMat.getSecondDim();
      worklist.push_back(
          createWork<MatMulWork>(std::vector<Matrix>{interMat, bMat, cMat}, 1.0, currentDeviceId, swapper));
    }

    if (shouldCombine)
    {
      if (shouldMemsetCombineMat)
      {
        combineMat = Matrix(nullptr, bMat.getFirstDim(), cMat.getSecondDim());
        worklist.push_back(createWork<MemsetWork>(std::vector<Matrix>{combineMat}, 0.0, currentDeviceId, swapper));
        shouldMemsetCombineMat = false;
      }
      flops += 2 * interMat.size();
      worklist.push_back(createWork<AddAccuWork>(std::vector<Matrix>{combineMat, interMat}, fval, currentDeviceId,
                                                 swapper, syncFinishEvent));

      if (!shouldFinalize)
      {
        // Not finalizing yet, skip R accumulation
        flopsPerDevice[currentDeviceId] += flops;
        continue;
      }

      interMat = combineMat;
      fval = 1.0;
    }

    if (shouldAddAccuRMat)
    {
      flops += aMat.getFirstDim() * interMat.getSecondDim() * aMat.getSecondDim() + 2 * rMat.size();
      worklist.push_back(createWork<MatMulAccuWork>(std::vector<Matrix>{rMat, aMat, interMat}, fval, currentDeviceId,
                                                    swapper, syncFinishEvent));
    }
    else
    {
      flops += aMat.getFirstDim() * interMat.getSecondDim() * aMat.getSecondDim();
      worklist.push_back(createWork<MatMulWork>(std::vector<Matrix>{rMat, aMat, interMat}, fval, currentDeviceId,
                                                swapper, syncFinishEvent));
      shouldAddAccuRMat = true;
    }
    if (shouldFinalize)
    {
      // Combined intermediates represent sum_i f_i * B_i*C_i for a single
      // (R,A) group.  The next A group for the same R must start from zero.
      shouldMemsetCombineMat = true;
    }

    flopsPerDevice[currentDeviceId] += flops;
  }

  if (syncResultToHost)
  {
    worklistsForTheRest[currentDeviceId].push_back(
        createWork<SyncWork>(std::vector<Matrix>{rMat}, 0.0, currentDeviceId, swapper));
  }
}

void Arranger::compileWorklistsForTheRest(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                                          const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats,
                                          std::vector<double>& flopsPerDevice, bool syncResultsToHost)
{
  const int sortedFTermCount = static_cast<int>(sortedFTerms.size());
  for (int i = 0; i < sortedFTermCount;)
  {
    int fTermsStart = i;
    int fTermsEnd = i;

    while (fTermsEnd < sortedFTermCount &&
           std::get<0>(sortedFTerms[fTermsEnd]) == std::get<0>(sortedFTerms[fTermsStart]))
    {
      fTermsEnd++;
    }

    int rIdx = std::get<0>(sortedFTerms[fTermsStart]);
    compileForSingleR(fTermsStart, fTermsEnd, rMats[rIdx], aMats, bMats, cMats, flopsPerDevice, syncResultsToHost);
    i = fTermsEnd;
  }
}

cudaEvent_t Arranger::createSyncFinishEvent()
{
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  syncFinishEvents.push_back(event);
  return event;
}

static std::unordered_set<Matrix> estimateAllocateableMatrices(const Arranger::WorklistTy& worklist, size_t freeMem,
                                                               int currentIdx)
{
  std::unordered_set<Matrix> allocateableMatrices;
  size_t accuSize = 0;

  // estimate which matrices could possibly be allocated.
  for (int i = currentIdx; i < static_cast<int>(worklist.size()); i++)
  {
    auto matWork = std::dynamic_pointer_cast<MatWorkBase>(worklist[i]);
    if (!matWork)
    {
      continue;
    }

    for (auto mat : matWork->getMatrices())
    {
      if (allocateableMatrices.find(mat) != allocateableMatrices.end())
      {
        continue;
      }
      if (accuSize + mat.sizeInByte() < freeMem)
      {
        allocateableMatrices.insert(mat);
        accuSize += mat.sizeInByte();
      }
      else
      {
        return allocateableMatrices;
      }
    }
  }

  return allocateableMatrices;
}

void Arranger::mpiExchangeCopies(const std::vector<std::vector<int>>& tokensNeededFromRank,
                                 const std::unordered_map<int, Matrix>& neededMatMap)
{
  std::vector<int> requestCounts(mpi_size, 0);
  for (int r = 0; r < mpi_size; r++)
    requestCounts[r] = tokensNeededFromRank[r].size();

  std::vector<int> supplyToOther(mpi_size, 0);
  MPI_Alltoall(requestCounts.data(), 1, MPI_INT, supplyToOther.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> sendDispls(mpi_size, 0);
  for (int i = 1; i < mpi_size; i++)
    sendDispls[i] = sendDispls[i - 1] + requestCounts[i - 1];
  int totalRequest = sendDispls.back() + requestCounts.back();

  std::vector<int> sendTokens(totalRequest);
  for (int r = 0; r < mpi_size; r++)
  {
    int offset = sendDispls[r];
    for (int k = 0; k < (int)tokensNeededFromRank[r].size(); k++)
      sendTokens[offset + k] = tokensNeededFromRank[r][k];
  }

  std::vector<int> recvDispls(mpi_size, 0);
  for (int i = 1; i < mpi_size; i++)
    recvDispls[i] = recvDispls[i - 1] + supplyToOther[i - 1];
  int totalSupply = recvDispls.back() + supplyToOther.back();

  std::vector<int> recvTokens(totalSupply);
  MPI_Alltoallv(sendTokens.data(), requestCounts.data(), sendDispls.data(), MPI_INT, recvTokens.data(),
                supplyToOther.data(), recvDispls.data(), MPI_INT, MPI_COMM_WORLD);

  std::vector<MPI_Request> requests;

  for (int r = 0; r < mpi_size; r++)
  {
    int base = recvDispls[r];
    for (int k = 0; k < supplyToOther[r]; k++)
    {
      int token = recvTokens[base + k];
      auto m = localMats.at(token);
      auto host = m.hostView();
      MPI_Request req;
      MPI_Isend(host.requireData(), m.size(), MPI_DOUBLE, r, token, MPI_COMM_WORLD, &req);
      requests.push_back(req);
    }
  }

  for (int r = 0; r < mpi_size; r++)
  {
    for (int token : tokensNeededFromRank[r])
    {
      auto m = neededMatMap.at(token);
      MPI_Request req;
      MPI_Irecv(m.hostView().requireData(), m.size(), MPI_DOUBLE, r, token, MPI_COMM_WORLD, &req);
      requests.push_back(req);
    }
  }

  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
}

void Arranger::preCopyMatrices(std::vector<Matrix>& aMats, std::vector<Matrix>& bMats, std::vector<Matrix>& cMats,
                               MatrixAllocator& allocator)
{
  if (mpi_size <= 1) return;

  // 1. Analyze sortedFTerms to know which matrices are needed.
  std::set<int> neededAIdx, neededBIdx, neededCIdx;
  for (const auto& term : sortedFTerms)
  {
    neededAIdx.insert(std::get<1>(term));
    neededBIdx.insert(std::get<2>(term));
    neededCIdx.insert(std::get<3>(term));
  }

  // 2. Determine where do these matrices locate
  //   a. locate on this node ==> do nothing.
  //   b. locate on other node's GPU ===> do nothing.
  //   c. locate on other node's CPU ===> We need to copy it to this node!
  std::vector<std::vector<Matrix>> matsFromRank(mpi_size);

  auto collectRemoteCpuMats = [&](std::vector<Matrix>& mats, const std::set<int>& needed) {
    for (int idx : needed)
    {
      Matrix& mat = mats[idx];
      int owner = mat.getNodeId();
      if (owner == mpi_rank) continue;
      if (swapper.isOnRemoteGpu(mat.getId())) continue;
      matsFromRank[owner].push_back(mat);
    }
  };

  collectRemoteCpuMats(aMats, neededAIdx);
  collectRemoteCpuMats(bMats, neededBIdx);
  collectRemoteCpuMats(cMats, neededCIdx);

  // Allocate CPU memory for incoming matrices before posting Irecvs
  // setPtr propagates to originals in aMats/bMats/cMats via shared impl
  std::vector<Matrix> incomingMats;
  for (int r = 0; r < mpi_size; r++)
    for (Matrix mat : matsFromRank[r])
      incomingMats.push_back(mat);
  if (!incomingMats.empty()) allocator.allocateMatrices(incomingMats);

  std::vector<std::vector<int>> tokensNeededFromRank(mpi_size);
  std::unordered_map<int, Matrix> neededMatMap;
  for (int r = 0; r < mpi_size; r++)
    for (Matrix mat : matsFromRank[r])
    {
      tokensNeededFromRank[r].push_back(mat.getId());
      neededMatMap[mat.getId()] = mat;
    }

  mpiExchangeCopies(tokensNeededFromRank, neededMatMap);
}

static std::vector<Matrix> allocateMatrices(const Arranger::WorklistTy& worklist, int& endIdx, int deviceId,
                                            Swapper& swapper)
{
  std::vector<Matrix> matricesToCopy;

  for (; endIdx < static_cast<int>(worklist.size()); endIdx++)
  {
    auto matWork = std::dynamic_pointer_cast<MatWorkBase>(worklist[endIdx]);
    if (!matWork)
    {
      continue;
    }

    for (auto mat : matWork->getMatrices())
    {
      if (!swapper.getGpuBufferOrNone(mat, deviceId))
      {
        auto buffer = swapper.allocate(mat, deviceId);
        if (buffer == nullptr)
        {
          return matricesToCopy;
        }

        bool isResultOfMatMul = std::dynamic_pointer_cast<MatMulWork>(matWork) && mat == matWork->getMatrices()[0];
        bool isResultOfMemset = std::dynamic_pointer_cast<MemsetWork>(matWork) && mat == matWork->getMatrices()[0];

        if (!isResultOfMatMul && !isResultOfMemset)
        {
          matricesToCopy.push_back(mat);
        }
      }
      else
      {
        auto [id, buffer] = swapper.getPreStoreBufferOrNone(mat);
        if (buffer != nullptr && id == deviceId)
        {
          DEBUG_HIT(swapper, mat.getId(), deviceId, mat.sizeInByte());
        }
      }
    }
  }

  return matricesToCopy;
}

static void analyzeLiveMatrices(Arranger::WorklistTy& worklist, int currentIdx,
                                std::unordered_set<Matrix>& unfreeableMatrices,
                                const Arranger::LiveIntervalMap& liveIntervalMap)
{
  for (const auto& [firstUse, lastUse, mat] : liveIntervalMap)
  {
    if (firstUse < currentIdx && lastUse >= currentIdx)
    {
      unfreeableMatrices.insert(mat);
    }
  }
}

// Free unused matrices and allocate buffers for the next batch of work.
// Returns the list of matrices that need to be copied to this device.
static std::vector<Matrix> freeAndAllocate(int deviceId, Arranger::WorklistTy& worklist, int& endIdx, size_t freeMem,
                                           Swapper& swapper, const Arranger::LiveIntervalMap& liveIntervalMap,
                                           bool allowFreeUnpinned)
{
  CUDA_CALL(cudaSetDevice(deviceId));
  std::unordered_set<Matrix> unfreeableMatrices = estimateAllocateableMatrices(worklist, freeMem, endIdx);
  // analyze which matrices cannot be free'd
  // Matrices who are still alive at endIdx cannot be free'd
  analyzeLiveMatrices(worklist, endIdx, unfreeableMatrices, liveIntervalMap);

  for (auto mat : unfreeableMatrices)
  {
    swapper.pinMatrix(mat, deviceId);
  }
  // In single-process multi-GPU mode, a buffer can be dead in its producer
  // device's local worklist while still being needed as the source for another
  // device's copy.  Keep multi-GPU eviction conservative until the scheduler
  // tracks cross-device source lifetimes explicitly.
  if (allowFreeUnpinned)
  {
    swapper.freeAllUnpinMatrices(deviceId);
  }
  for (auto mat : unfreeableMatrices)
  {
    swapper.unpinMatrix(mat, deviceId);
  }

  // allocate matrices
  return allocateMatrices(worklist, endIdx, deviceId, swapper);
}

// Copy matrices to this device's GPU. Matrices in CPU memory are copied
// directly; matrices on other GPUs are collected into dstMatPairs for
// NCCL send/recv.
static void copyLocalMatrices(int deviceId, const std::vector<Matrix>& matricesToCopy, Swapper& swapper,
                              const std::unordered_map<int, cudaEvent_t>& matToSyncFinishEventMap)
{
  CUDA_CALL(cudaSetDevice(deviceId));

  for (auto mat : matricesToCopy)
  {
    auto [srcDeviceId, srcBuffer] = swapper.findLocalSourceBuffer(mat, deviceId);
    const auto it = matToSyncFinishEventMap.find(mat.getId());
    cudaEvent_t syncFinishEvent = it == matToSyncFinishEventMap.end() ? nullptr : it->second;
    if (srcBuffer)
    {
      if (srcDeviceId == deviceId)
      {
        continue;
      }

      auto streamOwner = swapper.deviceContext(deviceId).leaseWorkStream();
      cudaStream_t stream = streamOwner.stream();
      auto dstBuffer = swapper.getForWrite(mat, deviceId, stream);
      swapper.waitForAccessDependencies({srcBuffer}, {dstBuffer}, stream);
      if (syncFinishEvent)
      {
        CUDA_CALL(cudaStreamWaitEvent(stream, syncFinishEvent));
      }
      CUDA_CALL(cudaMemcpyPeerAsync(dstBuffer->getPtr(), deviceId, srcBuffer->getPtr(), srcDeviceId, mat.sizeInByte(),
                                    stream));
      auto completion = streamOwner.recordCompletion();
      swapper.publishAccessCompletion({srcBuffer}, {dstBuffer}, stream, completion);
      DEBUG_SECOND_HIT(swapper, mat.getId(), srcDeviceId, deviceId, mat.sizeInByte());
    }
    else if (mat.hasHostStorage())
    {
      auto streamOwner = swapper.deviceContext(deviceId).leaseWorkStream();
      cudaStream_t stream = streamOwner.stream();
      auto buffer = swapper.getForWrite(mat, deviceId, stream);
      if (syncFinishEvent)
      {
        CUDA_CALL(cudaStreamWaitEvent(stream, syncFinishEvent));
      }
      swapper.copyHostToDevice(mat.hostView(), buffer, deviceId, stream);
    }
    else
    {
      throw std::logic_error("TensorContraction tried to copy a hostless matrix with no local GPU source");
    }
  }
}

static void copyMatrices(int deviceId, int deviceCount, const std::vector<Matrix>& matricesToCopy,
                         std::vector<std::tuple<int, Matrix, cudaEvent_t>>& dstMatPairs, std::atomic_int& counter,
                         const std::function<ncclComm_t()>& commProvider, Swapper& swapper,
                         const std::unordered_map<int, cudaEvent_t>& matToSyncFinishEventMap)
{
  CUDA_CALL(cudaSetDevice(deviceId));

  // `dstMatPairs` records which GPU needs which matrices.
  // It is shared across multiple threads that control other GPUs.
  // So we need synchronization here to prevent race condition.
  while (counter < deviceId)
  {}

  int device_count;
  int mpi_rank;

  CUDA_CALL(cudaGetDeviceCount(&device_count));
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  for (auto mat : matricesToCopy)
  {
    // Trying to figure out where does this matrix reside..
    auto [srcDeviceId, srcBuffer] = swapper.findLocalSourceBuffer(mat, deviceId);
    const auto it = matToSyncFinishEventMap.find(mat.getId());
    cudaEvent_t syncFinishEvent = it == matToSyncFinishEventMap.end() ? nullptr : it->second;
    if (srcBuffer)
    {
      if (srcDeviceId == deviceId)
      {
        // The matrix is already resident on the requesting device.  Do not add
        // it to the NCCL metadata path: for the one-device-per-MPI-rank launch
        // shape this local reuse was triggering global collectives with no data
        // movement.
        continue;
      }

      // This matrix resides on another GPU in this process.  Use a direct
      // peer copy rather than routing local traffic through NCCL; the NCCL
      // path is kept for inter-process transfers.
      auto streamOwner = swapper.deviceContext(deviceId).leaseWorkStream();
      cudaStream_t stream = streamOwner.stream();
      auto dstBuffer = swapper.getForWrite(mat, deviceId, stream);
      swapper.waitForAccessDependencies({srcBuffer}, {dstBuffer}, stream);
      if (syncFinishEvent)
      {
        CUDA_CALL(cudaStreamWaitEvent(stream, syncFinishEvent));
      }
      CUDA_CALL(cudaMemcpyPeerAsync(dstBuffer->getPtr(), deviceId, srcBuffer->getPtr(), srcDeviceId, mat.sizeInByte(),
                                    stream));
      auto completion = streamOwner.recordCompletion();
      swapper.publishAccessCompletion({srcBuffer}, {dstBuffer}, stream, completion);
      DEBUG_SECOND_HIT(swapper, mat.getId(), srcDeviceId, deviceId, mat.sizeInByte());
    }
    else if (mat.hasHostStorage())
    {
      // This matrix sits in CPU memory.
      auto streamOwner = swapper.deviceContext(deviceId).leaseWorkStream();
      cudaStream_t stream = streamOwner.stream();
      auto buffer = swapper.getForWrite(mat, deviceId, stream);
      if (syncFinishEvent)
      {
        CUDA_CALL(cudaStreamWaitEvent(stream, syncFinishEvent));
      }
      // Just copy it from CPU memory.
      swapper.copyHostToDevice(mat.hostView(), buffer, deviceId, stream);
    }
    else
    {
      // Hostless matrices are TensorContraction intermediates.  If they reach
      // this path without a remote owner, the scheduler has lost their producer
      // buffer and must not continue with a newly allocated zero-filled buffer.
      if (!swapper.isOnRemoteGpu(mat.getId()))
      {
        throw std::logic_error("TensorContraction tried to copy a hostless matrix with no GPU source");
      }
      int nccl_comm_id = mpi_rank * device_count + deviceId;
      dstMatPairs.push_back({nccl_comm_id, mat, nullptr});
      DEBUG_THIRD_HIT(swapper, mat.getId(), nccl_comm_id, deviceId, mat.sizeInByte());
    }
  }
  counter++;

  while (counter < deviceCount)
  {}

  if (deviceId == 0)
  {
    // Use MPI_Allgatherv to share dstMatPairs across all nodes.
    // After this, every thread on every node sees the complete set of pairs.
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    const size_t entry_size = sizeof(int) + sizeof(Matrix::Header);
    int local_count = dstMatPairs.size();

    std::vector<int> all_counts(mpi_size);
    MPI_Allgather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_entries = std::accumulate(all_counts.begin(), all_counts.end(), 0);
    if (total_entries == 0)
    {
      counter++;
      while (counter < deviceCount + 1)
      {}
      return;
    }

    std::vector<int> recvcounts(mpi_size);
    std::vector<int> displs(mpi_size);
    total_entries = 0;
    for (int i = 0; i < mpi_size; i++)
    {
      recvcounts[i] = all_counts[i] * entry_size;
      displs[i] = total_entries * entry_size;
      total_entries += all_counts[i];
    }

    std::vector<char> sendbuf(local_count * entry_size);
    for (int i = 0; i < local_count; i++)
    {
      auto& [nccl_id, mat, _] = dstMatPairs[i];
      Matrix::Header h = mat.toHeader();
      memcpy(sendbuf.data() + i * entry_size, &nccl_id, sizeof(int));
      memcpy(sendbuf.data() + i * entry_size + sizeof(int), &h, sizeof(Matrix::Header));
    }

    std::vector<char> recvbuf(total_entries * entry_size);
    MPI_Allgatherv(sendbuf.data(), local_count * entry_size, MPI_BYTE, recvbuf.data(), recvcounts.data(), displs.data(),
                   MPI_BYTE, MPI_COMM_WORLD);

    for (int rank = 0; rank < mpi_size; rank++)
    {
      if (rank == mpi_rank) continue;
      int base = displs[rank];
      for (int i = 0; i < all_counts[rank]; i++)
      {
        int nccl_id;
        Matrix::Header h;
        memcpy(&nccl_id, recvbuf.data() + base + i * entry_size, sizeof(int));
        memcpy(&h, recvbuf.data() + base + i * entry_size + sizeof(int), sizeof(Matrix::Header));
        Matrix mat(h);
        dstMatPairs.push_back({nccl_id, mat, nullptr});
      }
    }

    counter++;
  }

  while (counter < deviceCount + 1)
  {}

  // Let NCCL do the real matrices copy work.
  if (!dstMatPairs.empty())
  {
    ncclComm_t comm = commProvider();
    assert(comm != nullptr);
    auto work = createWork<NCCLSendRecvWork>(dstMatPairs, comm, deviceId, swapper);
    work->execute();
    if (!swapper.dependencyEventsActive())
    {
      return;
    }
    auto completion = work->completion();
    for (auto mat : work->readMatrices())
    {
      swapper.notifyMatrixRead(mat, deviceId, completion);
    }
    for (auto mat : work->writeMatrices())
    {
      swapper.notifyMatrixWrite(mat, deviceId, completion);
    }
  }
}

void Arranger::executeWorklists(std::vector<WorklistTy>& worklists, std::vector<LiveIntervalMap>& liveIntervals,
                                bool freeBuffersAtEnd)
{
  std::vector<int> scheduledDevices;
  scheduledDevices.reserve(static_cast<std::size_t>(deviceCount));
  for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
  {
    if (mpi_size == 1 && deviceCount > 1 && worklists[deviceId].empty())
    {
      continue;
    }
    scheduledDevices.push_back(deviceId);
  }
  if (scheduledDevices.empty())
  {
    return;
  }
  bool const localSingleScheduledDevice = mpi_size == 1 && scheduledDevices.size() == 1;
  int const effectiveDeviceCount = localSingleScheduledDevice ? 1 : deviceCount;

  std::vector<size_t> freeMems(deviceCount);

  for (int i : scheduledDevices)
  {
    // Worklist batching only needs a conservative capacity estimate. Querying
    // the driver for every tiny DMRG contraction phase dominated the CUDA API
    // profile, so use the device-context snapshot and let allocation failures
    // provide the exact backstop.
    freeMems[i] = swapper.deviceContext(i).freeMemorySnapshot();
  }

  for (int i : scheduledDevices)
  {
    (void)i;
    DEBUG_ARRANGER_FREE_MEM(i, freeMems[i]);
  }

  CUDA_CALL(cudaSetDevice(0));

  auto execute = [&](WorklistTy& worklist, int deviceId, int startIdx, int endIdx) {
    CUDA_CALL(cudaSetDevice(deviceId));
    if (startIdx == endIdx)
    {
      return;
    }
    for (int idx = startIdx; idx < endIdx; idx++)
    {
      auto& work = worklist[idx];
      work->execute();

      auto reads = work->readMatrices();
      auto writes = work->writeMatrices();
      if (reads.empty() && writes.empty())
      {
        continue;
      }
      if (!swapper.dependencyEventsActive())
      {
        continue;
      }

      auto completion = work->completion();
      if (completion == nullptr)
      {
        continue;
      }
      std::vector<std::shared_ptr<GpuBuffer>> readBuffers;
      std::vector<std::shared_ptr<GpuBuffer>> writeBuffers;
      readBuffers.reserve(reads.size());
      writeBuffers.reserve(writes.size());
      for (auto mat : reads)
      {
        readBuffers.push_back(swapper.getGpuBufferOrNone(mat, deviceId));
      }
      for (auto mat : writes)
      {
        writeBuffers.push_back(swapper.getGpuBufferOrNone(mat, deviceId));
      }
      swapper.publishAccessCompletion(readBuffers, writeBuffers, completion->stream(), completion);
    }
  };

  std::vector<std::thread> threads(deviceCount);

#if DEBUG_LOG
  for (int deviceId = 0; deviceId < deviceCount; deviceId++)
  {
    for (int idx = 0; idx < worklists[deviceId].size(); idx++)
    {
      fprintf(stderr, "%d: ", idx);
      worklists[deviceId][idx]->dump();
    }
  }
#endif

  std::vector<int> startIdxes(deviceCount);
  std::vector<int> endIdxes(deviceCount);

  while (true)
  {
    // Step 1: Free unused matrices and allocate buffers.
    std::vector<std::vector<Matrix>> matricesToCopy(deviceCount);
    if (deviceCount == 1)
    {
      // The common TensorContraction/MPI launch shape is one CUDA device per
      // process.  Avoid creating a worker thread only to join it immediately;
      // Nsight showed this dominating tiny-block DMRG benchmarks.
      matricesToCopy[0] = freeAndAllocate(0, worklists[0], endIdxes[0], freeMems[0], swapper, liveIntervals[0], true);
    }
    else if (mpi_size == 1)
    {
      // With one host process controlling local GPUs, CUDA work can be queued
      // sequentially on each device without losing device-side overlap.  This
      // avoids thousands of short-lived host threads in tiny-block workloads.
      for (int deviceId : scheduledDevices)
      {
        matricesToCopy[deviceId] =
            freeAndAllocate(deviceId, worklists[deviceId], endIdxes[deviceId], freeMems[deviceId], swapper,
                            liveIntervals[deviceId], localSingleScheduledDevice);
      }
    }
    else
    {
      for (int deviceId = 0; deviceId < deviceCount; deviceId++)
      {
        threads[deviceId] = std::thread([&, deviceId] {
          matricesToCopy[deviceId] = freeAndAllocate(deviceId, worklists[deviceId], endIdxes[deviceId],
                                                     freeMems[deviceId], swapper, liveIntervals[deviceId], false);
        });
      }

      for (int deviceId = 0; deviceId < deviceCount; deviceId++)
      {
        threads[deviceId].join();
      }
    }

    // Step 2: Copy matrices to their assigned buffers.
    const int batch_size = std::max(1, 500 / (mpi_size * effectiveDeviceCount));

    size_t maxMatrices = 0;
    for (int deviceId : scheduledDevices)
    {
      maxMatrices = std::max(maxMatrices, matricesToCopy[deviceId].size());
    }
    size_t globalMaxMatrices = 0;
    MPI_Allreduce(&maxMatrices, &globalMaxMatrices, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
    maxMatrices = globalMaxMatrices;

    for (size_t batchStart = 0; batchStart < maxMatrices; batchStart += batch_size)
    {
      std::vector<std::tuple<int, Matrix, cudaEvent_t>> dstMatPairs;
      std::atomic_int batchCounter(0);

      std::vector<std::vector<Matrix>> batchSlices(deviceCount);
      for (int deviceId : scheduledDevices)
      {
        auto& src = matricesToCopy[deviceId];
        size_t start = std::min(batchStart, src.size());
        size_t end = std::min(batchStart + batch_size, src.size());
        batchSlices[deviceId].assign(src.begin() + start, src.begin() + end);
      }

      if (deviceCount == 1)
      {
        auto commProvider = [&]() -> ncclComm_t {
          ensureNcclCommsInitialized();
          return allDeviceComms.empty() ? nullptr : allDeviceComms[0];
        };
        copyMatrices(0, deviceCount, batchSlices[0], dstMatPairs, batchCounter, commProvider, swapper,
                     matToSyncFinishEventMap);
      }
      else if (mpi_size == 1)
      {
        // Local multi-GPU transfers do not need the NCCL/MPI metadata
        // exchange, nor the spin-barriers used to build it.  The buffers for
        // all devices have already been allocated, so a sequential host-side
        // dispatch is enough; CUDA streams still carry the data dependencies.
        for (int deviceId : scheduledDevices)
        {
          copyLocalMatrices(deviceId, batchSlices[deviceId], swapper, matToSyncFinishEventMap);
        }
      }
      else
      {
        for (int deviceId = 0; deviceId < deviceCount; deviceId++)
        {
          auto commProvider = [&, deviceId]() -> ncclComm_t {
            ensureNcclCommsInitialized();
            return allDeviceComms.empty() ? nullptr : allDeviceComms[deviceId];
          };
          threads[deviceId] =
              std::thread(copyMatrices, deviceId, deviceCount, std::cref(batchSlices[deviceId]), std::ref(dstMatPairs),
                          std::ref(batchCounter), commProvider, std::ref(swapper), std::cref(matToSyncFinishEventMap));
        }

        for (int deviceId = 0; deviceId < deviceCount; deviceId++)
        {
          threads[deviceId].join();
        }
      }
    }

    for (int deviceId : scheduledDevices)
    {
      (void)deviceId;
      DEBUG_ARRANGER_BATCH_RANGE(deviceId, startIdxes[deviceId], endIdxes[deviceId], (int)worklists[deviceId].size());
    }

    for (int deviceId : scheduledDevices)
    {
      if (startIdxes[deviceId] == endIdxes[deviceId] &&
          endIdxes[deviceId] < static_cast<int>(worklists[deviceId].size()))
      {
        fprintf(stderr,
                "[ARRANGER][ERROR] Device %d: Unable to allocate enough "
                "memory.\n",
                deviceId);
        fprintf(stderr, "[ARRANGER][ERROR] Current position: %d/%zu in worklist\n", endIdxes[deviceId],
                worklists[deviceId].size());
        fprintf(stderr, "[ARRANGER][ERROR] Available memory: %.2f GB\n",
                freeMems[deviceId] / (1024.0 * 1024.0 * 1024.0));

        auto work = worklists[deviceId][startIdxes[deviceId]];
        auto matWork = std::dynamic_pointer_cast<MatWorkBase>(work);
        if (matWork)
        {
          size_t totalSize = 0;
          for (const auto& mat : matWork->getMatrices())
          {
            totalSize += mat.sizeInByte();
          }
          fprintf(stderr, "[ARRANGER][ERROR] Work at index %d requires: %.2f GB\n", startIdxes[deviceId],
                  totalSize / (1024.0 * 1024.0 * 1024.0));
        }

        exit(1);
      }
    }

    if (deviceCount == 1)
    {
      execute(worklists[0], 0, startIdxes[0], endIdxes[0]);
      startIdxes[0] = endIdxes[0];
    }
    else if (mpi_size == 1)
    {
      for (int deviceId : scheduledDevices)
      {
        execute(worklists[deviceId], deviceId, startIdxes[deviceId], endIdxes[deviceId]);
        startIdxes[deviceId] = endIdxes[deviceId];
      }
    }
    else
    {
      for (int deviceId = 0; deviceId < deviceCount; deviceId++)
      {
        threads[deviceId] =
            std::thread(execute, std::ref(worklists[deviceId]), deviceId, startIdxes[deviceId], endIdxes[deviceId]);
        startIdxes[deviceId] = endIdxes[deviceId];
      }
    }

    char areAllFinished = 1;
    if (deviceCount == 1)
    {
      if (endIdxes[0] < static_cast<int>(worklists[0].size()))
      {
        areAllFinished = 0;
      }
    }
    else if (mpi_size == 1)
    {
      for (int deviceId : scheduledDevices)
      {
        if (endIdxes[deviceId] < static_cast<int>(worklists[deviceId].size()))
        {
          areAllFinished = 0;
        }
      }
    }
    else
    {
      for (int deviceId = 0; deviceId < deviceCount; deviceId++)
      {
        threads[deviceId].join();
        if (endIdxes[deviceId] < static_cast<int>(worklists[deviceId].size()))
        {
          areAllFinished = 0;
        }
      }
    }

    std::vector<char> areRemoteFinished(mpi_size, 0);
    MPI_Allgather(&areAllFinished, 1, MPI_BYTE, areRemoteFinished.data(), 1, MPI_BYTE, MPI_COMM_WORLD);
    areAllFinished &= std::all_of(areRemoteFinished.begin(), areRemoteFinished.end(), [](char v) { return v == 1; });

    DEBUG_ARRANGER_ALL_FINISHED(areAllFinished);

    if (areAllFinished)
    {
      break;
    }
  }

  for (int deviceId : scheduledDevices)
  {
    swapper.deviceContext(deviceId).syncWorkStreams("arranger_execute_worklists");
  }

  if (!freeBuffersAtEnd)
  {
    CUDA_CALL(cudaSetDevice(0));
    return;
  }

  if (deviceCount == 1)
  {
    CUDA_CALL(cudaSetDevice(0));
    swapper.freeAllBuffer(0);
    swapper.freeAllEvents(0);
  }
  else if (localSingleScheduledDevice)
  {
    int const deviceId = scheduledDevices.front();
    CUDA_CALL(cudaSetDevice(deviceId));
    swapper.freeAllBuffer(deviceId);
    swapper.freeAllEvents(deviceId);
  }
  else
  {
    for (int deviceId = 0; deviceId < deviceCount; deviceId++)
    {
      threads[deviceId] = std::thread([&, deviceId] {
        CUDA_CALL(cudaSetDevice(deviceId));
        swapper.freeAllBuffer(deviceId);
        swapper.freeAllEvents(deviceId);
      });
    }

    for (int deviceId = 0; deviceId < deviceCount; deviceId++)
    {
      threads[deviceId].join();
    }
  }
}

void Arranger::doContraction(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                             const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats)
{
  // B*C intermediates are ordinary hostless buffers consumed by the second
  // phase.  Keep them resident until A*(B*C) has finished.
  executeWorklists(worklistsForInterMat, liveIntervalsForInterMat, false);
  executeWorklists(worklistsForTheRest, liveIntervalsForTheRest);
}

void Arranger::compileInnerProductForLinearAlgebra(Matrix m1, Matrix m2, double* result)
{
  auto [deviceId, _] = swapper.getPreStoreBufferOrNone(m1);
  if (deviceId == -1)
  {
    deviceId = this->leastBusyLinearAlgebraDevice();
  }
  linearAlgebraWorklists[deviceId].push_back(
      createWork<InnerProductWork>(std::vector<Matrix>{m1, m2}, 1.0, deviceId, swapper, result));
  linearAlgebraFlopsPerDevice[deviceId] += m1.size() * 2;
}

void Arranger::compileInnerProductAccumulateForLinearAlgebra(Matrix partial, Matrix m1, Matrix m2)
{
  auto [deviceId, _] = swapper.getPreStoreBufferOrNone(m1);
  if (deviceId == -1)
  {
    deviceId = this->leastBusyLinearAlgebraDevice();
  }
  auto [partialDeviceId, partialBuffer] = swapper.getPreStoreBufferOrNone(partial);
  if (partialBuffer == nullptr || partialDeviceId != deviceId)
  {
    throw std::logic_error("TensorContraction inner-product partial must be resident on the input block device");
  }
  linearAlgebraWorklists[deviceId].push_back(
      createWork<InnerProductAccumulateWork>(std::vector<Matrix>{partial, m1, m2}, 1.0, deviceId, swapper));
  linearAlgebraFlopsPerDevice[deviceId] += m1.size() * 2;
}

void Arranger::compileSyncForLinearAlgebra(Matrix result)
{
  auto [deviceId, buffer] = swapper.getPreStoreBufferOrNone(result);
  if (buffer == nullptr)
  {
    throw std::logic_error("TensorContraction cannot synchronize a matrix without a resident GPU buffer");
  }
  linearAlgebraWorklists[deviceId].push_back(createWork<SyncWork>(std::vector<Matrix>{result}, 0.0, deviceId, swapper));
}

void Arranger::compileZeroForLinearAlgebra(Matrix result, bool syncHost)
{
  auto [deviceId, _] = swapper.getPreStoreBufferOrNone(result);
  if (deviceId == -1)
  {
    deviceId = this->leastBusyLinearAlgebraDevice();
  }
  linearAlgebraWorklists[deviceId].push_back(
      createWork<MemsetWork>(std::vector<Matrix>{result}, 0.0, deviceId, swapper));
  if (syncHost)
  {
    linearAlgebraWorklists[deviceId].push_back(
        createWork<SyncWork>(std::vector<Matrix>{result}, 0.0, deviceId, swapper));
  }
}

void Arranger::compileMatMulForLinearAlgebra(Matrix result, Matrix m1, Matrix m2, bool syncHost)
{
  auto [deviceId, _] = swapper.getPreStoreBufferOrNone(result);
  if (deviceId == -1)
  {
    deviceId = this->leastBusyLinearAlgebraDevice();
  }
  linearAlgebraWorklists[deviceId].push_back(
      createWork<MatMulWork>(std::vector<Matrix>{result, m1, m2}, 1.0, deviceId, swapper));
  linearAlgebraFlopsPerDevice[deviceId] += m1.getFirstDim() * m1.getSecondDim() * m2.getSecondDim() * 2;
  if (syncHost)
  {
    linearAlgebraWorklists[deviceId].push_back(
        createWork<SyncWork>(std::vector<Matrix>{result}, 0.0, deviceId, swapper));
  }
}

void Arranger::compileAddAccuForLinearAlgebra(Matrix result, Matrix m1, double* coff, bool syncHost)
{
  auto [deviceId, _] = swapper.getPreStoreBufferOrNone(result);
  if (deviceId == -1)
  {
    deviceId = this->leastBusyLinearAlgebraDevice();
  }
  linearAlgebraWorklists[deviceId].push_back(
      createWork<AddAccuWork>(std::vector<Matrix>{result, m1}, *coff, deviceId, swapper));
  if (syncHost)
  {
    linearAlgebraWorklists[deviceId].push_back(
        createWork<SyncWork>(std::vector<Matrix>{result}, 0.0, deviceId, swapper));
  }
}

void Arranger::compileScalarMulForLinearAlgebra(Matrix result, double* coff, bool syncHost)
{
  auto [deviceId, _] = swapper.getPreStoreBufferOrNone(result);
  if (deviceId == -1)
  {
    deviceId = this->leastBusyLinearAlgebraDevice();
  }
  linearAlgebraWorklists[deviceId].push_back(
      createWork<ScalarMulWork>(std::vector<Matrix>{result}, 1.0, deviceId, swapper, coff));
  if (syncHost)
  {
    linearAlgebraWorklists[deviceId].push_back(
        createWork<SyncWork>(std::vector<Matrix>{result}, 0.0, deviceId, swapper));
  }
}

void Arranger::doLinearAlgebra()
{
  ensureMemoryPoolsInitialized();
  buildLiveInterval(linearAlgebraWorklists, liveIntervalsForLinearAlgebra);
  executeWorklists(linearAlgebraWorklists, liveIntervalsForLinearAlgebra);
  for (auto& wl : linearAlgebraWorklists)
    wl.clear();
  liveIntervalsForLinearAlgebra.clear();
  std::fill(linearAlgebraFlopsPerDevice.begin(), linearAlgebraFlopsPerDevice.end(), 0.0);
}

void Arranger::localizeForLinearAlgebra(const std::vector<Matrix>& mats, bool uploadFromHost, bool refreshExisting)
{
  ensureMemoryPoolsInitialized();

  // Localize MatrixFamily blocks into TensorContraction pre-store buffers.
  // With uploadFromHost=false this is allocation-only and preserves the current
  // GPU-resident value; with uploadFromHost=true host storage is the authority.
  std::vector<size_t> bytesPerDevice(deviceCount, 0);
  std::vector<bool> touchedDevice(deviceCount, false);
  for (auto mat : mats)
  {
    auto [deviceId, buffer] = swapper.getPreStoreBufferOrNone(mat);
    if (buffer != nullptr)
    {
      if (uploadFromHost && refreshExisting)
      {
        swapper.refreshHostMatrixToDevice(mat.hostView());
        touchedDevice[deviceId] = true;
      }
      continue;
    }

    deviceId = this->leastBusyLinearAlgebraDevice();
    CUDA_CALL(cudaSetDevice(deviceId));
    if (uploadFromHost)
    {
      swapper.uploadHostMatrix(mat.hostView(), deviceId);
    }
    else
    {
      swapper.registerGpuAllocation(mat, deviceId);
    }
    linearAlgebraFlopsPerDevice[deviceId] += static_cast<double>(mat.size());
    bytesPerDevice[deviceId] += mat.sizeInByte();
    touchedDevice[deviceId] = true;
  }

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
  {
    if (bytesPerDevice[deviceId] != 0 || touchedDevice[deviceId])
    {
      swapper.syncMemStream(deviceId, "linear_algebra_localize");
    }
  }
  std::fill(linearAlgebraFlopsPerDevice.begin(), linearAlgebraFlopsPerDevice.end(), 0.0);
  CUDA_CALL(cudaSetDevice(0));
}

void Arranger::localizeCoalescedForLinearAlgebra(const std::vector<Matrix>& mats, std::span<double const> values,
                                                 bool uploadFromHost, bool refreshExisting)
{
  ensureMemoryPoolsInitialized();

  std::size_t totalBytes = 0;
  for (auto mat : mats)
  {
    totalBytes += mat.sizeInByte();
  }

  auto finishLocalization = [&]() {
    std::fill(linearAlgebraFlopsPerDevice.begin(), linearAlgebraFlopsPerDevice.end(), 0.0);
    CUDA_CALL(cudaSetDevice(0));
  };

  if (!mats.empty() && totalBytes == values.size_bytes())
  {
    if (deviceCount == 1)
    {
      bool const anyExisting = swapper.anyPreStoreBuffer(mats);
      auto const existingDevice = swapper.commonPreStoreDevice(mats);
      if (!anyExisting)
      {
        int const deviceId = this->leastBusyLinearAlgebraDevice();
        if (uploadFromHost)
        {
          swapper.uploadHostMatricesCoalesced(mats, values, deviceId);
        }
        else
        {
          swapper.registerGpuAllocationsCoalesced(mats, deviceId);
        }
        finishLocalization();
        return;
      }

      if (existingDevice.has_value())
      {
        if (uploadFromHost && refreshExisting)
        {
          if (swapper.refreshHostMatricesToDeviceCoalesced(mats, values, *existingDevice))
          {
            finishLocalization();
            return;
          }
        }
        else
        {
          finishLocalization();
          return;
        }
      }
    }
    else
    {
      bool localized = true;
      for (auto const range : contiguousCoalescedPlacement(mats, deviceCount))
      {
        if (!localizeCoalescedRange(swapper, range, mats, values, uploadFromHost, refreshExisting))
        {
          localized = false;
          break;
        }
      }
      if (localized)
      {
        finishLocalization();
        return;
      }
    }
  }

  this->localizeForLinearAlgebra(mats, uploadFromHost, refreshExisting);
}

void Arranger::synchronizeLinearAlgebraToHost(const std::vector<Matrix>& mats)
{
  ensureMemoryPoolsInitialized();

  // Host synchronization is intentionally explicit for resident Lanczos mode:
  // most vector operations should not force D2H copies, but the current
  // operator adapter still needs host-visible MatrixFamily storage.
  std::vector<bool> touched(deviceCount, false);
  for (auto mat : mats)
  {
    auto [deviceId, buffer] = swapper.getPreStoreBufferOrNone(mat);
    if (buffer == nullptr)
    {
      continue;
    }
    swapper.downloadDeviceToHost(mat.hostView());
    touched[deviceId] = true;
  }

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
  {
    if (touched[deviceId])
    {
      swapper.syncMemStream(deviceId, "linear_algebra_sync_to_host");
    }
  }
  CUDA_CALL(cudaSetDevice(0));
}

void Arranger::synchronizeCoalescedLinearAlgebraToHost(const std::vector<Matrix>& mats, std::span<double> values)
{
  ensureMemoryPoolsInitialized();

  if (!mats.empty() && swapper.downloadDeviceMatricesToHostCoalesced(mats, values))
  {
    if (auto deviceId = swapper.commonPreStoreDevice(mats); deviceId.has_value())
    {
      swapper.syncMemStream(*deviceId, "linear_algebra_sync_to_host_coalesced");
    }
    CUDA_CALL(cudaSetDevice(0));
    return;
  }

  std::size_t totalBytes = 0;
  for (auto mat : mats)
  {
    totalBytes += mat.sizeInByte();
  }
  if (!mats.empty() && totalBytes == values.size_bytes())
  {
    std::vector<bool> touched(deviceCount, false);
    bool synchronized = true;
    for (auto const range : contiguousCoalescedPlacement(mats, deviceCount))
    {
      auto const rangeMats = matrixRange(mats, range);
      if (swapper.downloadDeviceMatricesToHostCoalesced(rangeMats, valueRange(values, range)))
      {
        if (auto deviceId = swapper.commonPreStoreDevice(rangeMats); deviceId.has_value())
        {
          touched[*deviceId] = true;
        }
      }
      else
      {
        synchronized = false;
        break;
      }
    }
    if (synchronized)
    {
      for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
      {
        if (touched[deviceId])
        {
          swapper.syncMemStream(deviceId, "linear_algebra_sync_to_host_coalesced");
        }
      }
      CUDA_CALL(cudaSetDevice(0));
      return;
    }
  }

  this->synchronizeLinearAlgebraToHost(mats);
}

double* Arranger::collectiveExchangeMatrix(Matrix m)
{
  double* ptr = nullptr;

  // Case 1: matrix is in local host memory.
  if (m.hasHostStorage())
  {
    ptr = m.hostView().data();
  }

  // Case 2: matrix is pre-stored in a local GPU. Copy it to a host buffer.
  if (ptr == nullptr)
  {
    auto [local_device_id, local_buffer] = swapper.getPreStoreBufferOrNone(m);
    if (local_buffer)
    {
      ptr = (double*)malloc(m.sizeInByte());
      CUDA_CALL(cudaSetDevice(local_device_id));
      CUDA_CALL(cudaMemcpy(ptr, local_buffer->getPtr(), m.sizeInByte(), cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaSetDevice(0));
    }
  }

  // Allgather: each rank broadcasts the matrix it still needs (Case 3).
  // Ranks that resolved ptr above broadcast an empty Header{} as a no-op.
  Matrix::Header myHeader = (ptr == nullptr) ? m.toHeader() : Matrix::Header{};
  std::vector<Matrix::Header> recvHeaders(mpi_size);
  MPI_Allgather(&myHeader, sizeof(Matrix::Header), MPI_BYTE, recvHeaders.data(), sizeof(Matrix::Header), MPI_BYTE,
                MPI_COMM_WORLD);

  std::vector<MPI_Request> requests(mpi_size, MPI_REQUEST_NULL);
  std::vector<double*> tmp_ptrs;
  for (int rank = 0; rank < mpi_size; rank++)
  {
    if (rank == mpi_rank && myHeader.id != -1)
    {
      // Case 3 (self): matrix is on a remote node. If it's on a remote GPU,
      // decode the MPI rank from the NCCL ID; otherwise use node_id directly.
      int src = swapper.isOnRemoteGpu(m.getId()) ? swapper.getRemoteNCCLId(m.getId()) / deviceCount : m.getNodeId();
      ptr = (double*)malloc(m.sizeInByte());
      MPI_Irecv(ptr, m.size(), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &requests[rank]);
      continue;
    }

    // Case 3 (peer): serve another rank's request if we hold that matrix.
    Matrix requested(recvHeaders[rank]);
    if (requested.getId() == -1) continue;

    auto [dev_id, buf] = swapper.getPreStoreBufferOrNone(requested);
    if (buf)
    {
      // Serve from local GPU pre-store buffer.
      double* tmp = (double*)malloc(requested.sizeInByte());
      tmp_ptrs.push_back(tmp);
      CUDA_CALL(cudaSetDevice(dev_id));
      CUDA_CALL(cudaMemcpy(tmp, buf->getPtr(), requested.sizeInByte(), cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaSetDevice(0));
      MPI_Isend(tmp, requested.size(), MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &requests[rank]);
    }
    else if (requested.getNodeId() == mpi_rank)
    {
      // Serve from local CPU memory.
      auto it = localMats.find(requested.getId());
      if (it != localMats.end() && it->second.hasHostStorage())
      {
        MPI_Isend(it->second.hostView().requireData(), requested.size(), MPI_DOUBLE, rank, 0, MPI_COMM_WORLD,
                  &requests[rank]);
      }
    }
  }

  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  for (auto* tmp : tmp_ptrs)
    free(tmp);

  return ptr;
}

void Arranger::resetWork()
{
  interMats.clear();
  interMatsIdx.clear();
  combineMats.clear();
  shouldReuseInter.clear();
  shouldCombineInter.clear();
  shouldFinalizeInter.clear();
  sortedFTerms.clear();
  matToSyncFinishEventMap.clear();
  localMats.clear();
  rFlops.clear();
  for (auto& wl : worklistsForInterMat)
    wl.clear();
  for (auto& wl : worklistsForTheRest)
    wl.clear();
  for (auto& wl : linearAlgebraWorklists)
    wl.clear();
  liveIntervalsForInterMat.clear();
  liveIntervalsForTheRest.clear();
  liveIntervalsForLinearAlgebra.clear();
  std::fill(linearAlgebraFlopsPerDevice.begin(), linearAlgebraFlopsPerDevice.end(), 0.0);
}

void Arranger::releaseResources()
{
  resetWork();
  for (auto event : syncFinishEvents)
  {
    CUDA_CALL(cudaEventDestroy(event));
  }
  syncFinishEvents.clear();
  for (auto comm : allDeviceComms)
  {
    NCCL_CALL(ncclCommDestroy(comm));
  }
  allDeviceComms.clear();
  ncclCommsInitialized = false;
}

void Arranger::distributeMatricesToNodes(std::vector<Matrix>& mats, std::string prefix)
{
  int mat_size = mats.size();
  MPI_Bcast(&mat_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (mpi_rank != 0)
  {
    mats.resize(mat_size);
  }

  if (mpi_rank == 0)
  {
    // Greedy size-balanced distribution
    std::vector<int> order(mat_size);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return mats[a].sizeInByte() > mats[b].sizeInByte(); });

    std::vector<size_t> totalPerRank(mpi_size, 0);
    for (int idx : order)
    {
      int minRank = 0;
      for (int r = 1; r < mpi_size; r++)
      {
        if (totalPerRank[r] < totalPerRank[minRank])
        {
          minRank = r;
        }
      }
      mats[idx].setNodeId(minRank);
      totalPerRank[minRank] += mats[idx].sizeInByte();
    }
  }

  for (int i = 0; i < mat_size; i++)
  {
    Matrix::Header h = mats[i].toHeader();
    MPI_Bcast(&h, sizeof(Matrix::Header), MPI_BYTE, 0, MPI_COMM_WORLD);
    mats[i] = Matrix(h);
  }

  for (int i = 0; i < (int)mats.size(); i++)
  {
    if (mats[i].getNodeId() == mpi_rank)
    {
      localMats[mats[i].getId()] = mats[i];
#if DEBUG_LOG
      swapper.setMatrixName(mats[i], prefix + std::to_string(i));
      DEBUG_MATRIX_DISTRIBUTION(swapper, mpi_rank, mats[i].getId());
#endif
    }
  }
}

void Arranger::distributeFTermsToNodes(std::vector<TermTy>& terms, const std::vector<Matrix>& rMats)
{
  // distribute terms to each node as evenly as possible so that
  // each node calculates a complete R.

  // Map each R index to a rank using actual nodeIds from R matrices.
  std::vector<int> rIdxToRank(rMats.size());
  for (size_t i = 0; i < rMats.size(); i++)
  {
    rIdxToRank[i] = rMats[i].getNodeId();
  }

  if (mpi_rank == 0)
  {
    // Group terms by destination rank
    std::vector<std::vector<TermTy>> termsPerRank(mpi_size);
    for (const auto& term : terms)
    {
      int destRank = rIdxToRank[std::get<0>(term)];
      termsPerRank[destRank].push_back(term);
    }

    // Send terms to each non-zero rank
    for (int rank = 1; rank < mpi_size; rank++)
    {
      int count = termsPerRank[rank].size();
      MPI_Send(&count, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
      if (count > 0)
      {
        MPI_Send(termsPerRank[rank].data(), count * sizeof(TermTy), MPI_BYTE, rank, 1, MPI_COMM_WORLD);
      }
    }

    // Keep rank 0's terms
    terms = std::move(termsPerRank[0]);
  }
  else
  {
    // Receive terms from rank 0
    int count;
    MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    terms.resize(count);
    if (count > 0)
    {
      MPI_Recv(terms.data(), count * sizeof(TermTy), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (auto [r, a, b, c, f] : terms)
  {
    DEBUG_TERM_DISTRIBUTION(mpi_rank, r, a, b, c, f);
    (void)r;
    (void)a;
    (void)b;
    (void)c;
    (void)f;
  }
}

void Arranger::broadcastRtoB(const std::vector<Matrix>& rMats, std::vector<Matrix>& bMats)
{
  assert(rMats.size() == bMats.size());

  std::vector<MPI_Request> requests;
  // (index, temp CPU buf) for GPU-resident B matrices receiving via MPI
  std::vector<std::pair<int, double*>> gpuRecvPending;

  // Phase 1: R on this node, B on another node — post Isend
  for (int i = 0; i < (int)rMats.size(); i++)
  {
    if (rMats[i].getNodeId() != mpi_rank) continue;
    if (bMats[i].getNodeId() == mpi_rank) continue;
    MPI_Request req;
    MPI_Isend(rMats[i].hostView().requireData(), rMats[i].size(), MPI_DOUBLE, bMats[i].getNodeId(), i, MPI_COMM_WORLD,
              &req);
    requests.push_back(req);
  }

  // Phase 2: B on this node, R on another node — post Irecv
  for (int i = 0; i < (int)rMats.size(); i++)
  {
    if (bMats[i].getNodeId() != mpi_rank) continue;
    if (rMats[i].getNodeId() == mpi_rank) continue;
    MPI_Request req;
    if (bMats[i].hasHostStorage())
    {
      MPI_Irecv(bMats[i].hostView().requireData(), bMats[i].size(), MPI_DOUBLE, rMats[i].getNodeId(), i, MPI_COMM_WORLD,
                &req);
      requests.push_back(req);
    }
    else
    {
      // B is GPU-only: receive into a temporary pinned CPU buffer
      double* tmp;
      cudaMallocHost(&tmp, bMats[i].sizeInByte());
      MPI_Irecv(tmp, bMats[i].size(), MPI_DOUBLE, rMats[i].getNodeId(), i, MPI_COMM_WORLD, &req);
      requests.push_back(req);
      gpuRecvPending.push_back({i, tmp});
    }
  }

  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

  // Phase 3: same node — direct copy, no MPI
  for (int i = 0; i < (int)rMats.size(); i++)
  {
    if (rMats[i].getNodeId() != mpi_rank) continue;
    if (bMats[i].getNodeId() != mpi_rank) continue;
    if (bMats[i].hasHostStorage())
    {
      memcpy(bMats[i].hostView().requireData(), rMats[i].hostView().requireData(), bMats[i].sizeInByte());
    }
    else
    {
      auto [deviceId, buf] = swapper.getPreStoreBufferOrNone(bMats[i]);
      cudaSetDevice(deviceId);
      cudaMemcpy(buf->getPtr(), rMats[i].hostView().requireData(), bMats[i].sizeInByte(), cudaMemcpyHostToDevice);
    }
  }

  // Copy received data into GPU-resident B matrices
  for (auto [i, tmp] : gpuRecvPending)
  {
    auto [deviceId, buf] = swapper.getPreStoreBufferOrNone(bMats[i]);
    cudaSetDevice(deviceId);
    cudaMemcpy(buf->getPtr(), tmp, bMats[i].sizeInByte(), cudaMemcpyHostToDevice);
    cudaFreeHost(tmp);
  }

  // Phase 4: propagate updated B to non-owner ranks that have local CPU copies
  // (created by preCopyMatrices; now stale after R→B update)
  if (mpi_size <= 1) return;

  std::vector<std::vector<int>> tokensNeededFromRank(mpi_size);
  std::unordered_map<int, Matrix> neededMatMap;
  for (auto& bm : bMats)
  {
    if (bm.getNodeId() != mpi_rank && bm.hasHostStorage())
    {
      tokensNeededFromRank[bm.getNodeId()].push_back(bm.getId());
      neededMatMap[bm.getId()] = bm;
    }
  }

  mpiExchangeCopies(tokensNeededFromRank, neededMatMap);
}

} // namespace tensor
