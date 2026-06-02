#include <mpi.h>

#include <random>

#include "MatrixAllocator.hpp"
#include "Swapper.hpp"
#include "Utils.h"

namespace tensor
{

std::vector<Matrix> MatrixAllocator::initializeMatricesFromChunk(const std::vector<std::pair<int, int>>& dims,
                                                                 double* chunkPtr, bool shouldInitialize,
                                                                 bool initWithZero, int scale1, int scale2)
{
  std::default_random_engine gen(0x004E006E);
  std::normal_distribution<double> dist;

  std::vector<Matrix> results;
  double* currentPtr = chunkPtr;

  for (auto [dim1, dim2] : dims)
  {
    dim1 *= scale1;
    dim2 *= scale2;
    size_t matSize = (size_t)dim1 * dim2;

    if (shouldInitialize)
    {
      if (initWithZero)
      {
        memset(currentPtr, 0, matSize * sizeof(double));
      }
      else
      {
        for (size_t i = 0; i < matSize; i++)
        {
          currentPtr[i] = dist(gen);
        }
      }
    }
    results.emplace_back(currentPtr, dim1, dim2);
    currentPtr += matSize;
  }

  return results;
}

std::vector<Matrix> MatrixAllocator::allocateMatrices(const std::vector<std::pair<int, int>>& dims,
                                                      bool shouldInitialize, bool initWithZero, int scale1, int scale2)
{
  // Calculate total size needed
  size_t totalSize = 0;
  for (auto [dim1, dim2] : dims)
  {
    dim1 *= scale1;
    dim2 *= scale2;
    totalSize += (size_t)dim1 * dim2;
  }

  // Allocate chunk
  double* chunkPtr;
  CUDA_CALL(cudaMallocHost((void**)&chunkPtr, totalSize * sizeof(double)));

  // Track the chunk
  allocatedChunks.push_back(chunkPtr);

  // Initialize and return matrices
  return initializeMatricesFromChunk(dims, chunkPtr, shouldInitialize, initWithZero, scale1, scale2);
}

void MatrixAllocator::allocateMatrices(std::vector<Matrix>& mats, bool shouldInitialize, bool initWithZero)
{
  // Calculate total size needed
  size_t totalSize = 0;
  for (const auto& mat : mats)
  {
    totalSize += mat.size();
  }

  // Allocate chunk
  double* chunkPtr;
  CUDA_CALL(cudaMallocHost((void**)&chunkPtr, totalSize * sizeof(double)));

  // Track the chunk
  allocatedChunks.push_back(chunkPtr);

  // Update Matrix pointers
  std::default_random_engine gen(0x004E006E);
  std::normal_distribution<double> dist;

  double* currentPtr = chunkPtr;
  for (auto& mat : mats)
  {
    size_t matSize = mat.size();

    if (shouldInitialize)
    {
      if (initWithZero)
      {
        memset(currentPtr, 0, matSize * sizeof(double));
      }
      else
      {
        for (size_t i = 0; i < matSize; i++)
        {
          currentPtr[i] = dist(gen);
        }
      }
    }

    // Update the Matrix object's pointer in place
    mat.setPtr(currentPtr);
    currentPtr += matSize;
  }
}

void MatrixAllocator::freeAll(const std::vector<Matrix>& aMats, const std::vector<Matrix>& bMats,
                              const std::vector<Matrix>& cMats)
{
  // Free GPU device memory from A, B, and C matrices
  std::vector<const std::vector<Matrix>*> allMats = {&aMats, &bMats, &cMats};

  // Free host memory chunks
  for (auto chunk : allocatedChunks)
  {
    CUDA_CALL(cudaFreeHost(chunk));
  }
  allocatedChunks.clear();
}

void MatrixAllocator::allocateMatrices(std::vector<Matrix>& r_mats, std::vector<Matrix>& a_mats,
                                       std::vector<Matrix>& b_mats, std::vector<Matrix>& c_mats,
                                       std::vector<Matrix>& inter_mats, Swapper& swapper)
{
  // Query 50% of available memory pool for each device
  int device_count = swapper.getDeviceCount();
  std::vector<size_t> availableMemory(device_count);
  std::vector<size_t> initialAvailableMemory(device_count);

  for (int i = 0; i < device_count; i++)
  {
    CUDA_CALL(cudaSetDevice(i));

    cudaMemPool_t pool = swapper.getMemPool(i);

    // Get pool's maxSize (0 for default pool, set value for custom pool)
    uint64_t maxSize = 0;
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &maxSize);

    uint64_t usedMem = 0;
    CUDA_CALL(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &usedMem));

    if (maxSize > 0)
    {
      // Custom pool with maxSize limit
      availableMemory[i] = (maxSize > usedMem) ? (maxSize - usedMem) / 2 : 0;
    }
    else
    {
      // Default-pool allocation uses this only as a planning budget. The
      // Swapper allocation path still observes exact CUDA allocation failures.
      availableMemory[i] = swapper.deviceContext(i).freeMemorySnapshot() / 2;
    }

    initialAvailableMemory[i] = availableMemory[i];
  }

  // Track matrices that need CPU allocation
  std::vector<Matrix*> cpuMatrices;

  int currentDeviceIdx = 0;
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  auto tryAllocateOnGpu = [&](Matrix& mat) {
    if (mat.getNodeId() != mpi_rank)
    {
      return;
    }
    size_t matSize = mat.sizeInByte();

    // Try to find a device with enough available memory
    for (int attempt = 0; attempt < device_count; attempt++)
    {
      int deviceId = (currentDeviceIdx + attempt) % device_count;
      if (matSize <= availableMemory[deviceId])
      {
        swapper.registerGpuAllocation(mat, deviceId);
        availableMemory[deviceId] -= matSize;
        currentDeviceIdx = (deviceId + 1) % device_count;
        return;
      }
    }

    // No GPU has space, add to CPU list
    cpuMatrices.push_back(&mat);
  };

  // Allocate A matrices
  for (auto& mat : a_mats)
  {
    tryAllocateOnGpu(mat);
  }

  // Allocate B matrices
  for (auto& mat : b_mats)
  {
    tryAllocateOnGpu(mat);
  }

  // Allocate C matrices
  for (auto& mat : c_mats)
  {
    tryAllocateOnGpu(mat);
  }

  // Summarize the size of pre-storing matrices on each device, as well as the
  // size of the matrices on CPU memory.
  fprintf(stderr, "[ALLOCATION][SUMMARY] Node=%d Matrix allocation summary:\n", mpi_rank);

  // Output total size of matrices
  size_t totalASize = 0, totalBSize = 0, totalCSize = 0, totalInterSize = 0;
  for (const auto& mat : a_mats)
  {
    if (mat.getNodeId() == mpi_rank)
    {
      totalASize += mat.sizeInByte();
    }
  }
  for (const auto& mat : b_mats)
  {
    if (mat.getNodeId() == mpi_rank)
    {
      totalBSize += mat.sizeInByte();
    }
  }
  for (const auto& mat : c_mats)
  {
    if (mat.getNodeId() == mpi_rank)
    {
      totalCSize += mat.sizeInByte();
    }
  }
  for (const auto& mat : inter_mats)
  {
    totalInterSize += mat.sizeInByte();
  }
  size_t totalRSize = 0;
  for (const auto& mat : r_mats)
  {
    totalRSize += mat.sizeInByte();
  }

  fprintf(stderr, "[ALLOCATION][SUMMARY] Node=%d  Total A: %.2f GB\n", mpi_rank,
          totalASize / (1024.0 * 1024.0 * 1024.0));
  fprintf(stderr, "[ALLOCATION][SUMMARY] Node=%d  Total B: %.2f GB\n", mpi_rank,
          totalBSize / (1024.0 * 1024.0 * 1024.0));
  fprintf(stderr, "[ALLOCATION][SUMMARY] Node=%d  Total C: %.2f GB\n", mpi_rank,
          totalCSize / (1024.0 * 1024.0 * 1024.0));
  fprintf(stderr, "[ALLOCATION][SUMMARY] Node=%d  Total InterMat: %.2f GB\n", mpi_rank,
          totalInterSize / (1024.0 * 1024.0 * 1024.0));
  fprintf(stderr, "[ALLOCATION][SUMMARY] Node=%d  Total R: %.2f GB\n", mpi_rank,
          totalRSize / (1024.0 * 1024.0 * 1024.0));
  fprintf(stderr, "[ALLOCATION][SUMMARY] Node=%d  Grand Total: %.2f GB\n", mpi_rank,
          (totalASize + totalBSize + totalCSize + totalInterSize + totalRSize) / (1024.0 * 1024.0 * 1024.0));

  // GPU allocations per device
  for (int i = 0; i < device_count; i++)
  {
    size_t allocatedSize = initialAvailableMemory[i] - availableMemory[i];
    fprintf(stderr, "[ALLOCATION][SUMMARY] Node=%d  Device %d: %.2f GB\n", mpi_rank, i,
            allocatedSize / (1024.0 * 1024.0 * 1024.0));
  }

  // Chunk-allocate all CPU matrices together
  if (!cpuMatrices.empty())
  {
    std::vector<Matrix> cpuMatCopies;
    for (Matrix* matPtr : cpuMatrices)
    {
      cpuMatCopies.push_back(*matPtr);
    }
    allocateMatrices(cpuMatCopies);

    // Update original Matrix objects
    for (size_t i = 0; i < cpuMatrices.size(); i++)
    {
      *cpuMatrices[i] = cpuMatCopies[i];
    }
  }

  // Allocate R & intermediate matrices on CPU
  allocateMatrices(inter_mats);
  allocateMatrices(r_mats);
  // Sync all memory streams
  for (int i = 0; i < device_count; i++)
  {
    swapper.syncMemStream(i, "matrix_allocator_sync");
  }

  // CPU allocations
  size_t cpuSize = 0;
  for (Matrix* matPtr : cpuMatrices)
  {
    cpuSize += matPtr->sizeInByte();
  }
  for (const auto& mat : inter_mats)
  {
    cpuSize += mat.sizeInByte();
  }
  for (const auto& mat : r_mats)
  {
    cpuSize += mat.sizeInByte();
  }
  fprintf(stderr, "[ALLOCATION][SUMMARY] Node=%d  CPU: %.2f GB\n", mpi_rank, cpuSize / (1024.0 * 1024.0 * 1024.0));

  CUDA_CALL(cudaSetDevice(0));
}
} // namespace tensor
