#pragma once

#include <random>
#include <vector>

#include "Matrix.hpp"

namespace tensor
{

class Swapper;
/**
 * Utility for efficiently allocating multiple matrices using chunked
 * allocation. Instead of calling cudaMallocHost for each matrix individually,
 * this allocator pre-allocates one large contiguous chunk and partitions it
 * among matrices. This significantly reduces initialization overhead.
 *
 * The allocator stores all allocated chunks and provides a single freeAll()
 * method to free all chunks at once. The first matrix of each allocation
 * holds the chunk pointer for cleanup purposes.
 */
class MatrixAllocator {
  private:
    // Storage for all allocated chunks
    std::vector<double*> allocatedChunks;

    /**
     * Initializes matrices from a pre-allocated chunk.
     * Splits the chunk and initializes each matrix in-place.
     */
    std::vector<Matrix> initializeMatricesFromChunk(const std::vector<std::pair<int, int>>& dims, double* chunkPtr,
                                                    bool shouldInitialize = false, bool initWithZero = false,
                                                    int scale1 = 1, int scale2 = 1);

  public:
    /**
     * Allocate and initialize matrices with chunked allocation.
     * The first matrix's pointer points to the beginning of the chunk.
     * All chunks are automatically tracked and can be freed with freeAll().
     *
     * @param dims Vector of (rows, cols) pairs for each matrix
     * @param shouldInitialize If true, initialize values; if false, leave
     * uninitialized
     * @param initWithZero If true, initialize to zeros; else random values
     * @param scale1 Scale factor for first dimension
     * @param scale2 Scale factor for second dimension
     * @return Vector of Matrix objects with first matrix holding chunk pointer
     */
    std::vector<Matrix> allocateMatrices(const std::vector<std::pair<int, int>>& dims, bool shouldInitialize = false,
                                         bool initWithZero = false, int scale1 = 1, int scale2 = 1);

    /**
     * Allocate and update Matrix descriptors in place with chunked allocation.
     * Updates the pointers of the input Matrix objects to point to allocated
     * memory.
     *
     * @param mats Vector of Matrix descriptors (with nullptr pointers)
     * @param shouldInitialize If true, initialize values; if false, leave
     * uninitialized
     * @param initWithZero If true, initialize to zeros; else random values
     */
    void allocateMatrices(std::vector<Matrix>& mats, bool shouldInitialize = false, bool initWithZero = false);

    void allocateMatrices(std::vector<Matrix>& r_mats, std::vector<Matrix>& a_mats, std::vector<Matrix>& b_mats,
                          std::vector<Matrix>& c_mats, std::vector<Matrix>& inter_mats, Swapper& swapper);

    /**
     * Free all allocated chunks and GPU device memory.
     * This must be called before destroying the allocator to clean up all
     * memory. After calling this, all matrices become invalid.
     *
     * @param aMats Vector of A matrices to free from GPU
     * @param bMats Vector of B matrices to free from GPU
     * @param cMats Vector of C matrices to free from GPU
     */
    void freeAll(const std::vector<Matrix>& aMats, const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats);
};

} // namespace tensor
