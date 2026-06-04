#pragma once

#include <nccl.h>

#include <cstddef>
#include <set>
#include <unordered_set>
#include <vector>

#include "Calculator.hpp"
#include "Matrix.hpp"
#include "MatrixAllocator.hpp"
#include "Utils.h"

namespace tensor
{

class Swapper;

class Arranger {
    Swapper& swapper;
    int deviceCount;
    ncclUniqueId ncclAllDeviceId;
    std::vector<ncclComm_t> allDeviceComms;
    int mpi_rank;
    int mpi_size;

    std::vector<Matrix> interMats;
    std::vector<std::pair<int, int>> interMatsIdx;

    std::vector<std::tuple<int, int, Matrix>> combineMats;

    std::vector<bool> shouldReuseInter;
    std::vector<bool> shouldCombineInter;
    std::vector<bool> shouldFinalizeInter;
    std::vector<TermTy> sortedFTerms;
    std::unordered_map<int, cudaEvent_t> matToSyncFinishEventMap;
    std::unordered_map<int, Matrix> localMats;
    std::vector<double> rFlops;

  public:
    using WorklistTy = std::vector<std::shared_ptr<WorkBase>>;
    using LiveIntervalMap = std::vector<std::tuple<int, int, Matrix>>;

  private:
    std::vector<WorklistTy> worklistsForInterMat;
    std::vector<WorklistTy> worklistsForTheRest;
    std::vector<WorklistTy> linearAlgebraWorklists;
    std::vector<LiveIntervalMap> liveIntervalsForInterMat;
    std::vector<LiveIntervalMap> liveIntervalsForTheRest;
    std::vector<LiveIntervalMap> liveIntervalsForLinearAlgebra;
    std::vector<double> linearAlgebraFlopsPerDevice;
    std::vector<cudaEvent_t> syncFinishEvents;
    int contractionScheduledDeviceCount = 1;
    int linearAlgebraScheduledDeviceCount = 1;
    bool memoryPoolsInitialized = false;
    bool ncclCommsInitialized = false;

    void preprocess(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                    const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats,
                    const std::vector<TermTy>& fTerms, std::vector<bool>& shouldReuseInter,
                    std::vector<bool>& shouldCombineInter, std::vector<bool>& shouldFinalizeInter, bool shouldAlloc);

    void calculateRFlops(const std::vector<TermTy>& fTerms, const std::vector<Matrix>& rMats,
                         const std::vector<Matrix>& aMats, const std::vector<Matrix>& bMats,
                         const std::vector<Matrix>& cMats, const std::vector<bool>& shouldReuseInter,
                         const std::vector<bool>& shouldCombineInter, const std::vector<bool>& shouldFinalizeInter);

    void enableP2PPeerAccess();

    std::shared_ptr<std::atomic_int> createSemaphore();
    cudaEvent_t createSyncFinishEvent();

    void buildLiveInterval(std::vector<WorklistTy>& worklists, std::vector<LiveIntervalMap>& liveIntervals);

    void executeWorklists(std::vector<WorklistTy>& worklists, std::vector<LiveIntervalMap>& liveIntervals,
                          bool freeBuffersAtEnd = true);
    int scheduledDeviceCountForFlops(double flops) const;
    int scheduledDeviceCountForBytes(std::size_t bytes) const;
    int leastBusyContractionDevice(const std::vector<double>& flopsPerDevice) const;
    int leastBusyLinearAlgebraDevice() const;
    void ensureNcclCommsInitialized();

    void mpiExchangeCopies(const std::vector<std::vector<int>>& tokensNeededFromRank,
                           const std::unordered_map<int, Matrix>& neededMatMap);

  public:
    Arranger(Swapper& swapper);
    ~Arranger();
    Swapper& residentSwapper() { return swapper; }
    void ensureMemoryPoolsInitialized();
    std::vector<Matrix>& getInterMats();
    void doContraction(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                       const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats);

    void compileInnerProductForLinearAlgebra(Matrix m1, Matrix m2, double* result);
    void compileZeroForLinearAlgebra(Matrix result, bool syncHost = true);
    void compileMatMulForLinearAlgebra(Matrix result, Matrix m1, Matrix m2, bool syncHost = true);
    void compileAddAccuForLinearAlgebra(Matrix result, Matrix m1, double* coff, bool syncHost = true);
    void compileScalarMulForLinearAlgebra(Matrix result, double* coff, bool syncHost = true);
    void doLinearAlgebra();
    void localizeForLinearAlgebra(const std::vector<Matrix>& mats, bool uploadFromHost, bool refreshExisting = true);
    void synchronizeLinearAlgebraToHost(const std::vector<Matrix>& mats);
    double* collectiveExchangeMatrix(Matrix m = Matrix());

    void analyzeComputation(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                            const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats,
                            const std::vector<TermTy>& fTerms);
    void preStoreToDevice(const std::vector<Matrix>& aMats, const std::vector<Matrix>& bMats,
                          const std::vector<Matrix>& cMats);
    void compileWorklistsForInterMat(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                                     const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats,
                                     std::vector<double>& flopsPerDevice);

    void compileWorklistsForTheRest(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                                    const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats,
                                    std::vector<double>& flopsPerDevice, bool syncResultsToHost);
    void compileForSingleR(int fTermsStart, int fTermsEnd, const Matrix rMats, const std::vector<Matrix>& aMats,
                           const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats,
                           std::vector<double>& flopsPerDevice, bool syncResultToHost);
    void compileWorklists(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                          const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats,
                          bool syncResultsToHost = true);
    void distributeMatricesToNodes(std::vector<Matrix>& mats, std::string prefix);
    void distributeFTermsToNodes(std::vector<TermTy>& terms, const std::vector<Matrix>& rMats);
    void preCopyMatrices(std::vector<Matrix>& aMats, std::vector<Matrix>& bMats, std::vector<Matrix>& cMats,
                         MatrixAllocator& allocator);
    void broadcastRtoB(const std::vector<Matrix>& rMats, std::vector<Matrix>& bMats);
#if DEBUG_LOG
    void initializeDebugInfo(const std::vector<Matrix>& rMats, const std::vector<Matrix>& aMats,
                             const std::vector<Matrix>& bMats, const std::vector<Matrix>& cMats);
#endif
    void resetWork();
    void releaseResources();
};

} // namespace tensor
