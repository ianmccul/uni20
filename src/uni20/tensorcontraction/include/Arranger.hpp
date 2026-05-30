#pragma once

#include <nccl.h>

#include <set>
#include <unordered_set>
#include <vector>

#include "Calculator.hpp"
#include "Matrix.hpp"
#include "MatrixAllocator.hpp"
#include "Utils.h"

namespace tensor {

class Swapper;

class Arranger {
  std::vector<StreamManager> streamManagers;
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

  void preprocess(const std::vector<Matrix>& rMats,
                  const std::vector<Matrix>& aMats,
                  const std::vector<Matrix>& bMats,
                  const std::vector<Matrix>& cMats,
                  const std::vector<TermTy>& fTerms,
                  std::vector<bool>& shouldReuseInter,
                  std::vector<bool>& shouldCombineInter,
                  std::vector<bool>& shouldFinalizeInter, bool shouldAlloc);

  void calculateRFlops(const std::vector<TermTy>& fTerms,
                       const std::vector<Matrix>& rMats,
                       const std::vector<Matrix>& aMats,
                       const std::vector<Matrix>& bMats,
                       const std::vector<Matrix>& cMats,
                       const std::vector<bool>& shouldReuseInter,
                       const std::vector<bool>& shouldCombineInter,
                       const std::vector<bool>& shouldFinalizeInter);

  void enableP2PPeerAccess();

  std::shared_ptr<std::atomic_int> createSemaphore();
  cudaEvent_t createSyncFinishEvent();

  void buildLiveInterval(std::vector<WorklistTy>& worklists,
                         std::vector<LiveIntervalMap>& liveIntervals);

  void executeWorklists(std::vector<WorklistTy>& worklists,
                        std::vector<LiveIntervalMap>& liveIntervals);

  void mpiExchangeCopies(
      const std::vector<std::vector<int>>& tokensNeededFromRank,
      const std::unordered_map<int, Matrix>& neededMatMap);

 public:
  Arranger(Swapper& swapper);
  std::vector<Matrix>& getInterMats();
  void doContraction(const std::vector<Matrix>& rMats,
                     const std::vector<Matrix>& aMats,
                     const std::vector<Matrix>& bMats,
                     const std::vector<Matrix>& cMats);

  void compileInnerProductForLinearAlgebra(Matrix m1, Matrix m2,
                                           double* result);
  void compileAddAccuForLinearAlgebra(Matrix result, Matrix m1, double* coff);
  void compileScalarMulForLinearAlgebra(Matrix result, double* coff);
  void doLinearAlgebra();
  double* collectiveExchangeMatrix(Matrix m = Matrix());

  void analyzeComputation(const std::vector<Matrix>& rMats,
                          const std::vector<Matrix>& aMats,
                          const std::vector<Matrix>& bMats,
                          const std::vector<Matrix>& cMats,
                          const std::vector<TermTy>& fTerms);
  void preStoreToDevice(const std::vector<Matrix>& aMats,
                        const std::vector<Matrix>& bMats,
                        const std::vector<Matrix>& cMats);
  void compileWorklistsForInterMat(const std::vector<Matrix>& rMats,
                                   const std::vector<Matrix>& aMats,
                                   const std::vector<Matrix>& bMats,
                                   const std::vector<Matrix>& cMats,
                                   std::vector<double>& flopsPerDevice);

  void compileWorklistsForTheRest(const std::vector<Matrix>& rMats,
                                  const std::vector<Matrix>& aMats,
                                  const std::vector<Matrix>& bMats,
                                  const std::vector<Matrix>& cMats,
                                  std::vector<double>& flopsPerDevice);
  void compileForSingleR(int fTermsStart, int fTermsEnd, const Matrix rMats,
                         const std::vector<Matrix>& aMats,
                         const std::vector<Matrix>& bMats,
                         const std::vector<Matrix>& cMats,
                         std::vector<double>& flopsPerDevice);
  void compileWorklists(const std::vector<Matrix>& rMats,
                        const std::vector<Matrix>& aMats,
                        const std::vector<Matrix>& bMats,
                        const std::vector<Matrix>& cMats);
  void distributeMatricesToNodes(std::vector<Matrix>& mats, std::string prefix);
  void distributeFTermsToNodes(std::vector<TermTy>& terms,
                               const std::vector<Matrix>& rMats);
  void preCopyMatrices(std::vector<Matrix>& aMats, std::vector<Matrix>& bMats,
                       std::vector<Matrix>& cMats, MatrixAllocator& allocator);
  void broadcastRtoB(const std::vector<Matrix>& rMats,
                     std::vector<Matrix>& bMats);
#if DEBUG_LOG
  void initializeDebugInfo(const std::vector<Matrix>& rMats,
                           const std::vector<Matrix>& aMats,
                           const std::vector<Matrix>& bMats,
                           const std::vector<Matrix>& cMats);
#endif
  void clear();
};

}  // namespace tensor