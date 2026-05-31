#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <string>

// Set to 1 to enable debug output, 0 to disable
#ifndef DEBUG_LOG
#define DEBUG_LOG 0
#endif

// Forward declare to avoid circular dependency
namespace tensor
{
class Swapper;
}

#if DEBUG_LOG
#define DEBUG_PRINT(fmt, ...) fprintf(stderr, "[DEBUG %s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define DEBUG_GPU_COPY_H2D(swapper, matId, deviceId, stream, size)                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr, "[SWAPPER][H2D_COPY] Matrix=%s Device=%d Stream=%p Size=%zu bytes\n", name.c_str(), deviceId,      \
            (void*)stream, size);                                                                                      \
  }                                                                                                                    \
  while (0)

#define DEBUG_GPU_COPY_D2H(swapper, matId, deviceId, stream, size)                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr, "[SWAPPER][D2H_COPY] Matrix=%s Device=%d Stream=%p Size=%zu bytes\n", name.c_str(), deviceId,      \
            (void*)stream, size);                                                                                      \
  }                                                                                                                    \
  while (0)

#define DEBUG_GPU_COPY_D2D(swapper, matId, srcDeviceId, src, dstDeviceId, dst, stream, size)                           \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr,                                                                                                    \
            "[SWAPPER][D2D_COPY] Matrix=%s srcDevice=%d src=%p dstDevice=%d "                                          \
            "dst=%p "                                                                                                  \
            "Stream=%p Size=%zu bytes\n",                                                                              \
            name.c_str(), srcDeviceId, (void*)src, dstDeviceId, (void*)dst, (void*)stream, size);                      \
  }                                                                                                                    \
  while (0)

#define DEBUG_GPU_ALLOC(swapper, matId, deviceId, stream, size)                                                        \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr, "[SWAPPER][GPU_ALLOC] Matrix=%s Device=%d Stream=%p Size=%zu bytes\n", name.c_str(), deviceId,     \
            (void*)stream, size);                                                                                      \
  }                                                                                                                    \
  while (0)

#define DEBUG_GPU_FREE(swapper, matId, deviceId, stream)                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr, "[SWAPPER][GPU_FREE] Matrix=%s Device=%d Stream=%p\n", name.c_str(), deviceId, (void*)stream);     \
  }                                                                                                                    \
  while (0)

#define DEBUG_GPU_EVICT(swapper, matId, deviceId, stream, reason)                                                      \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr, "[SWAPPER][GPU_EVICT] Matrix=%s Device=%d Stream=%p Reason=%s\n", name.c_str(), deviceId,          \
            (void*)stream, reason);                                                                                    \
  }                                                                                                                    \
  while (0)

#define DEBUG_MATRIX_PIN(swapper, matId, deviceId)                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr, "[SWAPPER][PIN] Matrix=%s Device=%d\n", name.c_str(), deviceId);                                   \
  }                                                                                                                    \
  while (0)

#define DEBUG_MATRIX_UNPIN(swapper, matId, deviceId)                                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr, "[SWAPPER][UNPIN] Matrix=%s Device=%d\n", name.c_str(), deviceId);                                 \
  }                                                                                                                    \
  while (0)

#define DEBUG_BUFFER_REUSE(swapper, matId, deviceId, stream, forWrite)                                                 \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr, "[SWAPPER][REUSE] Matrix=%s Device=%d Stream=%p ForWrite=%d\n", name.c_str(), deviceId,            \
            (void*)stream, forWrite);                                                                                  \
  }                                                                                                                    \
  while (0)

// StreamManager operation debug macros
#define DEBUG_MATMUL(swapper, result, m1, m2, alpha, deviceId, stream)                                                 \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string r = (swapper).getMatrixName(result);                                                                   \
    std::string a = (swapper).getMatrixName(m1);                                                                       \
    std::string b = (swapper).getMatrixName(m2);                                                                       \
    fprintf(stderr, "[CALCULATOR][MATMUL] %s = %s * %s * %.2lf (Device=%d Stream=%p)\n", r.c_str(), a.c_str(),         \
            b.c_str(), alpha, deviceId, (void*)stream);                                                                \
  }                                                                                                                    \
  while (0)

#define DEBUG_MATMUL_ACCU(swapper, result, m1, m2, alpha, deviceId, stream)                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string r = (swapper).getMatrixName(result);                                                                   \
    std::string a = (swapper).getMatrixName(m1);                                                                       \
    std::string b = (swapper).getMatrixName(m2);                                                                       \
    fprintf(stderr,                                                                                                    \
            "[CALCULATOR][MATMUL_ACCU] %s += %s * %s * %.2lf (Device=%d "                                              \
            "Stream=%p)\n",                                                                                            \
            r.c_str(), a.c_str(), b.c_str(), alpha, deviceId, (void*)stream);                                          \
  }                                                                                                                    \
  while (0)

#define DEBUG_ADD_ACCU(swapper, result, m1, alpha, deviceId, stream)                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string r = (swapper).getMatrixName(result);                                                                   \
    std::string a = (swapper).getMatrixName(m1);                                                                       \
    fprintf(stderr, "[CALCULATOR][ADD_ACCU] %s += %s * %.2f (Device=%d Stream=%p)\n", r.c_str(), a.c_str(), alpha,     \
            deviceId, (void*)stream);                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_SYNC(swapper, mat, deviceId, stream)                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(mat);                                                                   \
    fprintf(stderr, "[CALCULATOR][SYNC] Matrix=%s (Device=%d Stream=%p)\n", name.c_str(), deviceId, (void*)stream);    \
  }                                                                                                                    \
  while (0)

// Arranger workflow debug macros
#define DEBUG_TERM_START(swapper, termIdx, rIdx, aIdx, bIdx, cIdx, fval, deviceId)                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string r = (swapper).getMatrixName(rIdx);                                                                     \
    std::string a = (swapper).getMatrixName(aIdx);                                                                     \
    std::string b = (swapper).getMatrixName(bIdx);                                                                     \
    std::string c = (swapper).getMatrixName(cIdx);                                                                     \
    fprintf(stderr, "[ARRANGER][TERM_%d] R=%s A=%s B=%s C=%s fval=%.2f (Device=%d)\n", termIdx, r.c_str(), a.c_str(),  \
            b.c_str(), c.c_str(), fval, deviceId);                                                                     \
  }                                                                                                                    \
  while (0)

#define DEBUG_INTER_REUSE(swapper, interMat, bIdx, cIdx)                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string inter = (swapper).getMatrixName(interMat);                                                             \
    std::string b = (swapper).getMatrixName(bIdx);                                                                     \
    std::string c = (swapper).getMatrixName(cIdx);                                                                     \
    fprintf(stderr, "[ARRANGER][INTER_REUSE] %s (from B=%s C=%s)\n", inter.c_str(), b.c_str(), c.c_str());             \
  }                                                                                                                    \
  while (0)

#define DEBUG_COMBINE_START(swapper, combineMat, rIdx, aIdx)                                                           \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string combine = (swapper).getMatrixName(combineMat);                                                         \
    std::string r = (swapper).getMatrixName(rIdx);                                                                     \
    std::string a = (swapper).getMatrixName(aIdx);                                                                     \
    fprintf(stderr, "[ARRANGER][COMBINE_START] %s (for R=%s A=%s)\n", combine.c_str(), r.c_str(), a.c_str());          \
  }                                                                                                                    \
  while (0)

#define DEBUG_COMBINE_FINALIZE(swapper, combineMat, rMat)                                                              \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string combine = (swapper).getMatrixName(combineMat);                                                         \
    std::string r = (swapper).getMatrixName(rMat);                                                                     \
    fprintf(stderr, "[ARRANGER][COMBINE_FINALIZE] %s -> %s\n", combine.c_str(), r.c_str());                            \
  }                                                                                                                    \
  while (0)

#define DEBUG_PRESTORE(swapper, matId, mpi_rank, deviceId)                                                             \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr, "[ARRANGER][PRESTORE] Node=%d Device=%d Matrix = %s\n", mpi_rank, deviceId, name.c_str());         \
  }                                                                                                                    \
  while (0)

#define DEBUG_BARRIER_START(deviceId, target, value)                                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr,                                                                                                    \
            "[CALCULATOR][BARRIER_START] value=%d target=%d "                                                          \
            "Device=%d\n",                                                                                             \
            value, target, deviceId);                                                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_BARRIER_END(deviceId)                                                                                    \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[CALCULATOR][BARRIER_END] Device=%d\n", deviceId);                                                \
  }                                                                                                                    \
  while (0)

#define DEBUG_P2P_AVAILABLE(srcDevice, dstDevice, canAccess)                                                           \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[ARRANGER][P2P_AVAILABLE] Device %d -> Device %d: %s\n", srcDevice, dstDevice,                    \
            canAccess ? "YES" : "NO");                                                                                 \
  }                                                                                                                    \
  while (0)

#define DEBUG_P2P_ENABLED(srcDevice, dstDevice, success)                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[ARRANGER][P2P_ENABLED] Device %d -> Device %d: %s\n", srcDevice, dstDevice,                      \
            success ? "SUCCESS" : "FAILED");                                                                           \
  }                                                                                                                    \
  while (0)

#define DEBUG_SIGNAL_WORK(swapper, matId, deviceId, currentValue)                                                      \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr, "[CALCULATOR][SIGNAL] Matrix=%s Device=%d Counter: %d -> %d\n", name.c_str(), deviceId,            \
            currentValue, currentValue + 1);                                                                           \
  }                                                                                                                    \
  while (0)

#define DEBUG_WAIT_WORK(swapper, matId, deviceId, currentValue, targetValue)                                           \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr,                                                                                                    \
            "[CALCULATOR][WAIT] Matrix=%s Device=%d Counter: %d (waiting for "                                         \
            "%d)\n",                                                                                                   \
            name.c_str(), deviceId, currentValue, targetValue);                                                        \
  }                                                                                                                    \
  while (0)

// NCCL communication debug macros
#define DEBUG_NCCL_GROUP_START(deviceId)                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[NCCL][GROUP_START] Device=%d\n", deviceId);                                                      \
  }                                                                                                                    \
  while (0)

#define DEBUG_NCCL_GROUP_END(deviceId)                                                                                 \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[NCCL][GROUP_END] Device=%d\n", deviceId);                                                        \
  }                                                                                                                    \
  while (0)

#define DEBUG_NCCL_SEND(swapper, matId, srcDevice, dstDevice, size, stream)                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr,                                                                                                    \
            "[NCCL][SEND] Matrix=%s Device=%d -> Device=%d Size=%zu "                                                  \
            "Stream=%p\n",                                                                                             \
            name.c_str(), srcDevice, dstDevice, size, (void*)stream);                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_NCCL_RECV(swapper, matId, srcDevice, dstDevice, size, stream)                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr,                                                                                                    \
            "[NCCL][RECV] Matrix=%s Device=%d <- Device=%d Size=%zu "                                                  \
            "Stream=%p\n",                                                                                             \
            name.c_str(), dstDevice, srcDevice, size, (void*)stream);                                                  \
  }                                                                                                                    \
  while (0)

#define DEBUG_NCCL_REDUCE(swapper, matId, root, deviceId, id, count, comm, stream)                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr,                                                                                                    \
            "[NCCL][REDUCE] Matrix=%s Root=%d Device=%d ID=%d count=%d "                                               \
            "ncclComm_t=%p Stream=%p\n",                                                                               \
            name.c_str(), root, deviceId, id, count, (void*)comm, (void*)stream);                                      \
  }                                                                                                                    \
  while (0)

#define DEBUG_NCCL_COMM_CREATE_START(NodeID, numRanks, idPtr)                                                          \
  do                                                                                                                   \
  {                                                                                                                    \
    unsigned char* bytes = (unsigned char*)(idPtr);                                                                    \
    fprintf(stderr,                                                                                                    \
            "[NCCL][COMM_CREATE_START] Node=%d NumRanks=%d ncclUniqueId=%p "                                           \
            "[%02x%02x"                                                                                                \
            "%02x%02x...]\n",                                                                                          \
            NodeID, numRanks, (void*)idPtr, bytes[0], bytes[1], bytes[2], bytes[3]);                                   \
  }                                                                                                                    \
  while (0)

#define DEBUG_NCCL_COMM_INIT(NodeId, deviceId, rank, numRanks, commPtr, idPtr)                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    unsigned char* bytes = (unsigned char*)(idPtr);                                                                    \
    fprintf(stderr,                                                                                                    \
            "[NCCL][COMM_INIT] Node=%d Device=%d Rank=%d/%d ncclComm_t=%p "                                            \
            "ncclUniqueId=%p [%02x%02x%02x%02x...]\n",                                                                 \
            NodeId, deviceId, rank, numRanks, (void*)commPtr, (void*)idPtr, bytes[0], bytes[1], bytes[2], bytes[3]);   \
  }                                                                                                                    \
  while (0)

#define DEBUG_NCCL_COMM_CREATE_END(NodeId, numRanks)                                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[NCCL][COMM_CREATE_END] Node=%d NumRanks=%d\n", NodeId, numRanks);                                \
  }                                                                                                                    \
  while (0)

#define DEBUG_MEMSET(swapper, matId, deviceId, size, stream)                                                           \
  do                                                                                                                   \
  {                                                                                                                    \
    std::string name = (swapper).getMatrixName(matId);                                                                 \
    fprintf(stderr, "[CALCULATOR][MEMSET] Matrix=%s Device=%d Size=%zu Stream=%p\n", name.c_str(), deviceId, size,     \
            (void*)stream);                                                                                            \
  }                                                                                                                    \
  while (0)

#define DEBUG_FREE_BEGIN(deviceId)                                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[CALCULATOR][FREE_BEGIN] Device=%d\n", deviceId);                                                 \
  }                                                                                                                    \
  while (0)

#define DEBUG_FREE_END(deviceId)                                                                                       \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[CALCULATOR][FREE_END] Device=%d\n", deviceId);                                                   \
  }                                                                                                                    \
  while (0)

#define DEBUG_MEM_INFO(deviceId, freeMem, totalMem)                                                                    \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[MEMORY][INFO] Device=%d Free=%.2fGB Total=%.2fGB\n", deviceId,                                   \
            (freeMem) / (1024.0 * 1024.0 * 1024.0), (totalMem) / (1024.0 * 1024.0 * 1024.0));                          \
  }                                                                                                                    \
  while (0)

#define DEBUG_MEM_POOL_MODE(useDefault)                                                                                \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[MEMORY][POOL_MODE] Mode=%s\n", (useDefault) ? "DEFAULT" : "CUSTOM");                             \
  }                                                                                                                    \
  while (0)

#define DEBUG_NCCL_HEADROOM(ncclHeadroom)                                                                              \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[MEMORY][NCCL_HEADROOM] Size=%.2fGB\n", (ncclHeadroom) / (1024.0 * 1024.0 * 1024.0));             \
  }                                                                                                                    \
  while (0)

// Arranger worklist execution debug macros
#define DEBUG_ARRANGER_FREE_MEM(deviceId, freeMem)                                                                     \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[ARRANGER][FREE_MEMORY] Device=%d Free=%.2fGB\n", deviceId,                                       \
            (freeMem) / (1024.0 * 1024.0 * 1024.0));                                                                   \
  }                                                                                                                    \
  while (0)

#define DEBUG_ARRANGER_WORKLIST_START(listType)                                                                        \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[ARRANGER][WORKLIST] Executing %s worklist\n", listType);                                         \
  }                                                                                                                    \
  while (0)

#define DEBUG_ARRANGER_WORK_ITEM(deviceId, idx, workType)                                                              \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[ARRANGER][WORK] Device=%d Idx=%d Type=%s\n", deviceId, idx, workType);                           \
  }                                                                                                                    \
  while (0)

#define DEBUG_ARRANGER_BATCH_RANGE(deviceId, startIdx, endIdx, total)                                                  \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[ARRANGER][BATCH] Device=%d Range=[%d,%d) Total=%d\n", deviceId, startIdx, endIdx, total);        \
  }                                                                                                                    \
  while (0)

#define DEBUG_ARRANGER_ALL_FINISHED(finished)                                                                          \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[ARRANGER][STATUS] All finished: %s\n", (finished) ? "YES" : "NO");                               \
  }                                                                                                                    \
  while (0)
#define DEBUG_MATRIX_DISTRIBUTION(swapper, rank, mat_id)                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[ARRANGER][MPI] Node=%d get matrix %s (ID=%d)\n", rank, (swapper).getMatrixName(mat_id).c_str(),  \
            mat_id);                                                                                                   \
  }                                                                                                                    \
  while (0)

#define DEBUG_TERM_DISTRIBUTION(rank, r, a, b, c, f)                                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[ARRANGER][MPI] Node=%d R%d += A%d * B%d * C%d * %.2f\n", rank, r, a, b, c, f);                   \
  }                                                                                                                    \
  while (0)
#define DEBUG_HIT(swapper, mat_id, device_id, size)                                                                    \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, "[SWAPPER][HIT] Device=%d Matrix %s hit, Size=%zu\n", device_id,                                   \
            (swapper).getMatrixName(mat_id).c_str(), size);                                                            \
  }                                                                                                                    \
  while (0)
#define DEBUG_SECOND_HIT(swapper, mat_id, src_device_id, target_device_id, size)                                       \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr,                                                                                                    \
            "[SWAPPER][SECOND_HIT] DeviceId=%d Matrix %s hit on Device=%d, "                                           \
            "Size=%zu\n",                                                                                              \
            target_device_id, (swapper).getMatrixName(mat_id).c_str(), src_device_id, size);                           \
  }                                                                                                                    \
  while (0)
#define DEBUG_THIRD_HIT(swapper, mat_id, src_device_id, target_device_id, size)                                        \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr,                                                                                                    \
            "[SWAPPER][THIRD_HIT] DeviceId=%d Matrix %s hit on Device=%d, "                                            \
            "Size=%zu\n",                                                                                              \
            target_device_id, (swapper).getMatrixName(mat_id).c_str(), src_device_id, size);                           \
  }                                                                                                                    \
  while (0)

#else
// No-op macros when debug is disabled
#define DEBUG_PRINT(fmt, ...)                                                                                          \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_GPU_COPY_H2D(swapper, matId, deviceId, stream, size)                                                     \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_GPU_COPY_D2H(swapper, matId, deviceId, stream, size)                                                     \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_GPU_COPY_D2D(swapper, matId, srcDeviceId, src, dstDeviceId, dst, stream, size)                           \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_GPU_ALLOC(swapper, matId, deviceId, stream, size)                                                        \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_GPU_FREE(swapper, matId, deviceId, stream)                                                               \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_GPU_EVICT(swapper, matId, deviceId, stream, reason)                                                      \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_MATRIX_PIN(swapper, matId, deviceId)                                                                     \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_MATRIX_UNPIN(swapper, matId, deviceId)                                                                   \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_BUFFER_REUSE(swapper, matId, deviceId, stream, forWrite)                                                 \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_MATMUL(swapper, result, m1, m2, alpha, deviceId, stream)                                                 \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_MATMUL_ACCU(swapper, result, m1, m2, alpha, deviceId, stream)                                            \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_ADD_ACCU(swapper, result, m1, alpha, deviceId, stream)                                                   \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_SYNC(swapper, mat, deviceId, stream)                                                                     \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TERM_START(swapper, termIdx, rIdx, aIdx, bIdx, cIdx, fval, deviceId)                                     \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_INTER_REUSE(swapper, interMat, bIdx, cIdx)                                                               \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_COMBINE_START(swapper, combineMat, rIdx, aIdx)                                                           \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_COMBINE_FINALIZE(swapper, combineMat, rMat)                                                              \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_PRESTORE(swapper, matId, mpi_rank, deviceId)                                                             \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_BARRIER_START(deviceId, target, value)                                                                   \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_BARRIER_END(deviceId)                                                                                    \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_P2P_AVAILABLE(srcDevice, dstDevice, canAccess)                                                           \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_P2P_ENABLED(srcDevice, dstDevice, success)                                                               \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_SIGNAL_WORK(swapper, matId, deviceId, currentValue)                                                      \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_WAIT_WORK(swapper, matId, deviceId, currentValue, targetValue)                                           \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_NCCL_GROUP_START(deviceId)                                                                               \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_NCCL_GROUP_END(deviceId)                                                                                 \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_NCCL_SEND(swapper, matId, srcDevice, dstDevice, size, stream)                                            \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_NCCL_RECV(swapper, matId, srcDevice, dstDevice, size, stream)                                            \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_NCCL_REDUCE(swapper, matId, root, deviceId, id, count, comm, stream)                                     \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_NCCL_COMM_CREATE_START(NodeID, numRanks, idPtr)                                                          \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_NCCL_COMM_INIT(NodeId, deviceId, rank, numRanks, commPtr, idPtr)                                         \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_NCCL_COMM_CREATE_END(NodeId, numRanks)                                                                   \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_MEMSET(swapper, matId, deviceId, size, stream)                                                           \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_FREE_BEGIN(deviceId)                                                                                     \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_FREE_END(deviceId)                                                                                       \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_MEM_INFO(deviceId, freeMem, totalMem)                                                                    \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_MEM_POOL_MODE(useDefault)                                                                                \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_NCCL_HEADROOM(ncclHeadroom)                                                                              \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_ARRANGER_FREE_MEM(deviceId, freeMem)                                                                     \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_ARRANGER_WORKLIST_START(listType)                                                                        \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_ARRANGER_WORK_ITEM(deviceId, idx, workType)                                                              \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_ARRANGER_BATCH_RANGE(deviceId, startIdx, endIdx, total)                                                  \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)

#define DEBUG_ARRANGER_ALL_FINISHED(finished)                                                                          \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_MATRIX_DISTRIBUTION(swapper, rank, mat_id)                                                               \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_TERM_DISTRIBUTION(rank, r, a, b, c, f)                                                                   \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_HIT(swapper, mat_id, device_id, size)                                                                    \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_SECOND_HIT(swapper, mat_id, src_device_id, target_device_id, size)                                       \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#define DEBUG_THIRD_HIT(swapper, mat_id, src_device_id, target_device_id, size)                                        \
  do                                                                                                                   \
  {}                                                                                                                   \
  while (0)
#endif
