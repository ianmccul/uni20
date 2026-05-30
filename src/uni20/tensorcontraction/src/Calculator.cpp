#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <cassert>
#include <mutex>

#include "Calculator.hpp"
#include "Debug.hpp"
#include "Swapper.hpp"
#include "Utils.h"

namespace tensor {

void WorkBase::execute() {
  // Default implementation - does nothing
}

void MatMulWork::execute() {
  const auto &mats = getMatrices();
  int deviceId = streamManager.getDeviceId();
  // matrices: [result, m1, m2]
  const Matrix &result = mats[0];
  const Matrix &m1 = mats[1];
  const Matrix &m2 = mats[2];

  cudaStream_t stream = streamManager.setEnv();

  std::shared_ptr<GpuBuffer> m1OnGPU = swapper.getForRead(m1, deviceId, stream);
  std::shared_ptr<GpuBuffer> m2OnGPU = swapper.getForRead(m2, deviceId, stream);
  std::shared_ptr<GpuBuffer> resultOnGPU =
      swapper.getForWrite(result, deviceId, stream);

  double alpha = getAlpha();
  double beta = 0.0;

  DEBUG_MATMUL(swapper, result.getId(), m1.getId(), m2.getId(), alpha, deviceId,
               stream);

  assert(m1OnGPU);
  assert(m2OnGPU);
  assert(resultOnGPU);

  CUBLAS_CALL(cublasDgemm(
      streamManager.getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, m2.getSecondDim(),
      m1.getFirstDim(), m1.getSecondDim(), &alpha, m2OnGPU->getPtr(),
      m2.getSecondDim(), m1OnGPU->getPtr(), m1.getSecondDim(), &beta,
      resultOnGPU->getPtr(), result.getSecondDim()));

  resultOnGPU->notifyWriteFinish(stream);
  m1OnGPU->notifyReadFinish(stream);
  m2OnGPU->notifyReadFinish(stream);
}

void MatMulAccuWork::execute() {
  const auto &mats = getMatrices();
  // matrices: [result, m1, m2]
  cudaStream_t stream = streamManager.setEnv();

  std::shared_ptr<GpuBuffer> m1OnGPU =
      swapper.getForRead(mats[1], streamManager.getDeviceId(), stream);
  std::shared_ptr<GpuBuffer> m2OnGPU =
      swapper.getForRead(mats[2], streamManager.getDeviceId(), stream);
  std::shared_ptr<GpuBuffer> resultOnGPU =
      swapper.getForWrite(mats[0], streamManager.getDeviceId(), stream);

  double alpha = getAlpha();
  double beta = 1.0;

  DEBUG_MATMUL_ACCU(swapper, mats[0].getId(), mats[1].getId(), mats[2].getId(),
                    alpha, streamManager.getDeviceId(), stream);
  assert(m1OnGPU);
  assert(m2OnGPU);
  assert(resultOnGPU);

  CUBLAS_CALL(cublasDgemm(streamManager.getHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
                          mats[2].getSecondDim(), mats[1].getFirstDim(),
                          mats[1].getSecondDim(), &alpha, m2OnGPU->getPtr(),
                          mats[2].getSecondDim(), m1OnGPU->getPtr(),
                          mats[1].getSecondDim(), &beta, resultOnGPU->getPtr(),
                          mats[0].getSecondDim()));

  resultOnGPU->notifyWriteFinish(stream);
  m1OnGPU->notifyReadFinish(stream);
  m2OnGPU->notifyReadFinish(stream);
}

void AddAccuWork::execute() {
  const auto &mats = getMatrices();
  // matrices: [result, m1]
  cudaStream_t stream = streamManager.setEnv();

  std::shared_ptr<GpuBuffer> m1OnGPU =
      swapper.getForRead(mats[1], streamManager.getDeviceId(), stream);
  std::shared_ptr<GpuBuffer> resultOnGPU =
      swapper.getForWrite(mats[0], streamManager.getDeviceId(), stream);

  double alpha = getAlpha();
  DEBUG_ADD_ACCU(swapper, mats[0].getId(), mats[1].getId(), alpha,
                 streamManager.getDeviceId(), stream);

  assert(m1OnGPU);
  assert(resultOnGPU);

  CUBLAS_CALL(cublasDaxpy(streamManager.getHandle(), mats[1].size(), &alpha,
                          m1OnGPU->getPtr(), 1, resultOnGPU->getPtr(), 1));

  resultOnGPU->notifyWriteFinish(stream);
  m1OnGPU->notifyReadFinish(stream);
}

void InnerProductWork::execute() {
  const auto &mats = getMatrices();
  // matrices: [m1, m2]
  int deviceId = streamManager.getDeviceId();
  cudaStream_t stream = streamManager.setEnv();

  std::shared_ptr<GpuBuffer> m1OnGPU =
      swapper.getForRead(mats[0], deviceId, stream);
  std::shared_ptr<GpuBuffer> m2OnGPU =
      swapper.getForRead(mats[1], deviceId, stream);

  assert(m1OnGPU);
  assert(m2OnGPU);

  double *deviceDotResult;
  CUDA_CALL(cudaMalloc(&deviceDotResult, sizeof(double)));

  CUBLAS_CALL(cublasSetPointerMode(streamManager.getHandle(),
                                   CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_CALL(cublasDdot(streamManager.getHandle(), mats[0].size(),
                         m1OnGPU->getPtr(), 1, m2OnGPU->getPtr(), 1,
                         deviceDotResult));
  CUBLAS_CALL(cublasSetPointerMode(streamManager.getHandle(),
                                   CUBLAS_POINTER_MODE_HOST));

  CUDA_CALL(cudaMemcpyAsync(result, deviceDotResult, sizeof(double),
                            cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaFreeAsync(deviceDotResult, stream));

  m1OnGPU->notifyReadFinish(stream);
  m2OnGPU->notifyReadFinish(stream);
}

void ScalarMulWork::execute() {
  const auto &mats = getMatrices();
  // matrices: [mat]
  int deviceId = streamManager.getDeviceId();
  cudaStream_t stream = streamManager.setEnv();

  std::shared_ptr<GpuBuffer> buffer =
      swapper.getForWrite(mats[0], deviceId, stream);

  assert(buffer);

  double alpha = *coff;
  CUBLAS_CALL(cublasDscal(streamManager.getHandle(), mats[0].size(), &alpha,
                          buffer->getPtr(), 1));

  buffer->notifyWriteFinish(stream);
}

void BarrierWork::execute() {
  int orig = count->fetch_add(1);
  DEBUG_BARRIER_START(streamManager.getDeviceId(), target, orig);
  while (*count != target) {
  }
  DEBUG_BARRIER_END(streamManager.getDeviceId());
};

void SyncWork::execute() {
  auto mat = getMatrices()[0];
  // matrices: [mat]
  cudaStream_t stream = streamManager.setEnv();

  auto [matDeviceId, buffer] = swapper.getPreStoreBufferOrNone(mat);
  auto event = getSyncFinishEvent();

  if (!buffer || matDeviceId != streamManager.getDeviceId()) {
    DEBUG_SYNC(swapper, mat.getId(), streamManager.getDeviceId(), stream);
    swapper.syncBuffer(mat, streamManager.getDeviceId(), stream);
  } else if (event) {
    buffer->waitForWriteFinish(stream);
  }

  if (event) {
    CUDA_CALL(cudaEventRecord(event, stream));
  }
}

void UnpinWork::execute() {
  const auto &mats = getMatrices();
  // matrices: [mat]
  swapper.unpinMatrix(mats[0], streamManager.getDeviceId());
}

void PinWork::execute() {
  const auto &mats = getMatrices();
  // matrices: [mat]
  swapper.pinMatrix(mats[0], streamManager.getDeviceId());
}

void FreeWork::execute() {
  int deviceId = streamManager.getDeviceId();
  DEBUG_FREE_BEGIN(deviceId);
  auto stream = streamManager.setEnv();
  for (auto mat : getMatrices()) {
    swapper.freeBuffer(mat, deviceId, stream);
  }
  DEBUG_FREE_END(deviceId);
}

void NCCLSendRecvWork::execute() {
  int mpi_rank;
  int device_count;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  CUDA_CALL(cudaGetDeviceCount(&device_count));

  int thisDeviceId = streamManager.getDeviceId();
  int this_nccl_id = mpi_rank * device_count + thisDeviceId;
  auto stream = streamManager.setEnv();

  std::vector<std::tuple<int, std::shared_ptr<GpuBuffer>, Matrix>>
      dstBufferMatTuple;

  for (auto [dst_nccl_id, mat, syncFinishEvent] : dstMatPair) {
    if (dst_nccl_id == this_nccl_id) {
      // This matrix is requested by this device. Allocate a buffer for it.
      auto buffer = swapper.getForWrite(mat, thisDeviceId, stream);
      assert(buffer);
      dstBufferMatTuple.push_back({dst_nccl_id, buffer, mat});
    } else {
      dstBufferMatTuple.push_back({dst_nccl_id, nullptr, mat});
      auto [srcDeviceId, srcBuffer] = swapper.getPreStoreBufferOrNone(mat);
      int src_nccl_id =
          srcDeviceId == -1 ? -1 : mpi_rank * device_count + srcDeviceId;

      // This matrix is requested by other device, and it sits in this device's
      // memory. We need to make sure it has been written completely before
      // doing the transfer.
      if (src_nccl_id == this_nccl_id && syncFinishEvent) {
        CUDA_CALL(cudaStreamWaitEvent(stream, syncFinishEvent));
      }
    }
  }

  NCCL_CALL(ncclGroupStart());
  DEBUG_NCCL_GROUP_START(thisDeviceId);
  for (auto [dst_nccl_id, dstBuffer, mat] : dstBufferMatTuple) {
    auto [srcDeviceId, srcBuffer] = swapper.getPreStoreBufferOrNone(mat);
    int src_nccl_id;
    if (srcDeviceId != -1) {
      src_nccl_id = mpi_rank * device_count + srcDeviceId;
    } else {
      src_nccl_id = swapper.getRemoteNCCLId(mat.getId());
    }

    if (dst_nccl_id == this_nccl_id && src_nccl_id != this_nccl_id) {
      DEBUG_NCCL_RECV(swapper, mat.getId(), src_nccl_id, this_nccl_id,
                      mat.size(), stream);
      NCCL_CALL(ncclRecv(dstBuffer->getPtr(), dstBuffer->size(), ncclDouble,
                         src_nccl_id, comm, stream));
    } else if (dst_nccl_id != this_nccl_id && src_nccl_id == this_nccl_id) {
      DEBUG_NCCL_SEND(swapper, mat.getId(), this_nccl_id, dst_nccl_id,
                      mat.size(), stream);
      NCCL_CALL(ncclSend(srcBuffer->getPtr(), srcBuffer->size(), ncclDouble,
                         dst_nccl_id, comm, stream));
    }
  }
  NCCL_CALL(ncclGroupEnd());
  DEBUG_NCCL_GROUP_END(thisDeviceId);

  for (auto [_, buffer, __] : dstBufferMatTuple) {
    if (buffer) {
      buffer->notifyWriteFinish(stream);
    }
  }
}

void MemsetWork::execute() {
  auto stream = streamManager.setEnv();
  auto buffer = swapper.getForWrite(getMatrices()[0],
                                    streamManager.getDeviceId(), stream);
  DEBUG_MEMSET(swapper, getMatrices()[0].getId(), streamManager.getDeviceId(),
               buffer->sizeInByte(), stream);
  CUDA_CALL(cudaMemsetAsync(buffer->getPtr(), 0, buffer->sizeInByte(), stream));
  buffer->notifyWriteFinish(stream);
}

StreamManager::StreamManager(Swapper &swapper, int deviceId, int deviceCount)
    : swapper(swapper), deviceId(deviceId), streamIdx(0) {
  CUDA_CALL(cudaSetDevice(deviceId));
  CUBLAS_CALL(cublasCreate(&handle));
  for (int i = 0; i < streamCount; i++) {
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    streams.push_back(stream);
  }
}

void StreamManager::clear() {
  cudaSetDevice(deviceId);
  for (auto stream : streams) {
    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaStreamDestroy(stream));
  }

  CUBLAS_CALL(cublasDestroy(handle));
}

cudaStream_t StreamManager::getStream() {
  cudaStream_t stream = streams[streamIdx];
  streamIdx = (streamIdx + 1) % streamCount;
  return stream;
}

cudaStream_t StreamManager::setEnv() {
  CUDA_CALL(cudaSetDevice(deviceId));
  cudaStream_t stream = getStream();
  CUBLAS_CALL(cublasSetStream(handle, stream));
  return stream;
}

void StreamManager::syncAllStreams() const {
  cudaSetDevice(deviceId);
  for (auto stream : streams) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
}

#if DEBUG_LOG
// Base implementation of dump()
void WorkBase::dump() {
  std::string matrixInfo = getMatrixInfo();
  std::string typeInfo = getTypeSpecificInfo();
  fprintf(stderr, "[LIST][DEV_%d][%s] %s%s\n", streamManager.getDeviceId(),
          getTypeName(), matrixInfo.c_str(), typeInfo.c_str());
}

// Default matrix info for MatWorkBase - show all matrices with names
std::string MatWorkBase::getMatrixInfo() const {
  std::string result;
  for (size_t i = 0; i < matrices.size(); i++) {
    if (i > 0) result += ", ";
    result += swapper.getMatrixName(matrices[i].getId());
  }
  return result;
}

// Type-specific info implementations
std::string MatMulAccuWork::getTypeSpecificInfo() const {
  char buf[64];
  snprintf(buf, sizeof(buf), " (alpha=%.2f)", alpha);
  return std::string(buf);
}

std::string AddAccuWork::getTypeSpecificInfo() const {
  char buf[64];
  snprintf(buf, sizeof(buf), " (alpha=%.2f)", alpha);
  return std::string(buf);
}

std::string NCCLSendRecvWork::getMatrixInfo() const {
  std::string result;
  for (size_t i = 0; i < dstMatPair.size(); i++) {
    if (i > 0) result += ", ";
    auto [deviceId, mat, _] = dstMatPair[i];
    result += swapper.getMatrixName(mat.getId());
    char buf[32];
    snprintf(buf, sizeof(buf), "(dev=%d)", deviceId);
    result += buf;
  }
  return result;
}

std::string BarrierWork::getTypeSpecificInfo() const {
  char buf[64];
  snprintf(buf, sizeof(buf), " (target=%d)", target);
  return std::string(buf);
}
#endif

}  // namespace tensor
