#pragma once

#include <cublas_v2.h>
#include <nccl.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <tuple>

namespace tensor
{

using TermTy = std::tuple<int, int, int, int, double>;

inline int resolveActiveCudaDeviceCount(int visibleDeviceCount)
{
  if (visibleDeviceCount <= 0)
  {
    return 0;
  }

  auto const* devices = std::getenv("UNI20_TENSORCONTRACTION_DEVICES");
  if (devices == nullptr)
  {
    devices = std::getenv("TENSORCONTRACTION_DEVICES");
  }
  if (devices == nullptr)
  {
    // TensorContraction's CUDA/MPI path is intended to run one CUDA device per
    // MPI process.  The vendored default used every visible GPU in every
    // process, which creates large CUDA/NCCL virtual-address reservations.
    return 1;
  }

  std::string value(devices);
  if (value == "all" || value == "ALL")
  {
    return visibleDeviceCount;
  }

  try
  {
    return std::clamp(std::stoi(value), 1, visibleDeviceCount);
  }
  catch (...)
  {
    return 1;
  }
}

inline bool envFlagEnabled(const char* name)
{
  auto const* value = std::getenv(name);
  if (value == nullptr)
  {
    return false;
  }

  std::string text(value);
  return !(text.empty() || text == "0" || text == "OFF" || text == "off" || text == "false" || text == "FALSE");
}

inline int resolveTensorContractionStreamCount()
{
  auto const* env = std::getenv("UNI20_TENSORCONTRACTION_STREAMS");
  if (env == nullptr)
  {
    return 4;
  }

  try
  {
    return std::max(1, std::stoi(env));
  }
  catch (...)
  {
    return 4;
  }
}

inline int resolveTensorContractionCublasStreamCount(int workStreamCount)
{
  auto const* env = std::getenv("UNI20_TENSORCONTRACTION_CUBLAS_STREAMS");
  if (env != nullptr)
  {
    try
    {
      return std::max(1, std::stoi(env));
    }
    catch (...)
    {
      return std::max(1, std::min(2, workStreamCount));
    }
  }

  return std::max(1, std::min(2, workStreamCount));
}

inline std::size_t resolveTensorContractionCublasWorkspaceBytes()
{
  auto const* env = std::getenv("UNI20_TENSORCONTRACTION_CUBLAS_WORKSPACE_BYTES");
  if (env != nullptr)
  {
    try
    {
      return std::max<std::size_t>(16 * 1024, static_cast<std::size_t>(std::stoull(env)));
    }
    catch (...)
    {}
  }

  return 4ULL * 1024ULL * 1024ULL;
}

inline std::size_t resolveTensorContractionPoolCacheBytes()
{
  auto const* env = std::getenv("UNI20_TENSORCONTRACTION_POOL_CACHE_BYTES");
  if (env != nullptr)
  {
    try
    {
      return static_cast<std::size_t>(std::stoull(env));
    }
    catch (...)
    {}
  }

  return 256ULL * 1024ULL * 1024ULL;
}

} // namespace tensor

#define CUDA_CALL(func)                                                                                                \
  do                                                                                                                   \
  {                                                                                                                    \
    cudaError_t err = (func);                                                                                          \
    if (err != cudaSuccess)                                                                                            \
    {                                                                                                                  \
      fprintf(stderr, "CUDA Error at %s:%d\n", __FILE__, __LINE__);                                                    \
      fprintf(stderr, "  Error code: %d\n", err);                                                                      \
      fprintf(stderr, "  Error text: %s\n", cudaGetErrorString(err));                                                  \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

// Convenience macro for CUBLAS function calls
#define CUBLAS_CALL(func)                                                                                              \
  do                                                                                                                   \
  {                                                                                                                    \
    cublasStatus_t err = (func);                                                                                       \
    if (err != CUBLAS_STATUS_SUCCESS)                                                                                  \
    {                                                                                                                  \
      fprintf(stderr, "CUBLAS Error at %s:%d\n", __FILE__, __LINE__);                                                  \
      fprintf(stderr, "  Error code: %d\n", err);                                                                      \
      fprintf(stderr, "  Error text: %s\n", cublasGetStatusString(err));                                               \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)

// Convenience macro for NCCL function calls
#define NCCL_CALL(func)                                                                                                \
  do                                                                                                                   \
  {                                                                                                                    \
    ncclResult_t err = (func);                                                                                         \
    if (err != ncclSuccess)                                                                                            \
    {                                                                                                                  \
      fprintf(stderr, "NCCL Error at %s:%d\n", __FILE__, __LINE__);                                                    \
      fprintf(stderr, "  Error code: %d\n", err);                                                                      \
      fprintf(stderr, "  Error text: %s\n", ncclGetErrorString(err));                                                  \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  }                                                                                                                    \
  while (0)
