#pragma once

#include <cublas_v2.h>
#include <nccl.h>

#include <iostream>
#include <tuple>

namespace tensor
{

using TermTy = std::tuple<int, int, int, int, double>;

}

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
