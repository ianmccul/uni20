#include "../helpers.hpp"
#include "kernel/contract.hpp"
#include "gtest/gtest.h"
#include <numeric>

using namespace uni20;
using uni20::cpu_tag; // our tag‐dispatch for now

using namespace uni20::kernel;

// A tiny reference for 2D matmul C = α·A·B + β·C
static void naive_matmul_2d(size_t M, size_t K, size_t N, double alpha, double const* A, ptrdiff_t sA_i, ptrdiff_t sA_k,
                            double const* B, ptrdiff_t sB_k, ptrdiff_t sB_j, double beta, double* C, ptrdiff_t sC_i,
                            ptrdiff_t sC_j)
{
  for (size_t i = 0; i < M; ++i)
  {
    for (size_t j = 0; j < N; ++j)
    {
      double acc{};
      const double* ap = A + i * sA_i;
      const double* bp = B + j * sB_j;
      for (size_t k = 0; k < K; ++k)
      {
        acc += ap[k * sA_k] * bp[k * sB_k];
      }
      C[i * sC_i + j * sC_j] = (beta * C[i * sC_i + j * sC_j]) + (alpha * acc);
    }
  }
}

// Test: 2D row‐major matmul
TEST(ContractKernel2D, RowMajorMatmul)
{
  size_t M = 2, K = 3, N = 4;
  std::vector<double> av(M * K), bv(K * N), cv(M * N);
  std::iota(av.begin(), av.end(), 1.0);
  std::iota(bv.begin(), bv.end(), 10.0);
  std::fill(cv.begin(), cv.end(), 5.0);

  auto A = make_mdspan_2d(av, M, K);
  auto B = make_mdspan_2d(bv, K, N);
  auto C = make_mdspan_2d(cv, M, N);

  // contract A.dim1↔B.dim0
  std::array<std::pair<size_t, size_t>, 1> Kdims{{{1, 0}}};
  double alpha = 2.0, beta = 0.5;

  // reference
  std::vector<double> cref = cv;
  naive_matmul_2d(M, K, N, alpha, av.data(), K, 1, bv.data(), N, 1, beta, cref.data(), N, 1);

  // **NEW** kernel call:
  contract(alpha, A, B, Kdims, beta, C, cpu_tag{});

  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j)
      EXPECT_DOUBLE_EQ((C[i, j]), cref[i * N + j]);
}

// Test: B in column-major layout
TEST(ContractKernel2D, ColumnMajorB)
{
  size_t M = 2, K = 3, N = 4;
  std::vector<double> av(M * K), bv(K * N), cv(M * N, 0.0);
  std::iota(av.begin(), av.end(), 1.0);
  std::iota(bv.begin(), bv.end(), 10.0);

  auto A = make_mdspan_2d(av, M, K);
  // B strides = {1, K}  ⇒ column-major
  auto B = make_mdspan_2d(bv, K, N, {1, static_cast<ptrdiff_t>(K)});
  auto C = make_mdspan_2d(cv, M, N);

  std::array<std::pair<size_t, size_t>, 1> Kdims{{{1, 0}}};
  double alpha = 1.0, beta = 0.0;

  std::vector<double> cref(M * N, 0.0);
  naive_matmul_2d(M, K, N, alpha, av.data(), K, 1, bv.data(), 1, K, beta, cref.data(), N, 1);

  contract(alpha, A, B, Kdims, beta, C, cpu_tag{});

  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j)
      EXPECT_DOUBLE_EQ((C[i, j]), cref[i * N + j]);
}

// Test: reversed‐stride A
TEST(ContractKernel2D, ReversedA)
{
  size_t M = 2, K = 3, N = 2;
  std::vector<double> av(M * K), bv(K * N), cv(M * N, 0.0);
  std::iota(av.begin(), av.end(), 1.0);
  std::iota(bv.begin(), bv.end(), 10.0);

  // build A as reversed 2×3
  using stdex2 = stdex::dextents<ptrdiff_t, 2>;
  auto mapA = make_mapping<2>(std::array{M, K}, std::array{-static_cast<ptrdiff_t>(K), ptrdiff_t(1)});
  stdex::mdspan<double, stdex2, stdex::layout_stride> A(av.data() + av.size() - 1, mapA);

  auto B = make_mdspan_2d(bv, K, N);
  auto C = make_mdspan_2d(cv, M, N);

  std::array<std::pair<size_t, size_t>, 1> Kdims{{{1, 0}}};

  contract(1.0, A, B, Kdims, 0.0, C, cpu_tag{});

  // reference
  std::vector<double> cref = cv;
  naive_matmul_2d(M, K, N, 1.0, av.data() + av.size() - 1, -static_cast<ptrdiff_t>(K), 1, bv.data(), N, 1, 0.0,
                  cref.data(), N, 1);

  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j)
      EXPECT_DOUBLE_EQ((C[i, j]), cref[i * N + j]);
}

// Test: 3D double contraction → 2D
TEST(ContractKernel3D, DoubleContraction)
{
  constexpr size_t I = 2, Kdim = 2, L = 2, J = 2;
  std::vector<double> va(I * Kdim * L), vb(J * Kdim * L), vc(I * J, 7.0);
  for (size_t i = 0; i < va.size(); ++i)
    va[i] = double(i + 1);
  for (size_t i = 0; i < vb.size(); ++i)
    vb[i] = double(i + 100);

  // row-major A, B: strides {Kdim*L, L, 1}
  auto mapA = make_mapping<3>(std::array{I, Kdim, L},
                              std::array{static_cast<ptrdiff_t>(Kdim * L), static_cast<ptrdiff_t>(L), ptrdiff_t(1)});
  auto mapB = make_mapping<3>(std::array{J, Kdim, L},
                              std::array{static_cast<ptrdiff_t>(Kdim * L), static_cast<ptrdiff_t>(L), ptrdiff_t(1)});
  using stdex3 = stdex::dextents<ptrdiff_t, 3>;
  stdex::mdspan<double, stdex3, stdex::layout_stride> A(va.data(), mapA);
  stdex::mdspan<double, stdex3, stdex::layout_stride> B(vb.data(), mapB);

  auto C = make_mdspan_2d(vc, I, J);
  std::array<std::pair<size_t, size_t>, 2> Kdims{{{1, 1}, {2, 2}}};

  contract(1.0, A, B, Kdims, 0.0, C, cpu_tag{});

  // reference
  std::vector<double> cref = vc;
  for (size_t i = 0; i < I; ++i)
    for (size_t j = 0; j < J; ++j)
    {
      double acc = 0;
      for (size_t k = 0; k < Kdim; ++k)
        for (size_t l = 0; l < L; ++l)
          acc += va[i * Kdim * L + k * L + l] * vb[j * Kdim * L + k * L + l];
      cref[i * J + j] = acc;
    }

  for (size_t i = 0; i < I; ++i)
    for (size_t j = 0; j < J; ++j)
      EXPECT_DOUBLE_EQ((C[i, j]), cref[i * J + j]);
}

// Test: 3D double‐contraction with alpha != 1 and beta != 0
TEST(ContractKernel3D, AlphaBeta)
{
  constexpr size_t I = 2, Kdim = 3, L = 2, J = 2;
  std::vector<double> va(I * Kdim * L), vb(J * Kdim * L), vc(I * J);
  // fill A and B, and C initial
  for (size_t i = 0; i < va.size(); ++i)
    va[i] = double(i + 1);
  for (size_t i = 0; i < vb.size(); ++i)
    vb[i] = double(i + 10);
  std::iota(vc.begin(), vc.end(), 5.0); // initial C = [5,6,7,8]

  // mdspan mappings
  auto A = make_mdspan_3d(va, I, Kdim, L);
  auto B = make_mdspan_3d(vb, J, Kdim, L);
  auto C = make_mdspan_2d(vc, I, J);

  std::array<std::pair<size_t, size_t>, 2> Kdims{{{1, 1}, {2, 2}}};
  double alpha = 3.0, beta = 0.5;

  // build reference: cref = β·C_orig + α·sum_{k,l} A*B
  std::vector<double> cref(I * J);
  for (size_t idx = 0; idx < vc.size(); ++idx)
    cref[idx] = beta * vc[idx];
  for (size_t i = 0; i < I; ++i)
  {
    for (size_t j = 0; j < J; ++j)
    {
      double sum = 0;
      for (size_t k = 0; k < Kdim; ++k)
        for (size_t l = 0; l < L; ++l)
          sum += va[i * Kdim * L + k * L + l] * vb[j * Kdim * L + k * L + l];
      cref[i * J + j] += alpha * sum;
    }
  }

  contract(alpha, A, B, Kdims, beta, C, cpu_tag{});

  for (size_t i = 0; i < I; ++i)
    for (size_t j = 0; j < J; ++j)
      EXPECT_DOUBLE_EQ((C[i, j]), cref[i * J + j]);
}
