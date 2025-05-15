#include "../helpers.hpp"
#include "kernel/contract.hpp"
#include "gtest/gtest.h"
#include <numeric>
#include <cblas.h>
#include <random>
#include <unordered_map>
#include <string>
#include <vector>
#include "kernel/blas/blas.hpp"

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

  for(int m=0;m<int(M);m++)
    for(int n=0;n<int(N);n++){
        cref[m*N+n] = beta * cref[m*N+n];
      for(int k=0;k<int(K);k++)
        cref[m*N+n] = cref[m*N+n] + alpha*av[m*K+k]*bv[N*k+n];
    }
          

  //naive_matmul_2d(M, K, N, alpha, av.data(), K, 1, bv.data(), N, 1, beta, cref.data(), N, 1);
  
  for(int n=0 ;n<int(N);n++){
    for(int m=0 ;m<int(M);m++)
      std::cout<<cref.data()[m*N+n]<<" ";
    std::cout<<""<<std::endl;
  }
  // **NEW** kernel call:
  contract(alpha, A, B, Kdims, beta, C, blas_tag{});
  for(int n=0 ;n<int(N);n++){
    for(int m=0 ;m<int(M);m++)
      std::cout<<C[m,n]<<" ";
    std::cout<<""<<std::endl;
  }
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
/*
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
*/
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

  contract(1.0, A, B, Kdims, 0.0, C, blas_tag{});

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

TEST(Disdhh, rank2_TTGT) {
  std::cout<<"TEST disdhh 2 dim matirx:"<<std::endl;
  std::random_device rd;                
  std::mt19937 gen(rd());               
  std::uniform_int_distribution<> idis(2, 10);  
  size_t M = idis(gen), K = idis(gen), N = idis(gen);
  std::vector<double> av(M * K), bv(K * N), cv(M * N),cb(M * N);
  std::iota(av.begin(), av.end(), 1.0);
  std::iota(bv.begin(), bv.end(), 10.0);
  std::fill(cv.begin(), cv.end(), 5.0);
  std::fill(cb.begin(), cb.end(), 5.0);

  auto A = make_mdspan_2d(av, K, M);
  auto B = make_mdspan_2d(bv, K, N);
  auto C_cpu = make_mdspan_2d(cv, M, N);
  auto C_blas = make_mdspan_2d(cb, M, N);
  // contract A.dim1↔B.dim0
  std::array<std::pair<size_t, size_t>, 1> Kdims{{{0, 0}}};
  double alpha = 2.0, beta = 0.5;
  contract(alpha, A, B, Kdims, beta, C_blas, blas_tag{});
  contract(alpha, A, B, Kdims, beta, C_cpu, cpu_tag{});
  double standard = 0;
  for(int i=0 ;i<int(M);i++)
    for(int j=0 ;j<int(N);j++)
      standard = standard + (C_blas[i,j]-C_cpu[i,j])*(C_blas[i,j]-C_cpu[i,j]);
  std::cout<<"standard cpu-blas 2 dim tensor: "<<standard<<std::endl;
}




TEST(Disdhh, rank3_TTGT) {
  std::cout<<"TEST disdhh 3 dim matirx:"<<std::endl;
  std::random_device rd;                
  std::mt19937 gen(rd());               
  std::uniform_int_distribution<> idis(2, 10);  
  size_t I = idis(gen), Kdim = idis(gen), L = idis(gen), J = idis(gen);
  std::vector<double> va( Kdim* I * L), vb(J * Kdim * L), v_cpu(I * J, 7.0),v_blas(I * J, 7.0);
  for (size_t i = 0; i < va.size(); ++i)  va[i] = double(i + 1)/100;
  for (size_t i = 0; i < vb.size(); ++i)  vb[i] = double(i + 100)/100;
  double alpha = 0.7;
  double beta  = 0.3;
  // row-major A, B: strides {Kdim*L, L, 1}
  auto mapA = make_mapping<3>(std::array{L, Kdim, I},std::array{static_cast<ptrdiff_t>(Kdim * I), static_cast<ptrdiff_t>(I), ptrdiff_t(1)});
  auto mapB = make_mapping<3>(std::array{Kdim, J, L},std::array{static_cast<ptrdiff_t>(J * L), static_cast<ptrdiff_t>(L), ptrdiff_t(1)});
  using stdex3 = stdex::dextents<ptrdiff_t, 3>;
  stdex::mdspan<double, stdex3, stdex::layout_stride> A(va.data(), mapA);
  stdex::mdspan<double, stdex3, stdex::layout_stride> B(vb.data(), mapB);

  auto C_cpu  = make_mdspan_2d(v_cpu , I, J);
  auto C_blas = make_mdspan_2d(v_blas, I, J);
  std::array<std::pair<size_t, size_t>, 2> Kdims{{{1, 0}, {0, 2}}};
      
  contract(alpha, A, B, Kdims, beta, C_cpu, cpu_tag{});
  contract(alpha, A, B, Kdims, beta, C_blas, blas_tag{});

  // computing standard
  
  double standard = 0;
  for(int i=0 ;i<int(I);i++)
    for(int j=0 ;j<int(J);j++)
      standard = standard + (C_blas[i,j]-C_cpu[i,j])*(C_blas[i,j]-C_cpu[i,j]);
  std::cout<<"standard cpu-blas 3 dim tensor:: "<<standard<<std::endl;
}

TEST(Disdhh, rank4_TTGT) {
  std::cout<<"TEST disdhh 4 dim matirx:"<<std::endl;
  std::random_device rd;                
  std::mt19937 gen(rd());               
  std::uniform_int_distribution<> idis(2, 10);  
  std::uniform_real_distribution<> fdis(0,1);
  std::unordered_map<std::string, size_t> hash_table;


  hash_table["I"] = idis(gen);
  hash_table["K"] = idis(gen);
  hash_table["L"] = idis(gen);
  hash_table["J"] = idis(gen);
  hash_table["M"] = idis(gen);
  size_t dimA = hash_table["K"]*hash_table["I"]*hash_table["L"]*hash_table["M"];
  size_t dimB = hash_table["J"]*hash_table["K"]*hash_table["L"]*hash_table["M"];
  size_t I = hash_table["I"], Kdim = hash_table["K"], L = hash_table["L"], J = hash_table["J"], M = hash_table["M"];//, N = idis(gen);
  std::cout<<I<<" "<<Kdim<<" "<<L<<" "<<J<<" "<<M<<" "<<Kdim*L*M<<std::endl;
  std::vector<double> va(dimA), vb(dimB), v_cpu(I * J, 7.0),v_blas(I * J, 7.0);
  for (size_t i = 0; i < va.size(); ++i)  va[i] = fdis(gen);
  for (size_t i = 0; i < vb.size(); ++i)  vb[i] = fdis(gen);
  double alpha = 0.7,beta  = 0.3;

  std::vector<std::string> arrangeA = {"K","I","L","M"};
  std::vector<std::string> arrangeB = {"J","K","L","M"};
  std::shuffle(arrangeA.begin(), arrangeA.end(), gen);
  std::shuffle(arrangeB.begin(), arrangeB.end(), gen);
  std::cout<<arrangeA[0]<<arrangeA[1]<<arrangeA[2]<<arrangeA[3]<<std::endl;
  std::cout<<arrangeB[0]<<arrangeB[1]<<arrangeB[2]<<arrangeB[3]<<std::endl;
  std::vector<size_t> extentA,extentB,strideA,strideB;
  for(size_t i=0;i<arrangeA.size();i++) extentA.push_back(hash_table[arrangeA[i]]);
  for(size_t i=0;i<arrangeB.size();i++) extentB.push_back(hash_table[arrangeB[i]]);
  size_t stride_tmp = dimA;
  for(size_t i=0;i<arrangeA.size();i++){
    stride_tmp = stride_tmp/extentA[i];
    strideA.push_back(stride_tmp);
  }
  stride_tmp = dimB;
  for(size_t i=0;i<arrangeB.size();i++){
    stride_tmp = stride_tmp/extentB[i];
    strideB.push_back(stride_tmp);
  }
  for(size_t i=0;i<arrangeB.size();i++) extentB.push_back(hash_table[arrangeB[i]]);
  std::array<size_t,4> extentA_array = {0,0,0,0};
  std::array<size_t,4> extentB_array = {0,0,0,0};
  std::array<size_t,4> strideA_array = {0,0,0,0};
  std::array<size_t,4> strideB_array = {0,0,0,0};
  for(size_t i=0;i<4;i++) extentA_array[i] = extentA[i];
  for(size_t i=0;i<4;i++) extentB_array[i] = extentB[i];
  for(size_t i=0;i<4;i++) strideA_array[i] = strideA[i];
  for(size_t i=0;i<4;i++) strideB_array[i] = strideB[i];
  auto mapA = make_mapping<4>(extentA_array,
    std::array{static_cast<ptrdiff_t>(strideA[0]),static_cast<ptrdiff_t>(strideA[1]),static_cast<ptrdiff_t>(strideA[2]),static_cast<ptrdiff_t>(strideA[3])});
  auto mapB = make_mapping<4>(extentB_array,
    std::array{static_cast<ptrdiff_t>(strideB[0]),static_cast<ptrdiff_t>(strideB[1]),static_cast<ptrdiff_t>(strideB[2]),static_cast<ptrdiff_t>(strideB[3])});
    
  using stdex3 = stdex::dextents<ptrdiff_t, 4>;
  stdex::mdspan<double, stdex3, stdex::layout_stride> A(va.data(), mapA);
  stdex::mdspan<double, stdex3, stdex::layout_stride> B(vb.data(), mapB);


  auto C_cpu  = make_mdspan_2d(v_cpu , I, J);
  auto C_blas = make_mdspan_2d(v_blas, I, J);

  std::array<std::pair<size_t, size_t>, 3> Kdims;
  size_t p = 0;
  for(size_t i=0;i<4;i++)
    for(size_t j=0;j<4;j++)
      if(arrangeA[i] == arrangeB[j]){
        Kdims[p].first = i;
        Kdims[p].second = j;
        p = p + 1;
      }
  std::cout<<"Kdims:"<<Kdims[0].second<<std::endl;
  contract(alpha, A, B, Kdims, beta, C_cpu, cpu_tag{});
  contract(alpha, A, B, Kdims, beta, C_blas, blas_tag{});

  // computing standard
  
  double standard = 0;
  for(int i=0 ;i<int(I);i++)
    for(int j=0 ;j<int(J);j++)
      standard = standard + (C_blas[i,j]-C_cpu[i,j])*(C_blas[i,j]-C_cpu[i,j]);
  std::cout<<"standard cpu-blas 4 dim tensor:: "<<standard<<std::endl;
}




TEST(Disdhh, rank4_TTGT_GEMV) {
  typedef complex128 INPUTTYPE;
  std::cout<<"TEST disdhh 4 dim matirx gemv:"<<std::endl;
  std::random_device rd;                
  std::mt19937 gen(rd());               
  std::uniform_int_distribution<> idis(2, 10);  
  std::uniform_real_distribution<> fdis(0,1);
  std::unordered_map<std::string, size_t> hash_table;
  hash_table["I"] = idis(gen);
  hash_table["K"] = idis(gen);
  hash_table["L"] = idis(gen);
  hash_table["J"] = idis(gen);
  hash_table["M"] = idis(gen);
  constexpr size_t rankA = 2,rankB = 4,rankC = 2;
  std::vector<std::string> arrangeA = {"K","L"};
  std::vector<std::string> arrangeB = {"L","K","I","J"};
  std::vector<std::string> arrangeC = {"I","J"};
  size_t dimA = 1,dimB = 1,dimC = 1;
  for(size_t i=0; i<rankA;i++)  dimA = dimA * hash_table[arrangeA[i]];
  for(size_t i=0; i<rankB;i++)  dimB = dimB * hash_table[arrangeB[i]];
  for(size_t i=0; i<rankC;i++)  dimC = dimC * hash_table[arrangeC[i]];
  std::vector<INPUTTYPE> va(dimA), vb(dimB), v_cpu(dimC, 7.0),v_blas(dimC, 7.0);
  for (size_t i = 0; i < va.size(); ++i)  va[i] = fdis(gen);
  for (size_t i = 0; i < vb.size(); ++i)  vb[i] = fdis(gen);
  INPUTTYPE alpha = 0.7,beta  = 0.3;
  std::shuffle(arrangeA.begin(), arrangeA.end(), gen);
  std::shuffle(arrangeB.begin(), arrangeB.end(), gen);
  std::vector<size_t> extentA,extentB,extentC,strideA,strideB,strideC;
  for(size_t i=0;i<arrangeA.size();i++) extentA.push_back(hash_table[arrangeA[i]]);
  for(size_t i=0;i<arrangeB.size();i++) extentB.push_back(hash_table[arrangeB[i]]);
  for(size_t i=0;i<arrangeC.size();i++) extentB.push_back(hash_table[arrangeC[i]]);
  std::array<size_t,rankA> extentA_array;
  std::array<size_t,rankB> extentB_array;
  std::array<size_t,rankC> extentC_array;
  
  std::array<std::pair<size_t, size_t>, (rankA+rankB-rankC)/2> Kdims;
  size_t p = 0;
  for(size_t i=0;i<rankA;i++)
    for(size_t j=0;j<rankB;j++)
      if(arrangeA[i] == arrangeB[j]){
        Kdims[p].first = i;
        Kdims[p].second = j;
        p = p + 1;
      }
  std::vector<size_t> tmp; 
  for(size_t i=0;i<rankA;i++)
    for(size_t j=0;j<rankC;j++)
      if(arrangeA[i] == arrangeC[j])
        tmp.push_back(extentA[i]);
  for(size_t i=0;i<rankB;i++)
    for(size_t j=0;j<rankC;j++)
      if(arrangeB[i] == arrangeC[j])
        tmp.push_back(extentB[i]);
  
      
  for(size_t i=0;i<rankA;i++) extentA_array[i] = extentA[i];
  for(size_t i=0;i<rankB;i++) extentB_array[i] = extentB[i];
  for(size_t i=0;i<rankC;i++) extentC_array[i] = tmp[i];
  
  size_t stride_tmp = dimA;
  for(size_t i=0;i<arrangeA.size();i++){
    stride_tmp = stride_tmp/extentA[i];
    strideA.push_back(stride_tmp);
  }
  stride_tmp = dimB;
  for(size_t i=0;i<arrangeB.size();i++){
    stride_tmp = stride_tmp/extentB[i];
    strideB.push_back(stride_tmp);
  }
  stride_tmp = dimC;
  for(size_t i=0;i<arrangeC.size();i++){
    stride_tmp = stride_tmp/tmp[i];
    strideC.push_back(stride_tmp);
  }
  std::array<ptrdiff_t,rankA> strideA_array;
  std::array<ptrdiff_t,rankB> strideB_array;
  std::array<ptrdiff_t,rankC> strideC_array;
  for(size_t i=0;i<arrangeA.size();i++) strideA_array[i] = static_cast<ptrdiff_t>(strideA[i]);
  for(size_t i=0;i<arrangeB.size();i++) strideB_array[i] = static_cast<ptrdiff_t>(strideB[i]);
  for(size_t i=0;i<arrangeC.size();i++) strideC_array[i] = static_cast<ptrdiff_t>(strideC[i]);
  auto mapA = make_mapping<rankA>(extentA_array,strideA_array);
  auto mapB = make_mapping<rankB>(extentB_array,strideB_array);
  auto mapC = make_mapping<rankC>(extentC_array,strideC_array);
  
  using stdexA = stdex::dextents<ptrdiff_t, rankA>;
  using stdexB = stdex::dextents<ptrdiff_t, rankB>;
  using stdexC = stdex::dextents<ptrdiff_t, rankC>;
  stdex::mdspan<INPUTTYPE, stdexA, stdex::layout_stride> A(va.data(), mapA);
  stdex::mdspan<INPUTTYPE, stdexB, stdex::layout_stride> B(vb.data(), mapB);
  stdex::mdspan<INPUTTYPE, stdexC, stdex::layout_stride> C_cpu(v_cpu.data(), mapC);
  stdex::mdspan<INPUTTYPE, stdexC, stdex::layout_stride> C_blas(v_blas.data(), mapC);
  

  //auto C_cpu  = make_mdspan_2d(v_cpu , I, J);
  //auto C_blas = make_mdspan_2d(v_blas, I, J);

  
  contract(alpha, A, B, Kdims, beta, C_cpu, cpu_tag{});
  contract(alpha, A, B, Kdims, beta, C_blas, blas_tag{});

  // computing standard
  
  INPUTTYPE standard = 0;
  for(size_t i=0; i<dimC;i++) standard = standard + (C_blas.data_handle()[i]-C_cpu.data_handle()[i])*(C_blas.data_handle()[i]-C_cpu.data_handle()[i]);
  std::cout<<"standard cpu-blas: "<<standard<<" val:"<<C_blas.data_handle()[0]<<std::endl;
}
