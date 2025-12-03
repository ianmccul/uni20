// test_contract_common.hpp
#pragma once

#include "../helpers.hpp"
#include "kernel/blas/blas.hpp"
#include "kernel/contract.hpp"
#include "common/mdspan.hpp"
#include "gtest/gtest.h"
#include <cblas.h>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "../helpers.hpp"


using namespace uni20;
using uni20::cpu_tag; // our tag‐dispatch for now

using namespace uni20::kernel;

template <typename T> void matmul_col_major(int M, int K, int N, T alpha, T beta, T* A, T* B, T* C)
{
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < K; k++)
        C[i + j * M] = alpha * A[i + k * M] * B[k + j * K] + beta * C[i + j * M]; // col-major
}

template <typename T> void matmul_row_major(int M, int K, int N, T alpha, T beta, T* A, T* B, T* C)
{
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < K; ++k)
        C[i * N + j] = alpha * A[i * K + k] * B[k * N + j] + beta * C[i * N + j];
}

namespace uni20::test
{

template <typename Scalar, typename Tag> void test_rank2_contraction_correctness()
{
  typedef Scalar INPUTTYPE;
  std::cout << "TEST disdhh 4 dim matirx gemv:" << std::endl;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> idis(2, 10);
  std::uniform_real_distribution<> fdis(0, 1);
  std::unordered_map<std::string, size_t> hash_table;
  hash_table["I"] = 4;//idis(gen);
  hash_table["K"] = 5;//idis(gen);
  hash_table["L"] = 6;//idis(gen);
  hash_table["J"] = 7;//idis(gen);
  hash_table["M"] = 8;//idis(gen);
  constexpr size_t rankA = 3, rankB = 4, rankC = 3;
  std::vector<std::string> arrangeA = {"M","K", "L"};
  std::vector<std::string> arrangeB = {"L", "I","K", "J"};
  std::vector<std::string> arrangeC = {"M","I", "J"};
  size_t dimA = 1, dimB = 1, dimC = 1;
  for (size_t i = 0; i < rankA; i++)
    dimA = dimA * hash_table[arrangeA[i]];
  for (size_t i = 0; i < rankB; i++)
    dimB = dimB * hash_table[arrangeB[i]];
  for (size_t i = 0; i < rankC; i++)
    dimC = dimC * hash_table[arrangeC[i]];
  std::vector<INPUTTYPE> va(dimA), vb(dimB), v_cpu(dimC, 0.0), v_blas(dimC, 0.0);
  for (size_t i = 0; i < va.size(); ++i)
    va[i] = fdis(gen);
  for (size_t i = 0; i < vb.size(); ++i)
    vb[i] = fdis(gen);
  INPUTTYPE alpha = 0.7, beta = 0.3;
  std::shuffle(arrangeA.begin(), arrangeA.end(), gen);
  std::shuffle(arrangeB.begin(), arrangeB.end(), gen);
  std::vector<size_t> extentA, extentB, extentC, strideA, strideB, strideC;
  for (size_t i = 0; i < arrangeA.size(); i++)
    extentA.push_back(hash_table[arrangeA[i]]);
  for (size_t i = 0; i < arrangeB.size(); i++)
    extentB.push_back(hash_table[arrangeB[i]]);
  for (size_t i = 0; i < arrangeC.size(); i++)
    extentB.push_back(hash_table[arrangeC[i]]);
  std::array<size_t, rankA> extentA_array;
  std::array<size_t, rankB> extentB_array;
  std::array<size_t, rankC> extentC_array;

  std::array<std::pair<size_t, size_t>, (rankA + rankB - rankC) / 2> Kdims;
  size_t p = 0;
  for (size_t i = 0; i < rankA; i++)
    for (size_t j = 0; j < rankB; j++)
      if (arrangeA[i] == arrangeB[j])
      {
        Kdims[p].first = i;
        Kdims[p].second = j;
        p = p + 1;
      }
  std::vector<size_t> tmp;
  for (size_t i = 0; i < rankA; i++)
    for (size_t j = 0; j < rankC; j++)
      if (arrangeA[i] == arrangeC[j]) tmp.push_back(extentA[i]);
  for (size_t i = 0; i < rankB; i++)
    for (size_t j = 0; j < rankC; j++)
      if (arrangeB[i] == arrangeC[j]) tmp.push_back(extentB[i]);

  for (size_t i = 0; i < rankA; i++)
    extentA_array[i] = extentA[i];
  for (size_t i = 0; i < rankB; i++)
    extentB_array[i] = extentB[i];
  for (size_t i = 0; i < rankC; i++)
    extentC_array[i] = tmp[i];

  size_t stride_tmp = dimA;
  for (size_t i = 0; i < arrangeA.size(); i++)
  {
    stride_tmp = stride_tmp / extentA[i];
    strideA.push_back(stride_tmp);
  }
  stride_tmp = dimB;
  for (size_t i = 0; i < arrangeB.size(); i++)
  {
    stride_tmp = stride_tmp / extentB[i];
    strideB.push_back(stride_tmp);
  }
  stride_tmp = dimC;
  for (size_t i = 0; i < arrangeC.size(); i++)
  {
    stride_tmp = stride_tmp / tmp[i];
    strideC.push_back(stride_tmp);
  }
  std::array<ptrdiff_t, rankA> strideA_array;
  std::array<ptrdiff_t, rankB> strideB_array;
  std::array<ptrdiff_t, rankC> strideC_array;
  for (size_t i = 0; i < arrangeA.size(); i++)
    strideA_array[i] = static_cast<ptrdiff_t>(strideA[i]);
  for (size_t i = 0; i < arrangeB.size(); i++)
    strideB_array[i] = static_cast<ptrdiff_t>(strideB[i]);
  for (size_t i = 0; i < arrangeC.size(); i++)
    strideC_array[i] = static_cast<ptrdiff_t>(strideC[i]);
  auto mapA = make_mapping<rankA>(extentA_array, strideA_array);
  auto mapB = make_mapping<rankB>(extentB_array, strideB_array);
  auto mapC = make_mapping<rankC>(extentC_array, strideC_array);

  using stdexA = stdex::dextents<ptrdiff_t, rankA>;
  using stdexB = stdex::dextents<ptrdiff_t, rankB>;
  using stdexC = stdex::dextents<ptrdiff_t, rankC>;
  stdex::mdspan<INPUTTYPE, stdexA, stdex::layout_stride> A(va.data(), mapA);
  stdex::mdspan<INPUTTYPE, stdexB, stdex::layout_stride> B(vb.data(), mapB);
  stdex::mdspan<INPUTTYPE, stdexC, stdex::layout_stride> C_cpu(v_cpu.data(), mapC);
  stdex::mdspan<INPUTTYPE, stdexC, stdex::layout_stride> C_blas(v_blas.data(), mapC);

  contract(alpha, A, B, Kdims, beta, C_cpu, cpu_tag{});
  contract(alpha, A, B, Kdims, beta, C_blas, Tag{});
  std::cout<<"SDSDSD"<<std::endl;
  INPUTTYPE standard = 0;
  for (size_t i = 0; i < dimC; i++)
    standard = standard +
               (C_blas.data_handle()[i] - C_cpu.data_handle()[i]) * (C_blas.data_handle()[i] - C_cpu.data_handle()[i]);
  std::cout << "standard cpu-blas: " << standard << " val:" << C_blas.data_handle()[0] << std::endl;
}

template <typename Scalar, typename Tag> void test_colunm_major_matmul_correctness()
{
  typedef Scalar INPUTTYPE;
  std::cout << "TEST disdhh 4 dim matirx gemv:" << std::endl;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> idis(2, 10);
  std::uniform_real_distribution<> fdis(0, 1);
  std::unordered_map<std::string, size_t> hash_table;
  hash_table["I"] = idis(gen);
  hash_table["K"] = idis(gen);
  hash_table["L"] = idis(gen);
  hash_table["J"] = idis(gen);
  hash_table["M"] = idis(gen);
  constexpr size_t rankA = 2, rankB = 2, rankC = 2;
  std::vector<std::string> arrangeA = {"K", "L"};
  std::vector<std::string> arrangeB = {"L", "J"};
  std::vector<std::string> arrangeC = {"K", "J"};
  size_t dimA = 1, dimB = 1, dimC = 1;
  for (size_t i = 0; i < rankA; i++)
    dimA = dimA * hash_table[arrangeA[i]];
  for (size_t i = 0; i < rankB; i++)
    dimB = dimB * hash_table[arrangeB[i]];
  for (size_t i = 0; i < rankC; i++)
    dimC = dimC * hash_table[arrangeC[i]];
  std::vector<INPUTTYPE> va(dimA), vb(dimB), vc(dimC), v_cpu(dimC, 7.0), v_blas(dimC, 7.0);
  for (size_t i = 0; i < va.size(); ++i)
    va[i] = fdis(gen);
  for (size_t i = 0; i < vb.size(); ++i)
    vb[i] = fdis(gen);
  for (size_t i = 0; i < vc.size(); ++i)
    vc[i] = fdis(gen);
  for (size_t i = 0; i < v_cpu.size(); ++i)
    v_cpu[i] = vc[i];
  for (size_t i = 0; i < v_blas.size(); ++i)
    v_blas[i] = vc[i];

  INPUTTYPE alpha = 1.0, beta = 1.0;
  std::vector<size_t> extentA, extentB, extentC, strideA, strideB, strideC;
  for (size_t i = 0; i < arrangeA.size(); i++)
    extentA.push_back(hash_table[arrangeA[i]]);
  for (size_t i = 0; i < arrangeB.size(); i++)
    extentB.push_back(hash_table[arrangeB[i]]);
  for (size_t i = 0; i < arrangeC.size(); i++)
    extentC.push_back(hash_table[arrangeC[i]]);
  std::array<size_t, rankA> extentA_array;
  std::array<size_t, rankB> extentB_array;
  std::array<size_t, rankC> extentC_array;

  std::array<std::pair<size_t, size_t>, (rankA + rankB - rankC) / 2> Kdims;
  size_t p = 0;
  for (size_t i = 0; i < rankA; i++)
    for (size_t j = 0; j < rankB; j++)
      if (arrangeA[i] == arrangeB[j])
      {
        Kdims[p].first = i;
        Kdims[p].second = j;
        p = p + 1;
      }
  std::vector<size_t> tmp;
  for (size_t i = 0; i < rankA; i++)
    for (size_t j = 0; j < rankC; j++)
      if (arrangeA[i] == arrangeC[j]) tmp.push_back(extentA[i]);
  for (size_t i = 0; i < rankB; i++)
    for (size_t j = 0; j < rankC; j++)
      if (arrangeB[i] == arrangeC[j]) tmp.push_back(extentB[i]);

  for (size_t i = 0; i < rankA; i++)
    extentA_array[i] = extentA[i];
  for (size_t i = 0; i < rankB; i++)
    extentB_array[i] = extentB[i];
  for (size_t i = 0; i < rankC; i++)
    extentC_array[i] = tmp[i];
  size_t stride_tmp = 1;
  for (size_t i = 0; i < arrangeA.size(); i++)
  {
    strideA.push_back(stride_tmp);
    stride_tmp = stride_tmp * extentA[i];
  }
  stride_tmp = 1;
  for (size_t i = 0; i < arrangeB.size(); i++)
  {
    strideB.push_back(stride_tmp);
    stride_tmp = stride_tmp * extentB[i];
  }
  stride_tmp = 1;
  for (size_t i = 0; i < arrangeC.size(); i++)
  {
    strideC.push_back(stride_tmp);
    stride_tmp = stride_tmp * tmp[i];
  }

  std::array<ptrdiff_t, rankA> strideA_array;
  std::array<ptrdiff_t, rankB> strideB_array;
  std::array<ptrdiff_t, rankC> strideC_array;
  for (size_t i = 0; i < arrangeA.size(); i++)
    strideA_array[i] = static_cast<ptrdiff_t>(strideA[i]);
  for (size_t i = 0; i < arrangeB.size(); i++)
    strideB_array[i] = static_cast<ptrdiff_t>(strideB[i]);
  for (size_t i = 0; i < arrangeC.size(); i++)
    strideC_array[i] = static_cast<ptrdiff_t>(strideC[i]);
  auto mapA = make_mapping<rankA>(extentA_array, strideA_array);
  auto mapB = make_mapping<rankB>(extentB_array, strideB_array);
  auto mapC = make_mapping<rankC>(extentC_array, strideC_array);

  using stdexA = stdex::dextents<ptrdiff_t, rankA>;
  using stdexB = stdex::dextents<ptrdiff_t, rankB>;
  using stdexC = stdex::dextents<ptrdiff_t, rankC>;
  stdex::mdspan<INPUTTYPE, stdexA, stdex::layout_stride> A(va.data(), mapA);
  stdex::mdspan<INPUTTYPE, stdexB, stdex::layout_stride> B(vb.data(), mapB);
  stdex::mdspan<INPUTTYPE, stdexC, stdex::layout_stride> C_cpu(v_cpu.data(), mapC);
  stdex::mdspan<INPUTTYPE, stdexC, stdex::layout_stride> C_blas(v_blas.data(), mapC);

  contract(alpha, A, B, Kdims, beta, C_cpu, cpu_tag{});
  contract(alpha, A, B, Kdims, beta, C_blas, Tag{});
  matmul_col_major(hash_table["K"], hash_table["L"], hash_table["J"], alpha, beta, va.data(), vb.data(), vc.data());
  INPUTTYPE standard_blas = 0, standard_cpu = 0;
  for (size_t i = 0; i < dimC; i++)
    EXPECT_NEAR(vc.data()[i], C_blas.data_handle()[i], 1e-6);
  for (size_t i = 0; i < dimC; i++)
    EXPECT_NEAR(vc.data()[i], C_cpu.data_handle()[i], 1e-6);

  std::vector<size_t> strideA_row, strideB_row, strideC_row;
  stride_tmp = dimA;
  for (size_t i = 0; i < arrangeA.size(); i++)
  {
    stride_tmp = stride_tmp / extentA[i];
    strideA_row.push_back(stride_tmp);
  }
  stride_tmp = dimB;
  for (size_t i = 0; i < arrangeB.size(); i++)
  {
    stride_tmp = stride_tmp / extentB[i];
    strideB_row.push_back(stride_tmp);
  }
  stride_tmp = dimC;
  for (size_t i = 0; i < arrangeC.size(); i++)
  {
    stride_tmp = stride_tmp / tmp[i];
    strideC_row.push_back(stride_tmp);
  }

  for (size_t i = 0; i < va.size(); ++i)
    va[i] = fdis(gen);
  for (size_t i = 0; i < vb.size(); ++i)
    vb[i] = fdis(gen);
  for (size_t i = 0; i < vc.size(); ++i)
    vc[i] = fdis(gen);
  for (size_t i = 0; i < v_cpu.size(); ++i)
    v_cpu[i] = vc[i];
  for (size_t i = 0; i < v_blas.size(); ++i)
    v_blas[i] = vc[i];

  std::array<ptrdiff_t, rankA> strideA_array_row;
  std::array<ptrdiff_t, rankB> strideB_array_row;
  std::array<ptrdiff_t, rankC> strideC_array_row;
  for (size_t i = 0; i < arrangeA.size(); i++)
    strideA_array_row[i] = static_cast<ptrdiff_t>(strideA_row[i]);
  for (size_t i = 0; i < arrangeB.size(); i++)
    strideB_array_row[i] = static_cast<ptrdiff_t>(strideB_row[i]);
  for (size_t i = 0; i < arrangeC.size(); i++)
    strideC_array_row[i] = static_cast<ptrdiff_t>(strideC_row[i]);
  auto mapA_row = make_mapping<rankA>(extentA_array, strideA_array_row);
  auto mapB_row = make_mapping<rankB>(extentB_array, strideB_array_row);
  auto mapC_row = make_mapping<rankC>(extentC_array, strideC_array_row);

  stdex::mdspan<INPUTTYPE, stdexA, stdex::layout_stride> A_row(va.data(), mapA_row);
  stdex::mdspan<INPUTTYPE, stdexB, stdex::layout_stride> B_row(vb.data(), mapB_row);
  stdex::mdspan<INPUTTYPE, stdexC, stdex::layout_stride> C_cpu_row(v_cpu.data(), mapC_row);
  stdex::mdspan<INPUTTYPE, stdexC, stdex::layout_stride> C_blas_row(v_blas.data(), mapC_row);

  contract(alpha, A_row, B_row, Kdims, beta, C_cpu_row, cpu_tag{});
  contract(alpha, A_row, B_row, Kdims, beta, C_blas_row, Tag{});
  matmul_row_major(hash_table["K"], hash_table["L"], hash_table["J"], alpha, beta, va.data(), vb.data(), vc.data());
  standard_blas = 0, standard_cpu = 0;
  for (size_t i = 0; i < dimC; i++)
    standard_blas =
        standard_blas + (C_blas_row.data_handle()[i] - vc.data()[i]) * (C_blas_row.data_handle()[i] - vc.data()[i]);
  for (size_t i = 0; i < dimC; i++)
    standard_cpu =
        standard_cpu + (vc.data()[i] - C_cpu_row.data_handle()[i]) * (vc.data()[i] - C_cpu_row.data_handle()[i]);
  std::cout << "standard cpu-matmul-row-majpr: " << standard_cpu << "   "
            << "standard blas-matmul-row-majpr: " << standard_blas << std::endl;

  std::vector<INPUTTYPE> va_rearrange(24), vb_rearrange(24), vc_rearrange(24);
  std::vector<size_t> new_extent_rearrange = {3, 2, 4};
  std::vector<size_t> new_stride_rearrange = {8, 4, 1};
  std::vector<size_t> old_extent_rearrange = {3, 2, 4};
  std::vector<size_t> old_stride_rearrange = {1, 3, 6};
  for (size_t i = 0; i < 24; ++i)
    va_rearrange[i] = i;

  std::cout << "unrearrange:" << std::endl;
  for (size_t i = 0; i < va_rearrange.size(); i++)
    std::cout << va_rearrange.data()[i] << " ";
  std::cout << "\nrearrange 1:" << std::endl;
  rearrange(va_rearrange.data(), vb_rearrange.data(), old_extent_rearrange, new_extent_rearrange, old_stride_rearrange,
            new_stride_rearrange);
  for (size_t i = 0; i < vb_rearrange.size(); i++)
    std::cout << vb_rearrange.data()[i] << " ";
  std::cout << "\nrearrange 2:" << std::endl;
  rearrange(vb_rearrange.data(), vc_rearrange.data(), new_extent_rearrange, old_extent_rearrange, new_stride_rearrange,
            old_stride_rearrange);
  for (size_t i = 0; i < vc_rearrange.size(); i++)
    std::cout << vc_rearrange.data()[i] << " ";
}








template <typename Scalar, typename Tag> void test_colunm_major_matmul_12345678_correctness()
{
  typedef Scalar INPUTTYPE;
  INPUTTYPE alpha = 1,beta = 1;
  size_t M = 2, K = 3, N = 4;
  constexpr size_t rankA = 2,rankB = 2,rankC = 2;
  std::vector<INPUTTYPE> av(M * K), bv(K * N), cv(M * N),c_blas(M*N),c_cpu(M*N),c_v(M*N);
  for(size_t i=0;i<M*K;i++)  av[i] = i+1;
  for(size_t i=0;i<K*N;i++)  bv[i] = M*K+i+1;
  for(size_t i=0;i<c_blas.size();i++)  c_cpu[i] = c_blas[i] =c_v[i]= 0;
  auto mapA = make_mapping<2>(std::array{M,K},std::array{static_cast<ptrdiff_t>(K), ptrdiff_t(1)});
  auto mapB = make_mapping<2>(std::array{K,N},std::array{static_cast<ptrdiff_t>(N), ptrdiff_t(1)});
  auto mapC = make_mapping<2>(std::array{M,N},std::array{static_cast<ptrdiff_t>(N), ptrdiff_t(1)});
  for(int j = 0;j<8;j++){

   if(j%2>=1) mapC = make_mapping<2>(std::array{M,N},std::array{static_cast<ptrdiff_t>(1), ptrdiff_t(M)});
   else mapC = make_mapping<2>(std::array{M,N},std::array{static_cast<ptrdiff_t>(N), ptrdiff_t(1)});

   if(j%4>=2) mapB = make_mapping<2>(std::array{K,N},std::array{static_cast<ptrdiff_t>(1), ptrdiff_t(K)});
   else mapB = make_mapping<2>(std::array{K,N},std::array{static_cast<ptrdiff_t>(N), ptrdiff_t(1)});
   
   if(j%8>=4) mapA = make_mapping<2>(std::array{M,K},std::array{static_cast<ptrdiff_t>(1), ptrdiff_t(M)});
   else mapA = make_mapping<2>(std::array{M,K},std::array{static_cast<ptrdiff_t>(K), ptrdiff_t(1)});
    


    std::array<std::pair<size_t, size_t>, 1> Kdims{{{1, 0}}};
    using stdexA = stdex::dextents<ptrdiff_t, rankA>;
    using stdexB = stdex::dextents<ptrdiff_t, rankB>;
    using stdexC = stdex::dextents<ptrdiff_t, rankC>;
    stdex::mdspan<INPUTTYPE, stdexA, stdex::layout_stride> A(av.data(), mapA);
    stdex::mdspan<INPUTTYPE, stdexB, stdex::layout_stride> B(bv.data(), mapB);
    stdex::mdspan<INPUTTYPE, stdexC, stdex::layout_stride> C_cpu(c_cpu.data(), mapC);
    stdex::mdspan<INPUTTYPE, stdexC, stdex::layout_stride> C_blas(c_blas.data(), mapC);
    contract(alpha, A, B, Kdims, beta, C_cpu, cpu_tag{});
    contract(alpha, A, B, Kdims, beta, C_blas, Tag{});
    std::cout<<j<<"  blas: ";for(size_t i=0;i<c_blas.size();i++)  std::cout<<c_blas[i]<<" "; std::cout<<""<<std::endl;
    std::cout<<j<<"  cpu : ";for(size_t i=0;i<c_blas.size();i++)  std::cout<<c_cpu[i]<<" "; std::cout<<""<<std::endl;
    for(size_t i=0;i<M*K;i++)  av[i] = i+1;
    for(size_t i=0;i<K*N;i++)  bv[i] = M*K+i+1;
    for(size_t i=0;i<c_blas.size();i++)  c_cpu[i] = c_blas[i] =c_v[i]= 0;
    std::cout<<"----------------------------------------"<<std::endl;
  }
  
  
  

} // namespace uni20::test
}

/*
***********************************************************************************
A = [0,1,2,3,4,5]
B = [0,1,2,3,4,5,6,7]
extent A = [3,2]  extent B = [2,4]  extent C = [3,4]
stride A = [1,3]  stride B = [1,2]  stride C = [1,3]
A@B = C = [3,5,7,12,18,24,21,31,41,30,44,58]
[0 3    [0 2 4 6     [0 3 6  9     [3 12 21 30
 1 4  @  1 3 5 7] +   1 4 7 10   =  5 18 31 44
 2 5]                 2 5 8 11]     7 24 41 58]

my  result = [3, 5, 7,12,18,24,21,31,41,30,44,58]
you result = [8,25,18, 7,38,33,16,19,37,33,19,51]
my result and your result is diffrence i don't know what is true
i use rearrange function(not transport because transport must inverse extent).
this rearrange function can transfer any extent/stride to any extent/stride
**********************************************************************************



*/
