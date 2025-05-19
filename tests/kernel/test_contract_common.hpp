// test_contract_common.hpp
#pragma once

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

namespace uni20::test {

template<typename Scalar, typename Tag>
void test_rank2_contraction_correctness() {
  typedef Scalar INPUTTYPE;
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
  
  contract(alpha, A, B, Kdims, beta, C_cpu, cpu_tag{});
  contract(alpha, A, B, Kdims, beta, C_blas, Tag{});

  INPUTTYPE standard = 0;
  for(size_t i=0; i<dimC;i++) standard = standard + (C_blas.data_handle()[i]-C_cpu.data_handle()[i])*(C_blas.data_handle()[i]-C_cpu.data_handle()[i]);
  std::cout<<"standard cpu-blas: "<<standard<<" val:"<<C_blas.data_handle()[0]<<std::endl;
}
}
