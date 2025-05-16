#pragma once

#include "common/mdspan.hpp"
#include "blas.hpp"
#include "mdspan/strides.hpp"
#include "backend/blas/backend_blas.hpp"
#include <cblas.h>
#include <iostream>

namespace uni20::kernel
{
    template <typename T,StridedMdspan AType>
    void transpose_loop(std::vector<size_t>& permutation,size_t step,size_t rank,std::vector<size_t>& indexList,
                        std::vector<size_t>& extentA,std::vector<size_t>& extentB,
                        std::vector<size_t>& strideA,std::vector<size_t>& strideB,
                        AType const& A,std::vector<T>& dataA)
    {
      if(step == rank){
        size_t old_index = 0,new_index = 0;
        for(size_t i =0;i<strideA.size();i++){
            old_index = old_index + indexList[i]*strideA[i];
            new_index = new_index + indexList[i]*strideB[permutation[i]];
        }
        dataA[new_index] = A.data_handle()[old_index];
        return;
      }
      for(size_t i=0; i < extentA[step]; i++){
          indexList[step] = i;
          transpose_loop(permutation,step+1, rank,indexList, extentA,extentB, strideA,strideB,A,dataA);
      }
    }
    template <typename T,StridedMdspan AType,StridedMdspan BType, std::size_t N>
    std::pair<bool, bool> transpose_strided(AType const& A,BType const& B,std::array<std::pair<std::size_t, std::size_t>, N> const& contractDims,std::vector<T>& outputA,std::vector<T>& outputB,blas_tag)//, T const* In, std::span<std::ptrdiff_t> InStrides, T* Out,                            //std::span<std::ptrdiff_t> OutStrides)
    {
      std::vector<size_t> newExtentA(AType::rank()),oldExtentA(AType::rank()),newExtentB(BType::rank()),oldExtentB(BType::rank());
      std::vector<size_t> newStrideA(AType::rank()),oldStrideA(AType::rank()),newStrideB(BType::rank()),oldStrideB(BType::rank());
      std::vector<size_t> tmpA,tmpB;
      for(size_t i=0;i<AType::rank();i++){
        newExtentA[i] = A.extent(i);
        oldExtentA[i] = A.extent(i);
        oldStrideA[i] = A.stride(i);
      }
      for(size_t i=0;i<BType::rank();i++){
        newExtentB[i] = B.extent(i);
        oldExtentB[i] = B.extent(i);
        oldStrideB[i] = B.stride(i);
      }
      for(size_t i=0;i<N;i++){
        auto [a,b] = contractDims[i];
        tmpA.push_back(oldExtentA[a]);
        tmpB.push_back(oldExtentB[b]);
        newExtentA[a] = 0;
        newExtentB[b] = 0;
      }
      newExtentA.erase(std::remove(newExtentA.begin(), newExtentA.end(), 0), newExtentA.end());
      newExtentB.erase(std::remove(newExtentB.begin(), newExtentB.end(), 0), newExtentB.end());
      newExtentA.insert(newExtentA.end(), tmpA.begin(), tmpA.end());
      newExtentB.insert(newExtentB.end(), tmpB.begin(), tmpB.end());
      size_t stride_step = 1;
      for(size_t i=0;i<AType::rank();i++){
        newStrideA[AType::rank()-1-i] = stride_step;
        stride_step = stride_step * newExtentA[AType::rank()-1-i];
      }
      stride_step = 1;
      for(size_t i=0;i<BType::rank();i++){
        newStrideB[BType::rank()-1-i] = stride_step;
        stride_step = stride_step * newExtentB[BType::rank()-1-i];
      }
      std::vector<size_t> indexListA(AType::rank()),permutationA(AType::rank());
      std::vector<size_t> indexListB(BType::rank()),permutationB(BType::rank());
      std::vector<size_t> mova,movb,ord,ore;
      bool flagA = 0,flagB = 0;
    
      for(size_t i=0;i<N;i++){
        auto [a,b] = contractDims[i];
        mova.push_back(a);
        movb.push_back(b);
      }
      for(size_t i = 0; i < AType::rank(); ++i)
          if (std::find(mova.begin(), mova.end(), i) == mova.end())
              ord.push_back(i);
      ord.insert(ord.end(), mova.begin(), mova.end());
      for(size_t i = 0; i < ord.size(); ++i)   permutationA[ord[i]] = i;
      for(size_t i = 0; i < permutationA.size();i++) if(permutationA[i] != i) flagA = true;
      if(flagA == true) transpose_loop(permutationA,0,AType::rank(),indexListA,oldExtentA,newExtentA,oldStrideA,newStrideA,A,outputA);
      for (size_t i = 0; i < BType::rank(); ++i)
        if (std::find(movb.begin(), movb.end(), i) == movb.end())
            ore.push_back(i);
      ore.insert(ore.end(), movb.begin(), movb.end());
      for(size_t i = 0; i < ore.size(); ++i)   permutationB[ore[i]] = i;
      for(size_t i = 0;i < permutationB.size();i++) if(permutationB[i] != i) flagB = true;
      if(flagB == true) transpose_loop(permutationB,0,BType::rank(),indexListB,oldExtentB,newExtentB,oldStrideB,newStrideB,B,outputB);
      return std::pair{flagA, flagB};
}


template <typename T, std::size_t MR, std::size_t NR, std::size_t KR>
void contract_strided(static_vector<extent_strides<2>, MR> const& Mgrp,
                      static_vector<extent_strides<2>, NR> const& Ngrp,
                      static_vector<extent_strides<2>, KR> const& Kgrp, T alpha, T const* A, T const* B, T beta, T* C,
                      blas_tag)
{
    size_t K = 1, M = 1, N = 1;
    for(size_t i = 0; i < Kgrp.size(); i++) K = K * Kgrp[i].extent;
    for(size_t i = 0; i < Mgrp.size(); i++) M = M * Mgrp[i].extent;
    for(size_t i = 0; i < Ngrp.size(); i++) N = N * Ngrp[i].extent;
    if(N==1)            uni20::blas::gemv('T', K, M, alpha, A, K, B, 1, beta, C, 1);
    else if(M==1)       uni20::blas::gemv('T', K, N, alpha, B, K, A, 1, beta, C, 1);
    else                uni20::blas::gemm('T', 'N', N, M, K, alpha, B, K, A, K, beta, C, N);
}
//"XXXXXXXXPPXXXX"
} 
