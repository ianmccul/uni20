#pragma once

#include "common/mdspan.hpp"
#include "blas.hpp"
#include "mdspan/strides.hpp"
#include "backend/blas/backend_blas.hpp"
#include <cblas.h>
#include <iostream>

namespace uni20::kernel
{

    bool rearrange_flag(std::vector<size_t>& old_stride,std::vector<size_t>& new_stride,blas_tag){
        return (old_stride != new_stride);
    }
    template <typename T>
    void transpose_loop(std::vector<size_t>& permutation,size_t step,size_t rank,std::vector<size_t>& indexList,
                        std::vector<size_t>& old_extent,std::vector<size_t>& new_extent,
                        std::vector<size_t>& old_stride,std::vector<size_t>& new_stride,
                        std::vector<T>& data,T const* old_data)
    {
      if(step == rank){
        size_t old_index = 0,new_index = 0;
        for(size_t i =0;i<old_stride.size();i++){
            old_index = old_index + indexList[i]*old_stride[i];
            new_index = new_index + indexList[i]*new_stride[permutation[i]];
        }
        data[new_index] = old_data[old_index];
        return;
      }
      for(size_t i=0; i < old_extent[step]; i++){
          indexList[step] = i;
          transpose_loop(permutation,step+1, rank,indexList, old_extent,new_extent, old_stride,new_stride,data,old_data);
      }
    }
    template <typename T, std::size_t N>
    std::pair<bool, bool> transpose_strided(
      T const* A,T const* B,
      std::vector<size_t> oldExtentA, std::vector<size_t> oldExtentB,
      std::vector<size_t> oldStrideA, std::vector<size_t> oldStrideB,
      std::array<std::pair<std::size_t, std::size_t>, N> const& contractDims,
      std::vector<T>& outputA,std::vector<T>& outputB,char modea,char modeb,blas_tag)//, T const* In, std::span<std::ptrdiff_t> InStrides, T* Out,                            //std::span<std::ptrdiff_t> OutStrides)
    {
      std::vector<size_t> newExtentA(oldExtentA.size()),newExtentB(oldExtentB.size());
      std::vector<size_t> newStrideA(oldExtentA.size()),newStrideB(oldExtentB.size());
      std::vector<size_t> tmpA,tmpB;
      for(size_t i=0;i<oldExtentA.size();i++) newExtentA[i] = oldExtentA[i];
      for(size_t i=0;i<oldExtentB.size();i++) newExtentB[i] = oldExtentB[i];
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
      for(size_t i=0;i<oldExtentA.size();i++){
        if(modea == 'r'){
          newStrideA[oldExtentA.size()-1-i] = stride_step;
          stride_step = stride_step * newExtentA[oldExtentA.size()-1-i];
        }else{
          newStrideA[i] = stride_step;
          stride_step = stride_step * newExtentA[i];
        }
      }
      stride_step = 1;
      for(size_t i=0;i<oldExtentB.size();i++){
        if(modeb == 'r'){
          newStrideB[oldExtentB.size()-1-i] = stride_step;
          stride_step = stride_step * newExtentB[oldExtentB.size()-1-i];
        }else{
          newStrideB[i] = stride_step;
          stride_step = stride_step * newExtentB[i];
        }
      }
      std::vector<size_t> indexListA(oldExtentA.size()),permutationA(oldExtentA.size());
      std::vector<size_t> indexListB(oldExtentB.size()),permutationB(oldExtentB.size());
      std::vector<size_t> mova,movb,ord,ore;
      bool flagA = 0,flagB = 0;
    
      for(size_t i=0;i<N;i++){
        auto [a,b] = contractDims[i];
        mova.push_back(a);
        movb.push_back(b);
      }
      for(size_t i = 0; i < oldExtentA.size(); ++i)
          if (std::find(mova.begin(), mova.end(), i) == mova.end())
              ord.push_back(i);
      ord.insert(ord.end(), mova.begin(), mova.end());
      for(size_t i = 0; i < ord.size(); ++i)   permutationA[ord[i]] = i;
      for(size_t i = 0; i < permutationA.size();i++) if(permutationA[i] != i) flagA = true;
      if(flagA == true) transpose_loop(permutationA,0,oldExtentA.size(),indexListA,
                                       oldExtentA,newExtentA,
                                       oldStrideA,newStrideA,
                                       outputA,A);
      for (size_t i = 0; i < oldExtentB.size(); ++i)
        if (std::find(movb.begin(), movb.end(), i) == movb.end())
            ore.push_back(i);
      ore.insert(ore.end(), movb.begin(), movb.end());
      for(size_t i = 0; i < ore.size(); ++i)   permutationB[ore[i]] = i;
      for(size_t i = 0;i < permutationB.size();i++) if(permutationB[i] != i) flagB = true;
      if(flagB == true) transpose_loop(permutationB,0,oldExtentB.size(),indexListB,
                                       oldExtentB,newExtentB,
                                       oldStrideB,newStrideB
                                       ,outputB,B);
      return std::pair{flagA, flagB};
}


template <typename T, std::size_t MR, std::size_t NR, std::size_t KR>
void contract_strided(char a, char b, char c,
                      static_vector<extent_strides<2>, MR> const& Mgrp,
                      static_vector<extent_strides<2>, NR> const& Ngrp,
                      static_vector<extent_strides<2>, KR> const& Kgrp, 
                      T alpha, T const* A, T const* B, T beta, T* C,
                      blas_tag)
{
    size_t K = 1, M = 1, N = 1;
    for(size_t i = 0; i < Kgrp.size(); i++) K = K * Kgrp[i].extent;
    for(size_t i = 0; i < Mgrp.size(); i++) M = M * Mgrp[i].extent;
    for(size_t i = 0; i < Ngrp.size(); i++) N = N * Ngrp[i].extent;
    std::cout<<" M = "<<M<<" N = "<<N<<" K = "<<K<<std::endl;
    if ((a == 'r') && (b == 'r') && (c == 'r')){
      if(N==1)            uni20::blas::gemv('T', K, M, alpha, A, K, B, 1, beta, C, 1);
      else if(M==1)       uni20::blas::gemv('T', K, N, alpha, B, K, A, 1, beta, C, 1);
      else                uni20::blas::gemm('N', 'N', N, M, K, alpha, B, N, A, K, beta, C, N);
      //else uni20::blas::gemm('T', 'N', N, M, K, alpha, B, K, A, K, beta, C, N);
      //else uni20::blas::gemm('T', 'N', N, M, K, alpha, B, K, A, K, beta, C, N);
    }else if((a == 'r') && (b == 'r') && (c == 'c')){
      if(N==1)            uni20::blas::gemv('T', K, M, alpha, A, K, B, 1, beta, C, 1);
      else if(M==1)       uni20::blas::gemv('T', K, N, alpha, B, K, A, 1, beta, C, 1);
      else                uni20::blas::gemm('T', 'T', M, N, K, alpha, A, K, B, N, beta, C, M);//uni20::blas::gemm('T', 'N', M, N, K, alpha, A, K, B, K, beta, C, M);

//////////
    }else if((a == 'r') && (b == 'c') && (c == 'r')){
      if(N==1)            uni20::blas::gemv('T', K, M, alpha, A, K, B, 1, beta, C, 1);
      else if(M==1)       uni20::blas::gemv('T', K, N, alpha, B, K, A, 1, beta, C, 1);
      else                uni20::blas::gemm('T', 'N', N, M, K, alpha, B, K, A, K, beta, C, N);//uni20::blas::gemm('N', 'N', N, M, K, alpha, B, N, A, K, beta, C, N);
    }else if((a == 'r') && (b == 'c') && (c == 'c')){
      if(N==1)            uni20::blas::gemv('T', K, M, alpha, A, K, B, 1, beta, C, 1);
      else if(M==1)       uni20::blas::gemv('T', K, N, alpha, B, K, A, 1, beta, C, 1);
      else                uni20::blas::gemm('T', 'N', M, N, K, alpha, A, K, B, K, beta, C, M);//uni20::blas::gemm('T', 'T', M, N, K, alpha, A, K, B, N, beta, C, M);
//////////


    }else if((a == 'c') && (b == 'r') && (c == 'r')){
      if(N==1)            uni20::blas::gemv('T', K, M, alpha, A, K, B, 1, beta, C, 1);
      else if(M==1)       uni20::blas::gemv('T', K, N, alpha, B, K, A, 1, beta, C, 1);
      else                uni20::blas::gemm('N', 'T', N, M, K, alpha, B, N, A, M, beta, C, N);//uni20::blas::gemm('T', 'T', N, M, K, alpha, B, K, A, M, beta, C, N);
    }else if((a == 'c') && (b == 'r') && (c == 'c')){
      if(N==1)            uni20::blas::gemv('T', K, M, alpha, A, K, B, 1, beta, C, 1);
      else if(M==1)       uni20::blas::gemv('T', K, N, alpha, B, K, A, 1, beta, C, 1);
      else                uni20::blas::gemm('N', 'T', M, N, K, alpha, A, M, B, N, beta, C, M);//uni20::blas::gemm('N', 'N', M, N, K, alpha, A, M, B, K, beta, C, M);

    ////////  
    }else if((a == 'c') && (b == 'c') && (c == 'r')){
      if(N==1)            uni20::blas::gemv('T', K, M, alpha, A, K, B, 1, beta, C, 1);
      else if(M==1)       uni20::blas::gemv('T', K, N, alpha, B, K, A, 1, beta, C, 1);
      else                uni20::blas::gemm('T', 'T', N, M, K, alpha, B, K, A, M, beta, C, N);//uni20::blas::gemm('N', 'T', N, M, K, alpha, B, N, A, M, beta, C, N);
     
    }else if((a == 'c') && (b == 'c') && (c == 'c')){
      if(N==1)            uni20::blas::gemv('T', K, M, alpha, A, K, B, 1, beta, C, 1);
      else if(M==1)       uni20::blas::gemv('T', K, N, alpha, B, K, A, 1, beta, C, 1);
      else                uni20::blas::gemm('N', 'N', M, N, K, alpha, A, M, B, K, beta, C, M);//uni20::blas::gemm('N', 'T', M, N, K, alpha, A, M, B, N, beta, C, M);
       
    }else{}
    /////////
}
//"XXXXXXLLXXPPXXXX"
} 
