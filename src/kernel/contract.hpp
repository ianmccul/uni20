#pragma once

#include "common/mdspan.hpp"
#include "mdspan/strides.hpp"
#include "cpu/contract.hpp" // always available fallback

#ifdef UNI20_BACKEND_BLAS
#include "blas/contract.hpp"
#endif

#ifdef UNI20_BACKEND_MKL
//#include "mkl/contract.hpp"
#endif

#ifdef UNI20_BACKEND_CUDA
//#include "cuda/contract.hpp"
#endif

#include <iostream>
#include <vector>
namespace uni20::kernel
{

    template <typename T>
    void transposes_loop(size_t step,size_t rank,std::vector<size_t>& indexList,
                        std::vector<size_t>& extentA,std::vector<size_t>& extentB,
                        std::vector<size_t>& strideA,std::vector<size_t>& strideB,
                        T* const input,T* output)
    {
      if(step == rank){
        size_t old_index = 0,new_index = 0;
        for(size_t i =0;i<rank;i++){
            old_index = old_index + indexList[i]*strideA[i];
            new_index = new_index + indexList[i]*strideB[i];
        }
        output[new_index] = input[old_index];
        return;
      }
      for(size_t i=0; i < extentB[step]; i++){
          indexList[step] = i;
          transposes_loop(step+1, rank,indexList, extentA,extentB, strideA,strideB,input,output);
      }
    }
    template <typename T>
    void rearrange(
      T*  input,
      T* output,
      std::vector<size_t>& oldExtent, 
      std::vector<size_t>& newExtent,
      std::vector<size_t>& oldStride,
      std::vector<size_t>& newStride
    )
    {
      size_t rank = newStride.size();
      std::vector<size_t> indexList(rank);
      transposes_loop(0,rank,indexList,newExtent,newExtent,oldStride,newStride,input,output);
}




template <typename T, StridedMdspan AType, StridedMdspan BType, std::size_t N, typename U, MutableStridedMdspan CType,
          typename TagType>
  requires(AType::rank() + BType::rank() == CType::rank() + 2 * N)
void contract(T const& alpha, AType A, BType B, std::array<std::pair<std::size_t, std::size_t>, N> const& constractDims,
              U const& beta, CType C, TagType)
{
  std::vector<size_t> ExtentA(AType::rank()),ExtentB(BType::rank()),ExtentC(CType::rank());
  std::vector<size_t> newStrideA(AType::rank()),newStrideB(BType::rank()),newStrideC(CType::rank());
  std::vector<size_t> oldStrideA(AType::rank()),oldStrideB(BType::rank()),oldStrideC(CType::rank());
  size_t stride_tmpA = A.size(),stride_tmpB = B.size(),stride_tmpC = C.size();
  for(size_t i=0 ; i < AType::rank() ; i++){
        stride_tmpA = stride_tmpA/A.extent(i);
        newStrideA[i]  = stride_tmpA;
        oldStrideA[i]  = A.stride(i);
        ExtentA[i]     = A.extent(i);
      }
  for(size_t i=0 ; i < BType::rank() ; i++){
        stride_tmpB = stride_tmpB/B.extent(i);
        newStrideB[i]  = stride_tmpB;
        oldStrideB[i]  = B.stride(i);
        ExtentB[i]     = B.extent(i);
      }
  for(size_t i=0 ; i < CType::rank() ; i++){
        stride_tmpC = stride_tmpC/C.extent(i);
        newStrideC[i]  = stride_tmpC;
        oldStrideC[i]  = C.stride(i);
        ExtentC[i]     = C.extent(i);
      }
  std::vector<T> outputA(A.size()),outputB(B.size()),ToutputA(A.size()),ToutputB(B.size()),ToutputC(C.size());
  char a,b,c;
  
  //rearrange(A.data_handle(),ToutputA.data(),ExtentA,ExtentA,oldStrideA,newStrideA);
  //rearrange(B.data_handle(),ToutputB.data(),ExtentB,ExtentB,oldStrideB,newStrideB);
  //rearrange(C.data_handle(),ToutputC.data(),ExtentC,ExtentC,oldStrideC,newStrideC);    
  if(newStrideA == oldStrideA) a = 'r';       
  else a = 'c';
  if(newStrideB == oldStrideB) b = 'r';       
  else b = 'c';
  if(newStrideC == oldStrideC) c= 'r';       
  else c = 'c';
  

  auto [flagA,flagB] = transpose_strided(A.data_handle(),B.data_handle(),
                                         ExtentA,ExtentB,
                                         newStrideA,newStrideB,
                                         constractDims,outputA,outputB,a,b, TagType{});
  auto [Mgroup, Ngroup, Kgroup] = extract_strides(A, B, constractDims, C);
  if(flagA == false && flagB == false)  contract_strided(a,b,c,Mgroup, Ngroup, Kgroup, alpha, A.data_handle(), B.data_handle(), beta, C.data_handle(), TagType{});
  if(flagA == false && flagB ==  true)  contract_strided(a,b,c,Mgroup, Ngroup, Kgroup, alpha, A.data_handle(), outputB.data(), beta, C.data_handle(), TagType{});
  if(flagA ==  true && flagB == false)  contract_strided(a,b,c,Mgroup, Ngroup, Kgroup, alpha, outputA.data(), B.data_handle(), beta, C.data_handle(), TagType{});
  if(flagA ==  true && flagB ==  true)  contract_strided(a,b,c,Mgroup, Ngroup, Kgroup, alpha, outputA.data(), outputB.data(), beta, C.data_handle(), TagType{});
  //rearrange(ToutputC.data(),C.data_handle(),ExtentC,ExtentC,newStrideC,oldStrideC);
    }
  }
 // namespace uni20::kernel
