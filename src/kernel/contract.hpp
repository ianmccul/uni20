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
template <typename T, StridedMdspan AType, StridedMdspan BType, std::size_t N, typename U, MutableStridedMdspan CType,
          typename TagType>
  requires(AType::rank() + BType::rank() == CType::rank() + 2 * N)
void contract(T const& alpha, AType A, BType B, std::array<std::pair<std::size_t, std::size_t>, N> const& constractDims,
              U const& beta, CType C, TagType)
{
  size_t sizeA = 1,sizeB = 1;
  for(size_t i=0;i<AType::rank();i++) sizeA = sizeA * A.extent(i);
  for(size_t i=0;i<BType::rank();i++) sizeB = sizeB * B.extent(i);
  std::vector<T> outputA(sizeA),outputB(sizeB);
  auto [flagA,flagB] = transpose_strided(A,B,constractDims,outputA,outputB, TagType{});
  std::cout<<"transpose success!"<<std::endl;
  auto [Mgroup, Ngroup, Kgroup] = extract_strides(A, B, constractDims, C);
  std::cout<<"extract success!"<<std::endl;
  if(flagA == false && flagB == false)  contract_strided(Mgroup, Ngroup, Kgroup, alpha, A.data_handle(), B.data_handle(), beta, C.data_handle(), TagType{});
  if(flagA == false && flagB ==  true)  contract_strided(Mgroup, Ngroup, Kgroup, alpha, A.data_handle(), outputB.data(), beta, C.data_handle(), TagType{});
  if(flagA ==  true && flagB == false)  contract_strided(Mgroup, Ngroup, Kgroup, alpha, outputA.data(), B.data_handle(), beta, C.data_handle(), TagType{});
  if(flagA ==  true && flagB ==  true)  contract_strided(Mgroup, Ngroup, Kgroup, alpha, outputA.data(), outputB.data(), beta, C.data_handle(), TagType{});
  // contract_strided(Mgroup, Ngroup, Kgroup, alpha, A.data_handle(), B.data_handle(), beta, C.data_handle(), TagType{});
}

} // namespace uni20::kernel
