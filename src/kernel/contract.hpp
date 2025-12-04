#pragma once

#include "common/mdspan.hpp"
#include "cpu/contract.hpp" // always available fallback
#include "mdspan/strides.hpp"

#ifdef UNI20_BACKEND_BLAS
#include "blas/contract.hpp"
#endif

#ifdef UNI20_BACKEND_MKL
// #include "mkl/contract.hpp"
#endif

#ifdef UNI20_BACKEND_CUDA
// #include "cuda/contract.hpp"
#endif

#include <iostream>
#include <ranges>
#include <vector>

namespace uni20::kernel
{

inline std::ptrdiff_t index_find(const std::vector<size_t>& v, int x)
{
  auto it = std::ranges::find(v, x);
  CHECK(it != v.end());
  TRACE(std::ranges::distance(v.begin(), it));
  return std::ranges::distance(v.begin(), it);
}
inline void veiw_array(std::vector<size_t>& newStrideA)
{
  for (size_t i = 0; i < newStrideA.size(); i++)
    std::cout << newStrideA[i] << " ";
  std::cout << " " << std::endl;
}
template <StridedMdspan AType> char get_blas_trans(AType A)
{
  if (A.stride(0) == 1)
    return 'c';
  else if (A.stride(AType::rank() - 1) == 1)
    return 'r';
  else
    return 'e';
}

inline char get_blas_layout(std::vector<size_t> Strides)
{
  if (Strides.size() != 2) return 'e';
  if (Strides[0] == 1) return 'c';
  if (Strides[1] == 1) return 'r';
  return 'e';
}

template <typename T>
void transposes_loop(size_t step, size_t rank, std::vector<size_t>& indexList, std::vector<size_t>& extentA,
                     std::vector<size_t>& extentB, std::vector<size_t>& strideA, std::vector<size_t>& strideB,
                     T* const input, T* output)
{
  if (step == rank)
  {
    size_t old_index = 0, new_index = 0;
    for (size_t i = 0; i < rank; i++)
    {
      old_index = old_index + indexList[i] * strideA[i];
      new_index = new_index + indexList[i] * strideB[i];
    }
    output[new_index] = input[old_index];
    return;
  }
  for (size_t i = 0; i < extentB[step]; i++)
  {
    indexList[step] = i;
    transposes_loop(step + 1, rank, indexList, extentA, extentB, strideA, strideB, input, output);
  }
}
template <typename T>
void rearrange(T* input, T* output, std::vector<size_t>& oldExtent, std::vector<size_t>& newExtent,
               std::vector<size_t>& oldStride, std::vector<size_t>& newStride)
{
  size_t rank = newStride.size();
  std::vector<size_t> indexList(rank);
  transposes_loop(0, rank, indexList, newExtent, newExtent, oldStride, newStride, input, output);
}

template <int N, int R> size_t extent(static_vector<extent_strides<N>, MR> const& v)
{
  size_t Result = 1;
  for (auto n : v)
    Result *= n.extent;
  return Result;
}

template <typename T, StridedMdspan AType, StridedMdspan BType, std::size_t N, typename U, MutableStridedMdspan CType,
          typename TagType>
  requires(AType::rank() + BType::rank() == CType::rank() + 2 * N)
void contract(T const& alpha, AType A, BType B, std::array<std::pair<std::size_t, std::size_t>, N> const& constractDims,
              U const& beta, CType C, TagType)
{

  // std::vector<size_t> ExtentA(AType::rank()), ExtentB(BType::rank()), ExtentC(CType::rank());
  // std::vector<size_t> newStrideA(AType::rank()), newStrideB(BType::rank()), newStrideC(CType::rank());
  // std::vector<size_t> oldStrideA(AType::rank()), oldStrideB(BType::rank()), oldStrideC(CType::rank());
  // size_t stride_tmpA = A.size(), stride_tmpB = B.size(), stride_tmpC = C.size();
  // for (size_t i = 0; i < AType::rank(); i++)
  // {
  //   stride_tmpA = stride_tmpA / A.extent(i);
  //   newStrideA[i] = stride_tmpA;
  //   oldStrideA[i] = A.stride(i);
  //   ExtentA[i] = A.extent(i);
  // }
  // for (size_t i = 0; i < BType::rank(); i++)
  // {
  //   stride_tmpB = stride_tmpB / B.extent(i);
  //   newStrideB[i] = stride_tmpB;
  //   oldStrideB[i] = B.stride(i);
  //   ExtentB[i] = B.extent(i);
  // }
  // for (size_t i = 0; i < CType::rank(); i++)
  // {
  //   stride_tmpC = stride_tmpC / C.extent(i);
  //   newStrideC[i] = stride_tmpC;
  //   oldStrideC[i] = C.stride(i);
  //   ExtentC[i] = C.extent(i);
  // }
  //
  // auto a = get_blas_trans(A);
  // auto b = get_blas_trans(B);
  // auto c = get_blas_trans(C);

  /*
    auto [flagA,flagB] = transpose_strided(A.data_handle(),B.data_handle(),
                                           ExtentA,ExtentB,
                                           newStrideA,newStrideB,
                                           constractDims,outputA,outputB,a,b, TagType{});
   */
  auto [Mgroup, Ngroup, Kgroup] = extract_strides(A, B, constractDims, C);
  // auto [ap,bp] = constractDims[0];

  /*
  A_{ia jb ld kc}B_{jb qwer ld}
  */

  // reconstruct the strides of the merged A, B, C tensors

  std::vector<size_t> merged_stride_A;
  std::vector<size_t> merged_extent_A;
  for (auto const& s : Mgroup)
  {
    merged_stride_A.push_back(s.strides[0]);
    merged_extent_A.push_back(s.extent);
  }
  for (auto const& s : Kgroup)
  {
    merged_stride_A.push_back(s.strides[0]);
    merged_extent_A.push_back(s.extent);
  }

  std::vector<size_t> merged_stride_B;
  for (auto const& s : Kgroup)
    merged_stride_B.push_back(s.strides[1]);
  for (auto const& s : Ngroup)
    merged_stride_B.push_back(s.strides[0]);

  std::vector<size_t> merged_stride_C;
  for (auto const& s : Mgroup)
    merged_stride_C.push_back(s.strides[1]);
  for (auto const& s : Ngroup)
    merged_stride_C.push_back(s.strides[1]);

  TRACE(A.mapping().strides(), merged_stride_A);
  TRACE(B.mapping().strides(), merged_stride_B);
  TRACE(C.mapping().strides(), merged_stride_C);

  std::shared_ptr<std::vector<T>> tempA, tempB,
      tempC; // shared pointers to manage memory if we need temporary storage for A,B,C
  auto A_handle = A.data_handle();
  auto B_handle = B.data_handle();
  auto C_handle = C.data_handle();

  // Do we need to rearrange the memory of tensor A?
  auto merged_A_layout = blas_trans_layout(merged_stride_A);
  if (merged_A_layout == 'e')
  {
    // yes we do need to rearrange A
    tempA = std::make_shared<std::vector<T>>(A.size());

    // calculate rearranged_stride_A
    std::vector<size_t> rearranged_stride_A;
    size_t ext = A.size();
    for (auto e : merged_extent_A)
    {
      ext = ext / e;
      rearranged_stride_A.push_back(ext);
    }

    rearrange(A.data_handle(), tempA->data(), merged_extent_A, merged_stride_A, rearranged_stride_A);
    A_handle = outputA->data();

    // rearrange:
    // for (i = 0; i < extent(0); ++i)
    // {
    //    for (j = 0; j < extent(1); ++j)
    //    { //
    //       output[i * out.stride(0) + j * out.stride(1) + ...] = input[i * in.stride(0) + j * stride(1) + ...]
    //    }
    // }

    // update merged_A_layout, merged_stride_A, with new rank 2
    // update to rank 2 version  it will be {extent(Kgroup), 1}
    merged_stride_A = {extent(Kgroup), 1};
    merged_extent_A = {extent(Mgroup), extent(Kgroup)};
    merged_A_layout = blas_trans_layout(merged_stride_A);
    A_handle = tempA->data();
  }

  // Do we need to rearrange the memory of tensor B?
  auto merged_B_layout = blas_trans_layout(merged_stride_B);
  // ....

  // Do we need to rearrange the memory of tensor B?
  // This is slightly different, since we allocate memory ONLY, then AFTER the BLAS call we rearrange from temp buffer
  // to C

  // Now we have: (we don't need to reconstruct Mgroup, Ngroup, Kgroup)
  // CHECK_EQUAL(Mgroup.size() == 2);
  // CHECK_EQUAL(Ngroup.size() == 2);
  // CHECK_EQUAL(Kgroup.size() == 2);

  // I recommend:
  gemm(alpha, A_handle, merged_stride_A, B_handle, merged_stride_B, beta, C_handle, merged_stride_C);

  // Alternative:
  gemm(alpha, A_handle, leading_dimension(merged_stride_A), merged_A_layout, B_handle,
       leading_dimension(merged_stride_B), merged_B_layout, beta, C_handle, leading_dimension(merged_stride_C),
       merged_C_layout);

  // if we need to rearrange C, it goes in here...

  return;

  // rest of code is not required....

  std::vector<size_t> stride_tpA, stride_tpB; //(AType::rank()),stride_tpB(BType::rank());

  stride_tmpA = A.size();
  stride_tmpB = B.size();
  for (size_t i = 0; i < Mgroup.size(); i++)
  {
    stride_tmpA = stride_tmpA / Mgroup[i].extent;
    stride_tpA.push_back(stride_tmpA);
    // stride_tpA[i]  = stride_tmpA;
  }
  for (size_t i = 0; i < Kgroup.size(); i++)
  {
    stride_tmpA = stride_tmpA / Kgroup[i].extent;
    stride_tpA.push_back(stride_tmpA);
    // stride_tpA[Mgroup.size() + i]  = stride_tmpA;
    stride_tmpB = stride_tmpB / Kgroup[i].extent;
    stride_tpB.push_back(stride_tmpB);
    // stride_tpB[i]  = stride_tmpB;
  }
  for (size_t i = 0; i < Ngroup.size(); i++)
  {
    stride_tmpB = stride_tmpB / Ngroup[i].extent;
    stride_tpB.push_back(stride_tmpB);
    // stride_tpB[Kgroup.size() + i]  = stride_tmpB;
  }
  veiw_array(newStrideA);
  veiw_array(newStrideB);
  std::cout << "XXXXXX" << std::endl;
  veiw_array(stride_tpA);
  veiw_array(stride_tpB);
  std::cout << "ExtentA: ";
  veiw_array(ExtentA);
  std::cout << "ExtentB: ";
  veiw_array(ExtentB);

  for (size_t i = 0; i < Mgroup.size(); i++)
    std::cout << "Mgroup extent:" << Mgroup[i].extent << ",stride:{" << Mgroup[i].strides[0] << ","
              << Mgroup[i].strides[1] << "}" << std::endl;
  for (size_t i = 0; i < Ngroup.size(); i++)
    std::cout << "Ngroup extent:" << Ngroup[i].extent << ",stride:{" << Ngroup[i].strides[0] << ","
              << Ngroup[i].strides[1] << "}" << std::endl;
  for (size_t i = 0; i < Kgroup.size(); i++)
    std::cout << "Kgroup extent:" << Kgroup[i].extent << ",stride:{" << Kgroup[i].strides[0] << ","
              << Kgroup[i].strides[1] << "}" << std::endl;
  std::vector<size_t> Astrides(stride_tpA.size()), Bstrides(stride_tpB.size());
  // auto Astrides = stride_tpA;
  // auto Bstrides = stride_tpB;
  for (size_t i = 0; i < Mgroup.size(); i++)
    Astrides[index_find(oldStrideA, Mgroup[i].strides[0])] = stride_tpA[i];
  for (size_t i = 0; i < Kgroup.size(); i++)
    Astrides[index_find(oldStrideA, Kgroup[i].strides[0])] = stride_tpA[Mgroup.size() + i];
  for (size_t i = 0; i < Kgroup.size(); i++)
  {
    TRACE(Kgroup.size(), Bstrides.size(), oldStrideB, Kgroup[i].strides[1],
          index_find(oldStrideB, Kgroup[i].strides[1]));
    Bstrides[index_find(oldStrideB, Kgroup[i].strides[1])] = stride_tpB[i];
  }
  for (size_t i = 0; i < Ngroup.size(); i++)
    Bstrides[index_find(oldStrideB, Ngroup[i].strides[0])] = stride_tpB[Kgroup.size() + i];
  veiw_array(Astrides);
  veiw_array(Bstrides);
  std::cout << "XXXXXXXXX" << std::endl;
  bool flagA = rearrange_flag(Astrides, stride_tpA, TagType{});
  bool flagB = rearrange_flag(Bstrides, stride_tpB, TagType{});

  // std::vector<T> outputA(A.size()), outputB(B.size()), ToutputA(A.size()), ToutputB(B.size()), ToutputC(C.size());

  std::shared_ptr<std::vector<T>> outputA, outputB;

  auto A_handle = A.data_handle();
  if (flagA)
  {
    outputA = std::make_shared<std::vector<T>>(A.size());
    rearrange(A.data_handle(), outputA->data(), ExtentA, ExtentA, oldStrideA, Astrides);
    A_handle = outputA->data();
  }

  auto B_handle = B.data_handle();
  if (flagB)
  {
    outputB = std::make_shared<std::vector<T>>(B.size());
    rearrange(B.data_handle(), outputB->data(), ExtentB, ExtentB, oldStrideB, Bstrides);
    B_handle = outputB->data();
  }

  contract_strided(a, b, c, Mgroup, Ngroup, Kgroup, alpha, A_handle, B_handle, beta, C.data_handle(), TagType{});

  //
  //
  // if (flagB) rearrange(B.data_handle(), outputB.data(), ExtentB, ExtentB, oldStrideB, Bstrides);
  // // rearrange(C.data_handle(),ToutputC.data(),ExtentC,ExtentC,oldStrideC,newStrideC);
  //
  // if (flagA == false && flagB == false)
  //   contract_strided(a, b, c, Mgroup, Ngroup, Kgroup, alpha, A.data_handle(), B.data_handle(), beta, C.data_handle(),
  //                    TagType{});
  // if (flagA == false && flagB == true)
  //   contract_strided(a, b, c, Mgroup, Ngroup, Kgroup, alpha, A.data_handle(), outputB.data(), beta, C.data_handle(),
  //                    TagType{});
  // if (flagA == true && flagB == false)
  //   contract_strided(a, b, c, Mgroup, Ngroup, Kgroup, alpha, outputA.data(), B.data_handle(), beta, C.data_handle(),
  //                    TagType{});
  // if (flagA == true && flagB == true)
  //   contract_strided(a, b, c, Mgroup, Ngroup, Kgroup, alpha, outputA.data(), outputB.data(), beta, C.data_handle(),
  //                    TagType{});
  // rearrange(ToutputC.data(),C.data_handle(),ExtentC,ExtentC,newStrideC,oldStrideC);
  std::cout << "flagM :" << Mgroup.size() << "flagN :" << Ngroup.size() << "flagK :" << Kgroup.size() << std::endl;
}
} // namespace uni20::kernel
  // namespace uni20::kernel

/*
 A_{ijkl}B_{jmnk}--->A'_{iljk}B'{jkmn}
(i,j,k,l) = (0001),(0002),(0003),(0010)...
 A'[LJK*i+JK*l+K*j+k]------
  A[i*KJL+j*KL+k*L+l]
  A'[LJK*i+K*j+k+JK*l]
  */
