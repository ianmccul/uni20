#pragma once

#include "common/mdspan.hpp"
#include <array>
#include <vector>

// Convenience typedefs
using index_t = std::ptrdiff_t;

/// \brief  Helper to construct a mapping (extents + strides)
template <std::size_t N, typename index_type>
auto make_mapping(std::array<std::size_t, N> const& extents, std::array<index_type, N> const& strides)
{
  using extents_t = stdex::dextents<index_type, N>;
  using mapping_t = stdex::layout_stride::mapping<extents_t>;

  return mapping_t(extents_t(extents), strides);
}

/// \brief  A helper to build a 1D mdspan over a std::vector using layout_stride
inline auto make_mdspan_1d(std::vector<double>& v)
{
  using extents_t = stdex::dextents<index_t, 1>;
  std::array<std::ptrdiff_t, 1> strides{1};
  auto mapping = stdex::layout_stride::mapping<extents_t>(extents_t{v.size()}, strides);
  return stdex::mdspan<double, extents_t, stdex::layout_stride>(v.data(), mapping);
}

/// \brief  A helper to build a 2D row‐major mdspan using layout_stride
inline auto make_mdspan_2d(std::vector<double>& v, std::size_t R, std::size_t C)
{
  using extents_t = stdex::dextents<index_t, 2>;
  std::array<std::ptrdiff_t, 2> strides{static_cast<index_t>(C), 1}; // Row-major
  auto mapping = stdex::layout_stride::mapping<extents_t>(extents_t{R, C}, strides);
  return stdex::mdspan<double, extents_t, stdex::layout_stride>(v.data(), mapping);
}

inline auto make_mdspan_2d(std::vector<double>& v, std::size_t R, std::size_t C, std::array<std::ptrdiff_t, 2> strides)
{
  using extents_t = stdex::dextents<index_t, 2>;
  auto mapping = stdex::layout_stride::mapping<extents_t>(extents_t{R, C}, strides);
  return stdex::mdspan<double, extents_t, stdex::layout_stride>(v.data(), mapping);
}

/// \brief  A helper to build a reversed‐1D mdspan (negative stride)
inline auto make_reversed_1d(std::vector<double>& v)
{
  using extents_t = stdex::dextents<index_t, 1>;
  std::array<std::ptrdiff_t, 1> strides{-1};
  auto mapping = stdex::layout_stride::mapping<extents_t>(extents_t{v.size()}, strides);
  return stdex::mdspan<double, extents_t, stdex::layout_stride>(v.data() + v.size() - 1, mapping);
}

/// \brief  Build a 3D row-major mdspan with layout_stride
inline auto make_mdspan_3d(std::vector<double>& v, std::size_t R, std::size_t K, std::size_t L)
{
  using extents_t = stdex::dextents<index_t, 3>;
  // row-major: stride[0]=K*L, stride[1]=L, stride[2]=1
  std::array<index_t, 3> strides{static_cast<index_t>(K * L), static_cast<index_t>(L), 1};
  auto m = stdex::layout_stride::mapping<extents_t>(extents_t{R, K, L}, strides);
  return stdex::mdspan<double, extents_t, stdex::layout_stride>(v.data(), m);
}
