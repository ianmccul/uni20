#pragma once

#include "common/mdspan.hpp"
#include "common/static_vector.hpp"
#include "common/trace.hpp"
#include "concepts.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <array>

namespace uni20
{

/// \brief Structure to represent a common extent and an array of strides
template <std::size_t N> struct extent_strides
{
    index_type extent;
    std::array<std::ptrdiff_t, N> strides;

    constexpr extent_strides() = default;

    /// \brief Construct from extent and a list of strides.
    template <typename Ext>
      requires std::is_integral_v<Ext>
    constexpr extent_strides(Ext e, std::initializer_list<std::ptrdiff_t> s) : extent(static_cast<index_type>(e))
    {
      assert(s.size() == N);
      std::copy(s.begin(), s.end(), strides.begin());
    }

    /// \brief Construct from extent and an array of strides.
    template <typename Ext, typename Str>
      requires std::is_integral_v<Ext> && std::is_integral_v<Str>
    constexpr extent_strides(Ext e, std::array<Str, N> s) : extent(static_cast<index_type>(e))
    {
      for (std::size_t i = 0; i < N; ++i)
      {
        strides[i] = static_cast<std::ptrdiff_t>(s[i]);
      }
    }

    template <typename Ext, typename... Strides>
      requires(std::is_integral_v<Ext> && sizeof...(Strides) == N)
    constexpr extent_strides(Ext e, Strides... s)
        : extent(static_cast<index_type>(e)), strides{static_cast<std::ptrdiff_t>(s)...}
    {}

    /// @brief Returns true if the current (outer) dimension and the given inner dimension can be merged.
    ///
    /// Coalescing is allowed if the inner dimension's stride equals the outer stride multiplied by the outer extent.
    /// This ensures a contiguous memory layout between the two.
    constexpr bool can_merge_with_inner(extent_strides inner) const noexcept
    {
      for (std::size_t i = 0; i < N; ++i)
      {
        if (strides[i] != inner.strides[i] * static_cast<std::ptrdiff_t>(inner.extent)) return false;
      }
      return true;
    }

    /// @brief Merge an inner dimension into this one (assuming can_merge_with_inner returns true).
    ///
    /// After merging, the extent becomes the product, and the stride is updated to the inner stride.
    constexpr void merge_with_inner(extent_strides inner) noexcept
    {
      extent *= inner.extent;
      strides = inner.strides;
    }
};

template <typename Extents, std::size_t R>
  requires(Extents::rank() == R)
inline static_vector<extent_strides<2>, R> coalesce_strides(Extents const& ext,
                                                            std::array<std::ptrdiff_t, R> const& Stride1,
                                                            std::array<std::ptrdiff_t, R> const& Stride2)
{
  static_vector<extent_strides<2>, R> out;
  for (std::size_t i = 0; i < R; ++i)
  {
    out.emplace_back({ext[i], {Stride1[i], Stride2[i]}});
  }

  // Sort by stride1 descending.
  std::sort(out.begin(), out.end(),
            [](auto const& i, auto const& j) { return std::abs(i.strides[0]) > std::abs(j.strides[0]); });

  // Coalesce adjacent dims where both strides agree on contiguity.
  std::size_t write = 1;
  for (std::size_t i = 1; i < R; ++i)
  {
    if (out[write - 1].can_merge_with_inner(out[i]))
      out[write - 1].merge_with_inner(out[i]);
    else
      out[write++] = out[i];
  }
  out.resize(write);

  return out;
}

template <std::size_t N, std::size_t R> void coalesce_strides(static_vector<extent_strides<N>, R>& out)
{
  std::size_t write = 1;
  for (std::size_t i = 1; i < R; ++i)
  {
    if (out[write - 1].can_merge_with_inner(out[i]))
      out[write - 1].merge_with_inner(out[i]);
    else
      out[write++] = out[i];
  }
  out.resize(write);
}

template <StridedMdspan AType, StridedMdspan BType, StridedMdspan CType, std::size_t N>
auto extract_strides(AType const& A, BType const& B,
                     std::array<std::pair<std::size_t, std::size_t>, N> const& contractDims, CType const& C)
{
  constexpr std::size_t MR = AType::rank() - N; // rank of the N group (A/C legs that are not contracted over)
  constexpr std::size_t NR = BType::rank() - N; // rank of the M group (B/C legs that are not contracted over)
  constexpr std::size_t KR = N;                 // rank of the K group (A/Blegs that are contracted over)

  static_vector<extent_strides<2>, MR> Mgroup;
  static_vector<extent_strides<2>, NR> Ngroup;
  static_vector<extent_strides<2>, KR> Kgroup;

  // Assemble the K array of the contracted legs and mark which legs of A and B are contracted over
  std::array<bool, AType::rank()> AContracted{false};
  std::array<bool, BType::rank()> BContracted{false};
  for (std::size_t i = 0; i < N; ++i)
  {
    // check that the extents match
    auto [ai, bi] = contractDims[i];
    ERROR_IF(A.extent(ai) != B.extent(bi), "Extent along tensor contraction dimension does not match", ai, bi);
    AContracted[ai] = true;
    BContracted[bi] = true;
    Kgroup.emplace_back(A.extent(ai), A.stride(ai), B.stride(bi));
  }
  // Now fill out the uncontracted dimensions and verify that they match with C
  std::size_t ci = 0;
  for (std::size_t ai = 0; ai < AType::rank(); ++ai)
  {
    if (!AContracted[ai])
    {
      ERROR_IF(A.extent(ai) != C.extent(ci), "Extent along uncontracted dimension does not match", ai, ci);
      Mgroup.emplace_back(A.extent(ai), A.stride(ai), C.stride(ci));
      ++ci;
    }
  }
  for (std::size_t bi = 0; bi < BType::rank(); ++bi)
  {
    if (!BContracted[bi])
    {
      ERROR_IF(B.extent(bi) != C.extent(ci), "Extent along uncontracted dimension does not match", bi, ci);
      Ngroup.emplace_back(B.extent(bi), B.stride(bi), C.stride(ci));
      ++ci;
    }
  }

  coalesce_strides(Mgroup);
  coalesce_strides(Ngroup);
  coalesce_strides(Kgroup);

  return std::tuple{Mgroup, Ngroup, Kgroup};
}

} // namespace uni20
