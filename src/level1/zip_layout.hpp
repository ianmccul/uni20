#pragma once

#include "common/mdspan.hpp"
#include "common/trace.hpp"
#include "level1/concepts.hpp"
#include <functional>

namespace uni20
{

namespace detail
{

// Helper function to merge extents. If the extents are different, then one of them must be dynamic,
// otherwise they are not compatible. If one is dynamic and one is static, then return the static extent.

template <std::size_t A, std::size_t B> static constexpr std::size_t merge_extent()
{
  if constexpr (A == B)
    return A;
  else if constexpr (A == stdex::dynamic_extent)
    return B;
  else if constexpr (B == stdex::dynamic_extent)
    return A;
  else
    static_assert(A == B || A == stdex::dynamic_extent || B == stdex::dynamic_extent, "Incompatible static extents");
  return 0;
}

// Primary template: merge_pack over a pack of N extents
template <std::size_t... Ns> struct merge_pack;

// Base case: single value
template <std::size_t A> struct merge_pack<A>
{
    static constexpr std::size_t value = A;
};

// Recursive case: merge first two, then fold in the rest
template <std::size_t A, std::size_t B, std::size_t... Rest> struct merge_pack<A, B, Rest...>
{
    static constexpr std::size_t value = merge_pack<merge_extent<A, B>(), Rest...>::value;
};

// Given a set of mdspans, determine the common extents using the merge_extent() function above
template <typename FirstSpan, typename... OtherSpans> struct common_extents
{
    static constexpr std::size_t NumSpans = 1 + sizeof...(OtherSpans);

    // 1) All spans must have the same rank R
    static constexpr std::size_t R = FirstSpan::rank();
    static_assert((... && (OtherSpans::rank() == R)), "common_extents: ranks must match");

    using index_type = FirstSpan::index_type;

  private:
    // Fold-merge the static extents at dimension I
    template <std::size_t I> static constexpr std::size_t merged_static_extent()
    {
      // build a pack of the static‑extent<I> of each Span
      return merge_pack<FirstSpan::extents_type::static_extent(I),
                        OtherSpans::extents_type::static_extent(I)...>::value;
    }

    // Build the extents type by expanding merged_static_extent<Is>()...
    template <std::size_t... Is> static constexpr auto make_extents(std::index_sequence<Is...>)
    {
      return stdex::extents<index_type,
                            merged_static_extent<Is>()... // one per dimension
                            >{};
    }

  public:
    /// The merged extents type
    using type = decltype(make_extents(std::make_index_sequence<R>{}));

  private:
    template <std::size_t... Is>

    static type make_impl(std::index_sequence<Is...>, FirstSpan const& first, OtherSpans const&... otherspans)
    {
      // Construct the extents object. All of the extents of each span must agree.
      type ext{first.template extent(Is)...};

      auto do_check = [&](auto const& sp) { CHECK_EQUAL(ext, sp.extents()); };

      (do_check(otherspans), ...);

      return ext;
    }

  public:
    /// Runtime‑checked factory
    static type make(FirstSpan const& firstspan, OtherSpans const&... otherspans)
    {
      return make_impl(std::make_index_sequence<R>{}, firstspan, otherspans...);
    }
};

} // namespace detail

// Convenience alias to get the stdex::extents type that represents the common extents of a set of mdspans
template <typename... Spans> using common_extents_t = typename detail::common_extents<Spans...>::type;

template <typename... Spans> auto make_common_extents(Spans... spans)
{
  return detail::common_extents<Spans...>::make(spans...);
}

namespace detail
{

template <typename T, std::size_t N1, std::size_t N2, std::size_t... I, std::size_t... J>
constexpr std::array<T, N1 + N2> concat_impl(const std::array<T, N1>& a1, const std::array<T, N2>& a2,
                                             std::index_sequence<I...>, std::index_sequence<J...>)
{
  // pack-expand a1[I]..., then a2[J]...
  return {{a1[I]..., a2[J]...}};
}

template <typename T, std::size_t N, std::size_t... I>
constexpr std::array<T, N + 1> concat_impl(const std::array<T, N>& a, const T& x, std::index_sequence<I...>)
{
  return {{a[I]..., x}};
}

template <typename T, std::size_t N, std::size_t... I>
constexpr std::array<T, N + 1> concat_impl(const T& x, const std::array<T, N>& a, std::index_sequence<I...>)
{
  return {{x, a[I]...}};
}

} // namespace detail

/// \brief Concat two std::array into one (constexpr).
template <typename T, std::size_t N1, std::size_t N2>
constexpr std::array<T, N1 + N2> concat(const std::array<T, N1>& a1, const std::array<T, N2>& a2)
{
  return detail::concat_impl(a1, a2, std::make_index_sequence<N1>{}, std::make_index_sequence<N2>{});
}

/// \brief Prepend a single element in front of an array.
template <typename T, std::size_t N> constexpr std::array<T, N + 1> concat(const T& x, const std::array<T, N>& a)
{
  return detail::concat_impl(x, a, std::make_index_sequence<N>{});
}

/// \brief Append a single element to the end of an array.
template <typename T, std::size_t N> constexpr std::array<T, N + 1> concat(const std::array<T, N>& a, const T& x)
{
  return detail::concat_impl(a, x, std::make_index_sequence<N>{});
}

namespace detail
{

// build a std::tuple of N default-constructed T’s
template <typename T, std::size_t... I> constexpr auto make_tuple_n_impl(std::index_sequence<I...>)
{
  // the comma‐expression (static_cast<void>(I), T{}) ensures we produce T{} N times
  return std::tuple{(static_cast<void>(I), T{})...};
}

} // namespace detail

/// \brief Alias: tuple_n<T,N> is std::tuple<T,T,…> with N Ts.
/// \tparam T  The type to repeat.
/// \tparam N  How many copies.
template <typename T, std::size_t N>
using tuple_n = decltype(detail::make_tuple_n_impl<T>(std::make_index_sequence<N>{}));

/// \brief Zip layout for \p NumSpans strided spans (all using layout_stride).
/// \tparam NumSpans  Number of child spans to zip.
template <std::size_t NumSpans> struct StridedZipLayout
{
    static_assert(NumSpans >= 1, "StridedZipLayout requires at least one span");

    /// \brief Number of spans in this zip layout.
    static constexpr std::size_t num_spans = NumSpans;

    /// \brief Nested mapping policy for extents \p Ext, modeling LayoutMappingPolicy.
    template <typename Ext> struct mapping
    {
        using layout_type = StridedZipLayout; ///< back‑link to policy
        using extents_type = Ext;             ///< multidimensional shape
        using index_type = typename Ext::index_type;
        using rank_type = typename Ext::rank_type;

        using offset_type = tuple_n<index_type, NumSpans>;

        using mapping_type = mapping<Ext>;

        /// \brief Number of spans in this zip layout.
        static constexpr std::size_t num_spans = NumSpans;

        /// \brief Always unique for uni20 tensors
        static constexpr bool is_always_unique() noexcept { return true; }
        /// \brief Never exhaustive: no single contiguous backing buffer.
        static constexpr bool is_always_exhaustive() noexcept { return false; }
        /// \brief Statically unknown if strided: may differ per dimension.
        static constexpr bool is_always_strided() noexcept { return false; }

        /// \brief Always unique at runtime.
        constexpr bool is_unique() const noexcept { return true; }
        /// \brief Never exhaustive at runtime.
        constexpr bool is_exhaustive() const noexcept { return false; }

        // std::size_t required_span_size() const does not make sense for a StridedZipLayout

        /// \brief True if every dimension uses the same stride across all spans.
        constexpr bool is_strided() const noexcept
        {
          for (std::size_t d = 0; d < Ext::rank(); ++d)
          {
            auto s0 = all_strides_[0][d];
            for (std::size_t i = 1; i < NumSpans; ++i)
            {
              if (all_strides_[i][d] != s0) return false;
            }
          }
          return true;
        }

        constexpr std::array<index_type, Ext::rank()> strides() const noexcept
        {
          DEBUG_PRECONDITION(this->is_strided());
          return all_strides_[0];
        }

        /// \brief Return the common stride in dimension \p r.
        /// \pre All spans share the same stride in dimension \p r.
        /// \param r Dimension index, 0 ≤ r < rank().
        /// \return The stride (step) in the flattened data for dim \p r.
        constexpr index_type stride(rank_type r) const noexcept
        {
          DEBUG_PRECONDITION(this->is_strided());
          return all_strides_[0][r];
        }

        /// \brief Retrieve the 2-D array of all per-span strides.
        // \return An array [span][dim] of strides.
        std::array<std::array<index_type, Ext::rank()>, num_spans> const& all_strides() const noexcept
        {
          return all_strides_;
        }

        template <typename... Mappings>
        constexpr mapping(extents_type const& exts, Mappings... maps) noexcept
            : extents_(exts), all_strides_{maps.strides()...}
        {}

        /// \brief Construct from shared extents and per-span raw strides.
        /// \param exts         Common extents of the view.
        /// \param strides_pack strides_pack[s][d] is the stride of span s in dim d.
        constexpr mapping(extents_type const& exts,
                          std::array<std::array<index_type, Ext::rank()>, NumSpans> const& strides_pack) noexcept
            : extents_(exts), all_strides_(strides_pack)
        {}

        /// \brief Prepend one span’s strides before an existing mapping of N–1 spans.
        /// \tparam OtherExt  Extents type of the smaller mapping.
        template <typename OtherExt>
        constexpr mapping(std::array<index_type, Ext::rank()> const& new_strides,
                          typename StridedZipLayout<num_spans - 1>::template mapping<OtherExt> const& other) noexcept
            : extents_(other.extents()), all_strides_(concat(new_strides, other.all_strides()))
        {}

        /// \brief Append one span’s strides after an existing mapping of N–1 spans.
        /// \tparam OtherExt  Extents type of the smaller mapping.
        template <typename OtherExt>
        constexpr mapping(typename StridedZipLayout<num_spans - 1>::template mapping<OtherExt> const& other,
                          std::array<index_type, Ext::rank()> const& new_strides) noexcept
            : extents_(other.extents()), all_strides_(concat(other.all_strides(), new_strides))
        {}

        /// \brief Merge two sub‑mappings whose span counts sum to NumSpans.
        /// \tparam LeftMap   Mapping of the first n spans.
        /// \tparam RightMap  Mapping of the remaining spans.
        template <typename LeftMap, typename RightMap>
          requires(LeftMap::num_spans + RightMap::num_spans == num_spans)
        constexpr mapping(LeftMap const& left, RightMap const& right) noexcept
            : extents_(left.extents()), all_strides_(concat(left.all_strides(), right.all_strides()))
        {}

        /// \brief Return the extents of the layout
        /// \return The common extents of all of the child layouts
        constexpr extents_type const& extents() const noexcept { return extents_; }

        /// \brief Compute per-span linear offsets for a multi‑index.
        /// \tparam Idx  Types of each index argument.
        /// \param idxs  Indices for each dimension.
        /// \return      Array of per-span offsets.
        template <typename... Idx> constexpr offset_type operator()(Idx... idxs) const noexcept
        {
          static_assert(sizeof...(Idx) == Ext::rank(), "Wrong number of indices");
          // pack the multi-index into an array
          std::array<index_type, Ext::rank()> ix{static_cast<index_type>(idxs)...};
          // dispatch over spans
          return offset_impl(ix, std::make_index_sequence<NumSpans>{});
        }

      private:
        // For a given span S, compute its offset by folding over all dims D:
        template <std::size_t S, std::size_t... D>
        constexpr index_type span_offset(std::array<index_type, Ext::rank()> const& ix,
                                         std::index_sequence<D...>) const noexcept
        {
          // fold: 0 + (stride[S][0]*ix[0]) + (stride[S][1]*ix[1]) + …
          return ((all_strides_[S][D] * ix[D]) + ... + index_type(0));
        }

        // Build the tuple of all span-offsets in one go:
        template <std::size_t... S>
        constexpr offset_type offset_impl(std::array<index_type, Ext::rank()> const& ix,
                                          std::index_sequence<S...>) const noexcept
        {
          // pack-expand: span_offset<0>(ix), span_offset<1>(ix), …
          return offset_type{span_offset<S>(ix, std::make_index_sequence<Ext::rank()>{})...};
        }

        extents_type extents_;
        std::array<std::array<index_type, Ext::rank()>, num_spans> all_strides_;
    };
};

// \brief Fallback “zip” layout for arbitrary mapping policies.
/// Models the C++23 LayoutMappingPolicy requirements.
template <typename... Layouts> struct GeneralZipLayout
{
    template <typename Extents> struct mapping
    {
        using layout_type = GeneralZipLayout;
        using extents_type = Extents;
        using index_type = typename Extents::index_type;
        using rank_type = typename Extents::rank_type;
        static constexpr rank_type rank() noexcept { return Extents::rank(); }
        static constexpr rank_type rank_dynamic() noexcept { return Extents::rank_dynamic(); }
        using offset_type = std::tuple<span_offset_t<typename Layouts::template mapping<Extents>>...>;
        using mapping_type = mapping<Extents>;

        /// \brief Always unique for uni20 tensors
        static constexpr bool is_always_unique() noexcept { return true; }
        /// \brief Never exhaustive: no single contiguous backing buffer.
        static constexpr bool is_always_exhaustive() noexcept { return false; }
        /// \brief Not strided
        static constexpr bool is_always_strided() noexcept { return false; }

        /// \brief Construct child mappings
        explicit constexpr mapping(Extents const& ext, typename Layouts::template mapping<Extents> const&... m) noexcept
            : extents_(ext), impls_{m...}
        {}

        constexpr extents_type const& extents() const noexcept { return extents_; }

        constexpr std::size_t required_span_size() const noexcept
        {
          std::size_t max_sz = 0;
          std::apply(
              [&](auto const&... m) {
                (void)std::initializer_list<int>{
                    (max_sz = std::max<std::size_t>(max_sz, m.required_span_size()), 0)...};
              },
              impls_);
          return max_sz;
        }

        template <typename... Idx> constexpr offset_type operator()(Idx... idxs) const noexcept
        {
          return std::apply([&](auto const&... m) { return offset_type{m(idxs...)...}; }, impls_);
        }

        constexpr bool operator==(mapping const& o) const noexcept
        {
          return extents_ == o.extents_ && impls_ == o.impls_;
        }
        constexpr bool operator!=(mapping const& o) const noexcept { return !(*this == o); }

      private:
        extents_type extents_;
        std::tuple<typename Layouts::template mapping<Extents>...> impls_;
    };
};

/// \brief Picks the right Zip‑Layout for a pack of mdspan types.
/// \tparam Spans  The mdspan types (e.g. stdex::mdspan<…>).
template <typename... Spans> struct zip_layout_selector
{
    using type = GeneralZipLayout<typename Spans::layout_type...>;
};

/// \brief Specialization: if *all* Spans are StridedMdspan,
///        use StridedZipLayout instead.
/// \tparam Spans  The mdspan types.
template <StridedMdspan... Spans> struct zip_layout_selector<Spans...>
{
    using type = StridedZipLayout<sizeof...(Spans)>;
};

/// \brief Alias to extract the chosen layout.
/// \tparam Spans  The mdspan types.
template <typename... Spans> using zip_layout_t = typename zip_layout_selector<Spans...>::type;

} // namespace uni20
