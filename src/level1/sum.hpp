#pragma once

#include "apply_unary.hpp"
#include "common/mdspan.hpp"
#include "common/trace.hpp"
#include "common/types.hpp"
#include "level1/concepts.hpp"
#include <ranges>
#include <tuple>

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

// Convenience alias to get the stdex::extents type that represents the common extents of a set of mdspans
template <typename... Spans> using common_extents_t = typename detail::common_extents<Spans...>::type;

} // namespace detail

/// \brief Layout policy for the element‐wise sum of N spans.
///
/// \tparam NumSpans  Number of component spans being summed.
template <std::size_t NumSpans> struct SumLayout
{
    /// \brief Nested mapping type carrying per‐view state.
    ///
    /// \tparam Ext  The merged extents type (e.g. a `stdex::extents<…>`).
    template <class Ext> struct mapping
    {
        static constexpr std::size_t num_spans = NumSpans;
        using extents_type = Ext;
        using index_type = typename Ext::index_type;
        using rank_type = typename Ext::rank_type;
        static constexpr rank_type Rank = Ext::rank();
        using offset_type = std::array<index_type, NumSpans>;

        // For each logical dimension d:
        //  - extent = extents_.extent(d)
        //  - strides[s] = span s's stride in dim d
        using dim_t = multi_extent_stride<index_type, index_type, NumSpans>;
        using dims_t = std::array<dim_t, Rank>;

        extents_type extents_; // merged extents (static/dynamic)
        dims_t dims_;          // per‑dim, per‑span strides

        // Construct from:
        //   * the merged extents object (common_extents<…>::make(...))
        //   * a strides‐pack: array over spans of their stride arrays
        constexpr mapping(extents_type ext, std::array<std::array<index_type, Rank>, NumSpans> strides_pack)
            : extents_(ext), dims_(make_dims(std::move(strides_pack)))
        {}

        template <typename OtherExt>
        constexpr mapping(std::array<index_type, Rank> const& new_strides,
                          typename SumLayout<NumSpans - 1>::template mapping<OtherExt> const& other) noexcept
            : extents_(other.extents_), dims_(make_dims(new_strides, other.dims_))
        {}

        template <typename OtherExt>
        constexpr mapping(typename SumLayout<NumSpans - 1>::template mapping<OtherExt> const& other,
                          std::array<index_type, Rank> const& new_strides) noexcept
            : extents_(other.extents_), dims_(make_dims(other.dims_, new_strides))
        {}

        template <typename LeftMap, typename RightMap>
          requires(LeftMap::num_spans + RightMap::num_spans == num_spans)
        constexpr mapping(LeftMap const& left, RightMap const& right) noexcept
            : extents_(left.extents_), dims_(make_dims(left.dims_, right.dims_))
        {}

        // mdspan mapping API:
        constexpr extents_type const& extents() const noexcept { return extents_; }
        static constexpr auto rank() noexcept { return Rank; }

        // Given a multi‐index, compute each span’s linear offset:
        template <typename... Idx> constexpr offset_type operator()(Idx... idxs) const noexcept
        {
          static_assert(sizeof...(Idx) == Rank, "Wrong #indices");
          std::array<index_type, Rank> idx = {static_cast<index_type>(idxs)...};
          offset_type out{}; // zero‐initialized
          for (std::size_t d = 0; d < Rank; ++d)
            for (std::size_t s = 0; s < NumSpans; ++s)
              out[s] += dims_[d].strides[s] * idx[d];
          return out;
        }

      private:
        // Turn the strides_pack into our dims_ array
        constexpr dims_t make_dims(std::array<std::array<index_type, Rank>, NumSpans> sp) const noexcept
        {
          dims_t result{};
          for (std::size_t d = 0; d < Rank; ++d)
          {
            std::array<index_type, NumSpans> ss{};
            for (std::size_t s = 0; s < NumSpans; ++s)
            {
              ss[s] = sp[s][d];
            }
            // extent check is done by common_extents::make(...)
            result[d] = {extents_.template extent(d), ss};
          }
          return result;
        }

        constexpr dims_t make_dims(
            std::array<index_type, Rank> const& new_strides,
            std::array<multi_extent_stride<index_type, index_type, NumSpans - 1>, Rank> const& other) const noexcept
        {
          dims_t result{};
          for (std::size_t d = 0; d < Rank; ++d)
          {
            std::array<index_type, NumSpans> ss{};
            ss[0] = new_strides[d];
            std::ranges::copy(other[d].strides, ss.data() + 1);
            result[d] = {extents_.template extent(d), ss};
          }
          return result;
        }

        constexpr dims_t
        make_dims(std::array<multi_extent_stride<index_type, index_type, NumSpans - 1>, Rank> const& other,
                  std::array<index_type, Rank> const& new_strides) const noexcept
        {
          dims_t result{};
          for (std::size_t d = 0; d < Rank; ++d)
          {
            std::array<index_type, NumSpans> ss{};
            std::ranges::copy(other[d].strides, ss.data());
            ss[NumSpans - 1] = new_strides[d];
            result[d] = {extents_.template extent(d), ss};
          }
          return result;
        }

        template <size_t N>
        constexpr dims_t make_dims(std::array<multi_extent_stride<index_type, index_type, N>, Rank> const& a,
                                   std::array<multi_extent_stride<index_type, index_type, NumSpans - N>, Rank> const& b)
        {
          static_assert(N <= NumSpans, "Cannot split into more spans than NumSpans");
          dims_t result{};
          for (std::size_t d = 0; d < Rank; ++d)
          {
            std::array<index_type, NumSpans> ss{};
            // first the N strides from `a`:
            std::ranges::copy(a[d].strides, ss.data());
            std::ranges::copy(b[d].strides, ss.data() + N);
            result[d] = dim_t{extents_.template extent(d), ss};
          }
          return result;
        }
    };
};

template <StridedMdspan... Spans> struct SumAccessor
{
    static constexpr std::size_t NumSpans = sizeof...(Spans);

    static_assert(NumSpans >= 2, "Need at least two spans");

    using data_handle_type = std::tuple<typename Spans::data_handle_type...>;
    using accessor_tuple = std::tuple<typename Spans::accessor_type...>;

    // get the offset_type from the layout. This will be a std::array of index_type
    using mapping_type = typename SumLayout<NumSpans>::template mapping<detail::common_extents_t<Spans...>>;
    using offset_type = typename mapping_type::offset_type;

    // Deduce reference as sum of all underlying references:
    using reference = std::decay_t<decltype((std::declval<typename Spans::accessor_type>().access(
                                                 std::declval<typename Spans::data_handle_type>(), index_type()) +
                                             ...))>;

    // True element type (handles proxy references)
    using element_type = uni20::remove_proxy_reference_t<reference>;

    // Store each span’s accessor instance
    constexpr SumAccessor(typename Spans::accessor_type const&... accs) : accessors_(accs...) {}

    // Delegate to each contained accessor.offset():
    constexpr data_handle_type offset(data_handle_type const& handles, offset_type rel) const noexcept
    {
      return offset_impl(handles, rel, std::make_index_sequence<NumSpans>{});
    }

    // index via the array of offsets
    constexpr reference access(data_handle_type const& h, offset_type rel) const noexcept
    {
      return access_impl(h, rel, std::make_index_sequence<NumSpans>{});
    }

    accessor_tuple const& get_accessors() const noexcept { return accessors_; }

  private:
    // Helper that grabs each accessor from the tuple by index and forward to the .offset() method
    template <std::size_t... Is>
    constexpr data_handle_type offset_impl(data_handle_type const& handles, offset_type const& rel,
                                           std::index_sequence<Is...>) const noexcept
    {
      return {std::get<Is>(accessors_).offset(std::get<Is>(handles), rel[Is])...};
    }

    // Helper that grabs each accessor from the tuple by index and forward to the .access() method
    template <std::size_t... Is>
    constexpr reference access_impl(data_handle_type const& h, offset_type rel,
                                    std::index_sequence<Is...>) const noexcept
    {
      return (std::get<Is>(accessors_).access(std::get<Is>(h), rel[Is]) + ...);
    }

    accessor_tuple accessors_;
};

// Trait to detect SumAccessor instantiations
template <typename T> struct is_sum_accessor : std::false_type
{};

template <typename... Spans> struct is_sum_accessor<SumAccessor<Spans...>> : std::true_type
{};

template <typename T> inline constexpr bool is_sum_accessor_v = is_sum_accessor<T>::value;

template <class MDS>
concept SumMdspan = is_sum_accessor_v<typename MDS::accessor_type>;

/// \brief  Element‐wise sum of one or more strided mdspans.
///
/// For each index tuple (i₀,…,i{Rank−1}), returns
/// <code>first(i₀,…,i{Rank−1}) + rest₀(i₀,…,i{Rank−1}) + … + restₙ(i₀,…,i{Rank−1})</code>.
///
/// \tparam First  The first strided mdspan type (\c layout_stride).
/// \tparam Rest   Zero or more additional strided mdspan types.
/// \param  first  The first operand: a raw mdspan view.
/// \param  rest   The remaining operands: raw mdspan views.
/// \return        A new mdspan whose elements are the sum of all inputs.
template <StridedMdspan First, StridedMdspan... Rest> auto sum_view(First const& first, Rest const&... rest)
{
  static_assert((... && (First::rank() == Rest::rank())), "sum_view: rank mismatch");
  assert(((first.extents() == rest.extents()) && ...) && "sum_view: shape mismatch");

  // 1) Compute merged extents and build the mapping:
  using CE = detail::common_extents_t<First, Rest...>;
  CE ext = detail::common_extents<First, Rest...>::make(first, rest...);

  constexpr std::size_t N = 1 + sizeof...(Rest);
  constexpr std::size_t R = CE::rank();

  // 2) Pack per-span stride arrays into a [NumSpans][Rank] array:
  std::array<std::array<typename CE::index_type, R>, N> strides_pack{};
  {
    auto fill = [&](auto const& md, std::size_t s) {
      auto st = md.mapping().strides();
      for (std::size_t d = 0; d < R; ++d)
        strides_pack[s][d] = st[d];
    };
    fill(first, 0);
    std::size_t idx = 1;
    (fill(rest, idx++), ...);
  }

  // 3) Construct the mdspan mapping
  using Layout = SumLayout<N>;
  using Mapping = Layout::template mapping<CE>;
  auto mapping = Mapping{ext, strides_pack};

  // 4) Pack data_handles and accessors
  auto handles = std::tuple{first.data_handle(), rest.data_handle()...};

  using accessor_type = SumAccessor<First, Rest...>;
  accessor_type accessor{first.accessor(), rest.accessor()...};

  // 5) Build resulting mdspan
  return stdex::mdspan<typename accessor_type::element_type, CE, Layout, SumAccessor<First, Rest...>>{handles, mapping,
                                                                                                      accessor};
}

namespace detail
{
template <typename AAcc, typename BAcc> struct join_sum_acc;
template <typename AAcc, typename... Bs> struct join_sum_acc<AAcc, SumAccessor<Bs...>>
{
    using type = SumAccessor<AAcc, Bs...>;
};

template <typename... As, typename Bacc> struct join_sum_acc<SumAccessor<As...>, Bacc>
{
    using type = SumAccessor<As..., Bacc>;
};

template <typename... As, typename... Bs> struct join_sum_acc<SumAccessor<As...>, SumAccessor<Bs...>>
{
    using type = SumAccessor<As..., Bs...>;
};

template <typename SA, typename SB> using join_sum_acc_t = typename join_sum_acc<SA, SB>::type;

} // namespace detail

/// \overload
/// \brief  Sum of a raw strided mdspan and an existing sum‑mdspan.
///
/// Prepends one more span \c A to the front of a \c SumMdspan \c B.  Otherwise
/// the parameter names and return value are the same as the primary overload.
template <StridedMdspan A, SumMdspan B> auto sum_view(A const& a, B const& b)
{
  // Compute merged extents type & object
  using CE = detail::common_extents_t<A, B>;
  CE ext = detail::common_extents<A, B>::make(a, b);

  // Gather A’s per‑dim strides into an array<index_type,Rank>
  constexpr size_t Rank = CE::rank();
  using idx_t = typename CE::index_type;

  // Build the new mapping by prepending A onto B’s mapping
  // Extract how many spans B already has:
  constexpr size_t M = B::mapping_type::num_spans;

  using NewLayout = SumLayout<M + 1>;
  using Mapping = NewLayout::template mapping<CE>;
  Mapping map(a.mapping().strides(), b.mapping());

  // Build the new accessor by prepending A’s accessor onto B’s SumAccessor
  using accessor_type = detail::join_sum_acc_t<A, typename B::accessor_type>;

  accessor_type accessor =
      std::apply([&](auto&&... bs) { return accessor_type(a.accessor(), bs...); }, b.accessor().get_accessors());

  // Concatenate data_handle tuples: one from A, then the tuple from B
  using HNew = typename accessor_type::data_handle_type; // a tuple of child accessors

  HNew handles = std::tuple_cat(std::make_tuple(a.data_handle()), b.data_handle());

  // Return the new sum‑mdspan
  return stdex::mdspan<typename accessor_type::element_type, CE, NewLayout, accessor_type>{handles, map, accessor};
}

/// \overload
/// \brief  Sum of an existing sum‑mdspan and a raw strided mdspan.
///
/// Appends one more span \c B to the end of a \c SumMdspan \c A.  Otherwise
/// the parameter names and return value are the same as the primary overload.
template <SumMdspan A, StridedMdspan B> auto sum_view(A const& a, B const& b)
{
  // Compute the merged extents type & object
  using CE = detail::common_extents_t<A, B>;
  CE ext = detail::common_extents<A, B>::make(a, b);

  // Gather B’s per‑dim strides
  constexpr std::size_t Rank = CE::rank();
  using idx_t = typename CE::index_type;

  // Build the new mapping by appending B onto A’s SumLayout
  constexpr std::size_t M = A::mapping_type::num_spans; // how many spans A already has
  using NewLayout = SumLayout<M + 1>;
  using Mapping = typename NewLayout::template mapping<CE>;

  // this ctor was added to SumLayout::mapping:
  //   mapping(old_mapping, new_strides)
  Mapping map(a.mapping(), b.mapping().strides());

  using accessor_type = detail::join_sum_acc_t<typename A::accessor_type, B>;

  // now unpack A’s tuple and append b.accessor()
  accessor_type accessor =
      std::apply([&](auto&&... as) { return accessor_type(as..., b.accessor()); }, a.accessor().get_accessors());

  // Concatenate the data_handle tuples: tuple from A, one from B
  using HNew = typename accessor_type::data_handle_type;                            // a tuple of child accessors
  HNew handles = std::tuple_cat(a.data_handle(), std::make_tuple(b.data_handle())); // one more for B

  // Return the new sum-mdspan
  return stdex::mdspan<typename accessor_type::element_type, CE, NewLayout, accessor_type>{handles, map, accessor};
}

/// \overload
/// \brief  Sum of two existing sum‑mdspans.
///
/// Flattens both \c A and \c B into a single new sum‑mdspan (i.e.
/// `A₀ + A₁ + … + B₀ + B₁ + …`).  Otherwise the parameter names and return
/// value are the same as the primary overload.
template <SumMdspan A, SumMdspan B> auto sum_view(A const& a, B const& b)
{
  // Compute the merged extents type & object
  using CE = detail::common_extents_t<A, B>;
  CE ext = detail::common_extents<A, B>::make(a, b);

  // Gather B’s per‑dim strides
  constexpr std::size_t Rank = CE::rank();
  using idx_t = typename CE::index_type;

  // Build the new mapping by appending B onto A's SumLayout
  constexpr std::size_t Na = A::mapping_type::num_spans;
  constexpr std::size_t Nb = B::mapping_type::num_spans;
  using NewLayout = SumLayout<Na + Nb>;
  using Mapping = typename NewLayout::template mapping<CE>;

  // this ctor was added to SumLayout::mapping:
  //   mapping(old_mapping, new_strides)
  Mapping map(a.mapping(), b.mapping());

  using accessor_type = detail::join_sum_acc_t<typename A::accessor_type, typename B::accessor_type>;

  // now unpack A’s tuple and append b.accessor()
  accessor_type accessor =
      std::apply([&](auto&&... accs) { return accessor_type(std::forward<decltype(accs)>(accs)...); },
                 std::tuple_cat(a.accessor().get_accessors(), b.accessor().get_accessors()));

  // Concatenate the data_handle tuples: tuple from A, tuple from B
  using HNew = typename accessor_type::data_handle_type; // a tuple of child accessors
  HNew handles = std::tuple_cat(a.data_handle(), b.data_handle());

  // Return the new sum-mdspan
  return stdex::mdspan<typename accessor_type::element_type, CE, NewLayout, accessor_type>{handles, map, accessor};
}

} // namespace uni20
