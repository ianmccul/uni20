#pragma once

#include "apply_unary.hpp"
#include "common/mdspan.hpp"
#include "common/trace.hpp"
#include "core/types.hpp"
#include "mdspan/concepts.hpp"
#include "zip_layout.hpp"
#include <ranges>
#include <tuple>

namespace uni20
{

/// \brief Accessor that sums elements from multiple child spans.
/// \tparam Spans Pack of StridedMdspan types participating in the sum.
/// \ingroup internal
template <StridedMdspan... Spans> struct SumAccessor
{
    static constexpr std::size_t NumSpans = sizeof...(Spans);

    static_assert(NumSpans >= 2, "Need at least two spans");

    using data_handle_type = std::tuple<typename Spans::data_handle_type...>;
    using accessor_tuple = std::tuple<typename Spans::accessor_type...>;

    // get the offset_type from the layout. This will be a std::array of index_type
    using mapping_type = typename StridedZipLayout<NumSpans>::template mapping<common_extents_t<Spans...>>;
    using offset_type = typename mapping_type::offset_type;

    // Deduce reference as sum of all underlying references:
    using reference = std::decay_t<decltype((std::declval<typename Spans::accessor_type>().access(
                                                 std::declval<typename Spans::data_handle_type>(), index_type()) +
                                             ...))>;

    // True element type (handles proxy references)
    using element_type = uni20::remove_proxy_reference_t<reference>;

    /// \brief Store each span's accessor instance.
    /// \param accs Accessor objects sourced from each input span.
    /// \ingroup internal
    constexpr SumAccessor(typename Spans::accessor_type const&... accs) : accessors_(accs...) {}

    /// \brief Delegate to each contained accessor.offset().
    /// \param handles Tuple of child data handles.
    /// \param rel     Per-child relative offsets produced by the mapping.
    /// \return Tuple of advanced data handles.
    /// \ingroup internal
    constexpr data_handle_type offset(data_handle_type const& handles, offset_type rel) const noexcept
    {
      return offset_impl(handles, rel, std::make_index_sequence<NumSpans>{});
    }

    /// \brief Index via the array of offsets.
    /// \param h   Tuple of child data handles.
    /// \param rel Per-child offsets relative to the common origin.
    /// \return    Sum of the referenced elements.
    /// \ingroup internal
    constexpr reference access(data_handle_type const& h, offset_type rel) const noexcept
    {
      return access_impl(h, rel, std::make_index_sequence<NumSpans>{});
    }

    /// \brief Access the stored accessor tuple.
    /// \return Tuple containing the original accessors.
    /// \ingroup internal
    accessor_tuple const& get_accessors() const noexcept { return accessors_; }

  private:
    // Helper that grabs each accessor from the tuple by index and forward to the .offset() method
    template <std::size_t... Is>
    constexpr data_handle_type offset_impl(data_handle_type const& handles, offset_type const& rel,
                                           std::index_sequence<Is...>) const noexcept
    {
      return {std::get<Is>(accessors_).offset(std::get<Is>(handles), std::get<Is>(rel))...};
    }

    // Helper that grabs each accessor from the tuple by index and forward to the .access() method
    template <std::size_t... Is>
    constexpr reference access_impl(data_handle_type const& h, offset_type rel,
                                    std::index_sequence<Is...>) const noexcept
    {
      return (std::get<Is>(accessors_).access(std::get<Is>(h), std::get<Is>(rel)) + ...);
    }

    accessor_tuple accessors_;
};

/// \brief Trait to detect SumAccessor instantiations.
/// \tparam T Candidate accessor type.
/// \ingroup internal
template <typename T> struct is_sum_accessor : std::false_type
{};

/// \brief Specialization for SumAccessor instantiations.
/// \tparam Spans Pack of mdspan types used by the accessor.
/// \ingroup internal
template <typename... Spans> struct is_sum_accessor<SumAccessor<Spans...>> : std::true_type
{};

/// \brief Convenience boolean for the is_sum_accessor trait.
/// \tparam T Candidate accessor type.
/// \ingroup internal
template <typename T> inline constexpr bool is_sum_accessor_v = is_sum_accessor<T>::value;

/// \brief Concept that detects mdspans using SumAccessor.
/// \tparam MDS Mdspan type to inspect.
/// \ingroup internal
template <class MDS>
concept SumMdspan = is_sum_accessor_v<typename MDS::accessor_type>;

/// \brief Element-wise sum of one or more strided mdspans.
/// \details For each index tuple (i₀,…,i{Rank−1}) the resulting view evaluates to
/// <code>first(i₀,…,i{Rank−1}) + rest₀(i₀,…,i{Rank−1}) + … + restₙ(i₀,…,i{Rank−1})</code>.
/// \tparam First First strided mdspan type (typically layout_stride based).
/// \tparam Rest  Zero or more additional strided mdspan types.
/// \param first  First operand as a raw mdspan view.
/// \param rest   Remaining operands as raw mdspan views.
/// \return Mdspan whose elements are the sum of all inputs.
/// \ingroup level1_ops
template <StridedMdspan First, StridedMdspan... Rest> auto sum_view(First const& first, Rest const&... rest)
{
  static_assert((... && (First::rank() == Rest::rank())), "sum_view: rank mismatch");
  assert(((first.extents() == rest.extents()) && ...) && "sum_view: shape mismatch");

  // 1) Compute merged extents and build the mapping:
  using CE = common_extents_t<First, Rest...>;
  CE ext = make_common_extents(first, rest...);

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
  using Layout = StridedZipLayout<N>;
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

/// \brief Sum of a raw strided mdspan and an existing sum-mdspan.
/// \overload
/// \details Prepends one more span \c A to the front of a \c SumMdspan \c B.
/// \tparam A Raw strided mdspan type.
/// \tparam B Sum mdspan type.
/// \param a New leading operand.
/// \param b Existing sum view.
/// \return Mdspan view representing a + b (element-wise).
/// \ingroup level1_ops
template <StridedMdspan A, SumMdspan B> auto sum_view(A const& a, B const& b)
{
  // Compute merged extents type & object
  using CE = common_extents_t<A, B>;

  // Build the new mapping by prepending A onto B’s mapping
  // Extract how many spans B already has:
  constexpr size_t M = B::mapping_type::num_spans;

  using NewLayout = StridedZipLayout<M + 1>;
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

/// \brief Sum of an existing sum-mdspan and a raw strided mdspan.
/// \overload
/// \details Appends one more span \c B to the end of a \c SumMdspan \c A.
/// \tparam A Existing sum mdspan type.
/// \tparam B Raw strided mdspan type.
/// \param a Existing sum view.
/// \param b New trailing operand.
/// \return Mdspan view representing a + b (element-wise).
/// \ingroup level1_ops
template <SumMdspan A, StridedMdspan B> auto sum_view(A const& a, B const& b)
{
  // Compute the merged extents type & object
  using CE = common_extents_t<A, B>;

  // Build the new mapping by appending B onto A’s StridedZipLayout
  constexpr std::size_t M = A::mapping_type::num_spans; // how many spans A already has
  using NewLayout = StridedZipLayout<M + 1>;
  using Mapping = typename NewLayout::template mapping<CE>;

  // this ctor was added to StridedZipLayout::mapping:
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

/// \brief Sum of two existing sum-mdspans.
/// \overload
/// \details Flattens both \c A and \c B into a single new sum-mdspan (i.e.
/// `A₀ + A₁ + … + B₀ + B₁ + …`).
/// \tparam A First sum mdspan type.
/// \tparam B Second sum mdspan type.
/// \param a First sum view.
/// \param b Second sum view.
/// \return Mdspan view representing the concatenated sum of all operands.
/// \ingroup level1_ops
template <SumMdspan A, SumMdspan B> auto sum_view(A const& a, B const& b)
{
  // Compute the merged extents type & object
  using CE = common_extents_t<A, B>;

  // Build the new mapping by appending B onto A's StridedZipLayout
  constexpr std::size_t Na = A::mapping_type::num_spans;
  constexpr std::size_t Nb = B::mapping_type::num_spans;
  using NewLayout = StridedZipLayout<Na + Nb>;
  using Mapping = typename NewLayout::template mapping<CE>;

  // this ctor was added to StridedZipLayout::mapping:
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
