#pragma once

#include "common/mdspan.hpp"
#include "core/types.hpp"
#include "storage/vectorstorage.hpp"

namespace uni20
{

/// \brief Given an ElementType, pick the right accessor for owned storage.
struct DefaultAccessorFactory
{
    /// \brief The accessor to use for plain memory-backed tensors.
    template <typename ElementType> using accessor_t = stdex::default_accessor<ElementType>;
};

/// \brief A dense tensor that manages its own storage
/// \tparam T               Element type (e.g. float, double, complex<double>)
/// \tparam Extents         An extents type (static or dynamic rank)
/// \tparam StoragePolicy   Policy that provides a container for T
/// \tparam LayoutPolicy    mdspan layout (e.g. stdex::layout_stride)
/// \tparam AccessorPolicy  mdspan accessor (e.g. stdex::default_accessor<T>)
template <typename ElementType, typename Extents, typename StoragePolicy, typename LayoutPolicy,
          typename AccessorPolicy = DefaultAccessorFactory>
class TensorView;

template <typename T, typename Extents, typename StoragePolicy, typename LayoutPolicy, typename AccessorPolicy>
class TensorView<T const, Extents, StoragePolicy, LayoutPolicy, AccessorPolicy> {
  public:
    /// \name Public type aliases
    ///@{
    using element_type = T const;
    using value_type = T;
    using extents_type = Extents;
    using storage_policy = StoragePolicy;
    using layout_policy = LayoutPolicy;
    using accessor_policy = AccessorPolicy;

    using handle_type = typename storage_policy::template handle_t<T>;
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    using accessor_type = typename accessor_policy::template accessor_t<element_type>;
    using const_accessor_type = const_accessor_t<accessor_type>;
    using mdspan_type = stdex::mdspan<element_type, extents_type, layout_type, accessor_type>;
    using const_mdspan_type = stdex::mdspan<element_type, extents_type, layout_type, const_accessor_type>;
    using index_type = typename extents_type::index_type;
    using size_type = uni20::size_type;
    using reference = typename const_accessor_type::reference;
    ///@}

    /// \brief Default‐construct an “empty” tensor (no storage, uninitialized extents).
    TensorView() = default;

    /// \brief Construct from extents (+ optional strides).
    /// \param exts     The shape in each dimension.
    /// \param strides  Must be an array of length rank().
    explicit TensorView(extents_type const& exts, std::array<index_type, extents_type::rank()> strides)
        : data_{exts.required_span_size()}, view_{data_.data(), mapping_type{exts, strides}}
    {}

    /// \brief Construct from extents (+ optional strides).
    /// \param exts         The shape in each dimension.
    /// \param strides_opt  If provided, must be an array of length rank().
    ///                     Otherwise a row-major default is used.
    explicit TensorView(extents_type const& exts)
        : data{exts.required_span_size()}, view_{data_.data(), mapping_type{exts, default_strides(exts)}}
    {}

    /// \brief Access via bracket-operator: t[i0,i1,…]
    template <typename... Idx> reference operator[](Idx... idxs) const noexcept
    {
      return this->accessor_(handle_, mapping_(static_cast<index_type>(idxs...)));
    }

    /// \brief The mdspan view (mapping + accessor).
    const_mdspan_type mdspan() const noexcept { return const_mdspan_type(handle_, mapping_, accessor_); }

    /// \brief Rank of the tensor
    static constexpr size_type rank() noexcept { return view_.rank(); }

    /// \brief Number of elements = required_span_size().
    size_type size() const noexcept { return mapping_.required_span_size(); }

    /// \brief handle to the buffer -- only exists if the storage is local
    handle_type handle() const noexcept { return handle_; }

    /// \brief The extents (shape).
    extents_type extents() const noexcept { return extents_; }

    /// \brief The layout mapping (holds strides + extents).
    mapping_type mapping() const noexcept { return mapping_; }

  protected:
    handle_type handle_; ///< Element buffer, may be owned, or not
    [[no_unique_address]] extents_type extents_;
    [[no_unique_address]] mapping_type mapping_;
    [[no_unique_address]] accessor_type accessor_;

    /// \brief Default row-major strides helper.
    static constexpr std::array<index_type, extents_type::rank()> default_strides(extents_type const& exts) noexcept
    {
      std::array<index_type, extents_type::rank()> s{};
      index_type run = 1;
      // fill backwards for row-major
      for (int d = int(extents_type::rank()) - 1; d >= 0; --d)
      {
        s[d] = run;
        run *= exts.extent(d);
      }
      return s;
    }
};

template <typename T, typename Extents, typename StoragePolicy, typename LayoutPolicy, typename AccessorFactory>
class TensorView<T, Extents, StoragePolicy, LayoutPolicy, AccessorFactory>
    : public TensorView<T const, Extents, StoragePolicy, LayoutPolicy, AccessorFactory> {
  public:
    /// \name Public type aliases
    ///@{
    using base_t = TensorView<T const, Extents, StoragePolicy, LayoutPolicy, AccessorFactory>;
    using element_type = T;
    using typename base_t::accessor_type;
    using typename base_t::const_mdspan_type;
    using typename base_t::extents_type;
    using typename base_t::layout_type;
    using typename base_t::mapping_type;
    using typename base_t::mdspan_type;
    using typename base_t::size_type;
    using typename base_t::storage_type;
    using typename base_t::value_type;
    using mdspan_type = stdex::mdspan<element_type, extents_type, layout_type, accessor_type>;
    using accessor_type = typename accessor_policy::template accessor_policy<element_type>;
    using const_accessor_type = const_accessor_t<accessor_type>;
    using reference = typename accessor_type::reference;
    ///@}

    using base_t::extents;
    using base_t::mapping;
    using base_t::mdspan;
    using base_t::rank;
    using base_t::size;
    using base_t::storage;

    mdspan_type mutable_mdspan() const noexcept { return view_; }
}

} // namespace uni20
