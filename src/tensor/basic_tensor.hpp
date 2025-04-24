#pragma once

#include "common/mdspan.hpp"
#include "common/types.hpp"
#include "storage/vectorstorage.hpp"

namespace uni20
{

/// \brief Given an ElementType, pick the right accessor for owned storage.
struct DefaultAccessorFactory
{
    /// \brief The accessor to use for plain memory-backed tensors.
    template <typename ElementType> using accessor_policy = stdex::default_accessor<ElementType>;
};

/// \brief A dense tensor that manages its own storage
/// \tparam T               Element type (e.g. float, double, complex<double>)
/// \tparam Extents         An extents type (static or dynamic rank)
/// \tparam StoragePolicy   Policy that provides a container for T
/// \tparam LayoutPolicy    mdspan layout (e.g. stdex::layout_stride)
/// \tparam AccessorPolicy  mdspan accessor (e.g. stdex::default_accessor<T>)
template <typename ElementType, stdex::extents_t Extents, typename StoragePolicy, typename LayoutPolicy,
          typename AccessorFactory>
class BasicTensor {
  public:
    /// \name Public type aliases
    ///@{
    using element_type = T;
    using extents_type = Extents;
    using storage_type = typename StoragePolicy::template storage_t<T>;
    using layout_type = LayoutPolicy;
    using mapping_type = typename LayoutPolicy::template mapping<extents_type>;
    using accessor_type = typename AccessorFactory::template accessor_policy<element_type>;
    using mdspan_type = stdex::mdspan<T, extents_type, layout_type, accessor_type>;
    using index_type = typename extents_type::index_type;
    using size_type = uni20::size_type;
    using reference = typename accessor_type::reference;
    ///@}

    /// \brief Default‐construct an “empty” tensor (no storage, uninitialized extents).
    BasicTensor() = default;

    /// \brief Construct from extents (+ optional strides).
    /// \param exts     The shape in each dimension.
    /// \param strides  Must be an array of length rank().
    explicit BasicTensor(extents_type const& exts, std::array<index_type, extents_type::rank()> strides)
        : data_{exts.required_span_size()}, view_{data_.data(), mapping_type{exts, strides}}
    {}

    /// \brief Construct from extents (+ optional strides).
    /// \param exts         The shape in each dimension.
    /// \param strides_opt  If provided, must be an array of length rank().
    ///                     Otherwise a row-major default is used.
    explicit BasicTensor(extents_type const& exts)
        : data{exts.required_span_size()}, view_{data_.data(), mapping_type{exts, default_strides(exts)}}
    {}

    /// \brief Access via bracket-operator: t[i0,i1,…]
    template <typename... Idx> reference operator[](Idx... idxs) { return view_[static_cast<index_type>(idxs)...]; }
    template <typename... Idx> element_type operator[](Idx... idxs) const
    {
      return view_[static_cast<index_type>(idxs)...];
    }

    /// \brief Underlying storage container (e.g. std::vector<T>).
    storage_type& storage() noexcept { return data_; }
    storage_type const& storage() const noexcept { return data_; }

    /// \brief The mdspan view (mapping + accessor).
    mdspan_type& view() noexcept { return view_; }
    mdspan_type const& view() const noexcept { return view_; }

    /// \brief Rank of the tensor
    static constexpr size_type rank() noexcept { return view_.rank(); }

    /// \brief Number of elements = required_span_size().
    size_type size() const noexcept { return view_.mapping().required_span_size(); }

    /// \brief The extents (shape).
    extents_type extents() const noexcept { return view_.mapping().extents(); }

    /// \brief The layout mapping (holds strides + extents).
    mapping_type mapping() const noexcept { return view_.mapping(); }

  private:
    storage_type data_; ///< Owned element buffer
    mdspan_type view_;  ///< mdspan over that buffer

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

// A convenient alias for the  common case:
//   - dynamic extents
//   - default to std::vector storage
//   - default to layout_stride + default accessor factory
template <typename ElementType, std::size_t Rank, typename StoragePolicy = VectorStorage,
          typename LayoutPolicy = stdex::latout_strided, typename AccessorFactory = DefaultAccessorFactory>
using Tensor =
    BasicTensor<ElementType, stdex::dextents<index_type, Rank>, StoragePolicy, LayoutPolicy, AccessorFactory>;

/// \brief Non-owning view: wraps any mdspan-like object (pointer+mapping+accessor).
/// \tparam T              Element type.
/// \tparam Extents        mdspan extents type.
/// \tparam StoragePolicy  Carries the tag_t for backend dispatch.
/// \tparam LayoutPolicy   mdspan layout.
/// \tparam AccessorPolicy The actual accessor type (e.g. default_accessor).
template <typename T, class Extents, class StoragePolicy, class LayoutPolicy, class AccessorPolicy> class TensorView {
  public:
    using element_type = T;
    using extents_type = Extents;
    using layout_type = LayoutPolicy;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    using accessor_type = AccessorPolicy;
    using reference = typename acessor_type::reference;

    using mdspan_type = stdex::mdspan<element_type, extents_type, layout_type, accessor_type>;

    using index_type = typename extents_type::index_type;
    using size_type = uni20::size_type;

    /// \brief Wrap a raw pointer + mapping + accessor.
    TensorView(T* data, mapping_type const& map, accessor_type acc = {}) : view_{data, map, std::move(acc)} {}

    /// \brief Wrap any mdspan-compatible object.
    TensorView(mdspan_type span) : view_{std::move(span)} {}

    /// \brief Element access.
    template <typename... Idx> reference operator[](Idx... idxs) { return view_[static_cast<index_type>(idxs)...]; }
    template <typename... Idx> element_type operator[](Idx... idxs) const
    {
      return view_[static_cast<index_type>(idxs)...];
    }

    /// \brief Access the underlying mdspan.
    mdspan_type& view() noexcept { return view_; }
    mdspan_type const& view() const noexcept { return view_; }

  private:
    mdspan_type view_;
};

} // namespace uni20
