#pragma once

#include "common/mdspan.hpp"
#include "core/types.hpp"
#include "mdspan/concepts.hpp"
#include "storage/vectorstorage.hpp"

#include <array>
#include <type_traits>
#include <utility>

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

    using mapping_type = typename layout_policy::template mapping<extents_type>;
    using mutable_accessor_type = typename accessor_policy::template accessor_t<value_type>;
    using accessor_type = const_accessor_t<mutable_accessor_type>;
    using handle_type = typename accessor_type::data_handle_type;
    using mdspan_type = stdex::mdspan<element_type, extents_type, layout_policy, accessor_type>;
    using index_type = typename extents_type::index_type;
    using size_type = uni20::size_type;
    using reference = typename accessor_type::reference;
    ///@}

    /// \brief Default-construct an empty view.
    TensorView() = default;

    /// \brief Construct from a handle, mapping, and accessor.
    TensorView(handle_type handle, mapping_type mapping, accessor_type accessor = accessor_type{})
        : handle_{std::move(handle)}, mapping_{std::move(mapping)}, extents_{mapping_.extents()}, accessor_{std::move(
                                                                                                      accessor)}
    {}

    /// \brief Construct from a handle and extents, using the layout mapping's preferred form.
    TensorView(handle_type handle, extents_type const& exts, accessor_type accessor = accessor_type{})
        : TensorView(std::move(handle), construct_mapping(exts), std::move(accessor))
    {}

    /// \brief Construct from a handle, extents, and explicit strides.
    TensorView(handle_type handle, extents_type const& exts,
               std::array<index_type, extents_type::rank()> const& strides, accessor_type accessor = accessor_type{})
        : TensorView(std::move(handle), mapping_type{exts, strides}, std::move(accessor))
    {}

    /// \brief Access via bracket-operator: t[i0,i1,…]
    template <typename... Idx>
      requires(sizeof...(Idx) == extents_type::rank())
    reference operator[](Idx... idxs) const noexcept
    {
      auto const offset = this->mapping_(static_cast<index_type>(idxs)...);
      return this->accessor_.access(this->handle_, offset);
    }

    /// \brief The mdspan view (mapping + accessor).
    mdspan_type mdspan() const noexcept { return mdspan_type(handle_, mapping_, accessor_); }

    /// \brief Rank of the tensor.
    static constexpr size_type rank() noexcept { return extents_type::rank(); }

    /// \brief Number of elements = required_span_size().
    size_type size() const noexcept { return mapping_.required_span_size(); }

    /// \brief Handle to the buffer.
    handle_type handle() const noexcept { return handle_; }

    /// \brief The extents (shape).
    extents_type const& extents() const noexcept { return extents_; }

    /// \brief The layout mapping (holds strides + extents).
    mapping_type const& mapping() const noexcept { return mapping_; }

    /// \brief The accessor in use.
    accessor_type const& accessor() const noexcept { return accessor_; }

  protected:
    handle_type handle_{}; ///< Element buffer, may be owned, or not
    [[no_unique_address]] mapping_type mapping_{};
    [[no_unique_address]] extents_type extents_{};
    [[no_unique_address]] accessor_type accessor_{};

    /// \brief Default row-major strides helper.
    static constexpr std::array<index_type, extents_type::rank()> default_strides(extents_type const& exts) noexcept
    {
      std::array<index_type, extents_type::rank()> s{};
      index_type run = 1;
      for (int d = int(extents_type::rank()) - 1; d >= 0; --d)
      {
        s[d] = run;
        run *= static_cast<index_type>(exts.extent(d));
      }
      return s;
    }

    static constexpr mapping_type construct_mapping(extents_type const& exts)
    {
      if constexpr (std::is_constructible_v<mapping_type, extents_type const&>)
      {
        return mapping_type{exts};
      }
      else if constexpr (std::is_constructible_v<mapping_type, extents_type const&,
                                                 std::array<index_type, extents_type::rank()>>)
      {
        return mapping_type{exts, default_strides(exts)};
      }
      else
      {
        static_assert(std::is_constructible_v<mapping_type, extents_type const&>,
                      "TensorView cannot construct the requested layout mapping");
        return mapping_type{exts};
      }
    }
};

template <typename T, typename Extents, typename StoragePolicy, typename LayoutPolicy, typename AccessorPolicy>
class TensorView<T, Extents, StoragePolicy, LayoutPolicy, AccessorPolicy>
    : public TensorView<T const, Extents, StoragePolicy, LayoutPolicy, AccessorPolicy> {
  public:
    using base_type = TensorView<T const, Extents, StoragePolicy, LayoutPolicy, AccessorPolicy>;

    using element_type = T;
    using value_type = T;
    using typename base_type::accessor_policy;
    using typename base_type::extents_type;
    using typename base_type::handle_type;
    using typename base_type::index_type;
    using typename base_type::layout_policy;
    using typename base_type::mapping_type;
    using typename base_type::size_type;

    using accessor_type = typename accessor_policy::template accessor_t<element_type>;
    using const_accessor_type = const_accessor_t<accessor_type>;
    using mdspan_type = stdex::mdspan<element_type, extents_type, layout_policy, accessor_type>;
    using reference = typename accessor_type::reference;

    TensorView() = default;

    TensorView(handle_type handle, mapping_type mapping, accessor_type accessor = accessor_type{})
        : base_type(handle, std::move(mapping), const_accessor(accessor)), accessor_mut_{std::move(accessor)}
    {}

    TensorView(handle_type handle, extents_type const& exts, accessor_type accessor = accessor_type{})
        : TensorView(std::move(handle), base_type::construct_mapping(exts), std::move(accessor))
    {}

    TensorView(handle_type handle, extents_type const& exts,
               std::array<index_type, extents_type::rank()> const& strides, accessor_type accessor = accessor_type{})
        : base_type(handle, mapping_type{exts, strides}, const_accessor(accessor)), accessor_mut_{std::move(accessor)}
    {}

    template <typename... Idx>
      requires(sizeof...(Idx) == extents_type::rank())
    reference operator[](Idx... idxs) noexcept
    {
      auto const offset = this->mapping_(static_cast<index_type>(idxs)...);
      return accessor_mut_.access(this->handle_, offset);
    }

    using base_type::operator[];

    mdspan_type mdspan() const noexcept { return mdspan_type(this->handle_, this->mapping_, accessor_mut_); }

    accessor_type const& accessor() const noexcept { return accessor_mut_; }

  private:
    accessor_type accessor_mut_{};
};

} // namespace uni20
