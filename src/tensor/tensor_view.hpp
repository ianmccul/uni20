/**
 * \file tensor_view.hpp
 * \ingroup tensor
 * \brief Tensor view abstractions and accessor factories built on std::mdspan.
 */

/**
 * \defgroup tensor Tensor container primitives
 * \brief Owning and non-owning tensor abstractions layered on std::mdspan.
 */

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

/// \brief Factory that provides default accessors for tensor storage containers.
/// \ingroup tensor
struct DefaultAccessorFactory
{
    /// \brief Accessor alias for plain memory-backed tensors.
    /// \tparam ElementType Value type accessed by the tensor view.
    template <typename ElementType> using accessor_t = stdex::default_accessor<ElementType>;

    /// \brief Construct an accessor instance for the provided storage container.
    /// \tparam ElementType Value type accessed by the tensor view.
    /// \tparam Storage Storage container type.
    /// \param storage Storage container used to determine accessor customisation.
    /// \return Accessor that can interact with the supplied storage object.
    template <typename ElementType, typename Storage>
    constexpr auto make_accessor(Storage const&) const noexcept -> accessor_t<ElementType>
    {
      return accessor_t<ElementType>{};
    }
};

/// \brief Forward declaration for the TensorView template.
/// \ingroup tensor
/// \tparam ElementType Value type viewed by the tensor.
/// \tparam Extents Extents type describing the tensor shape.
/// \tparam StoragePolicy Policy that identifies how storage is owned or referenced.
/// \tparam LayoutPolicy Layout policy that maps indices to offsets.
/// \tparam AccessorPolicy Accessor policy used to interact with the storage handle.
template <typename ElementType, typename Extents, typename StoragePolicy, typename LayoutPolicy,
          typename AccessorPolicy = DefaultAccessorFactory>
class TensorView;

/// \brief Tensor view specialisation for const-qualified element access.
/// \ingroup tensor
/// \tparam T Base value type without const-qualification.
/// \tparam Extents Extents type describing the tensor shape.
/// \tparam StoragePolicy Policy that identifies how storage is owned or referenced.
/// \tparam LayoutPolicy Layout policy that maps indices to offsets.
/// \tparam AccessorPolicy Accessor policy used to interact with the storage handle.
template <typename T, typename Extents, typename StoragePolicy, typename LayoutPolicy, typename AccessorPolicy>
class TensorView<T const, Extents, StoragePolicy, LayoutPolicy, AccessorPolicy> {
  public:
    /// \brief Element type exposed by the view.
    using element_type = T const;
    /// \brief Value type without cv-qualification.
    using value_type = T;
    /// \brief Extents type representing the tensor shape.
    using extents_type = Extents;
    /// \brief Policy controlling storage ownership semantics.
    using storage_policy = StoragePolicy;
    /// \brief Layout policy that governs multidimensional index ordering.
    using layout_policy = LayoutPolicy;
    /// \brief Policy providing accessors for the underlying handle.
    using accessor_policy = AccessorPolicy;

    /// \brief Mapping type derived from the layout policy and extents.
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    /// \brief Accessor type that provides mutable access to the view elements.
    using mutable_accessor_type = typename accessor_policy::template accessor_t<value_type>;
    /// \brief Accessor type providing const access semantics.
    using accessor_type = const_accessor_t<mutable_accessor_type>;
    /// \brief Data handle type exposed by the accessor.
    using handle_type = typename mutable_accessor_type::data_handle_type;
    /// \brief Mdspan specialisation used to present the tensor view.
    using mdspan_type = stdex::mdspan<element_type, extents_type, layout_policy, accessor_type>;
    /// \brief Index type used for addressing elements.
    using index_type = typename extents_type::index_type;
    /// \brief Size type representing the number of elements.
    using size_type = uni20::size_type;
    /// \brief Reference type returned by the accessor.
    using reference = typename accessor_type::reference;

    /// \brief Default-construct an empty view.
    TensorView() = default;

    /// \brief Construct from a handle, mapping, and accessor.
    /// \param handle Data handle referencing tensor elements.
    /// \param mapping Mapping that translates indices into offsets.
    /// \param accessor Accessor that interacts with the data handle.
    TensorView(handle_type handle, mapping_type mapping, accessor_type accessor = accessor_type{})
        : handle_{std::move(handle)}, mapping_{std::move(mapping)}, extents_{mapping_.extents()}, accessor_{std::move(
                                                                                                      accessor)}
    {}

    /// \brief Construct from a handle and extents, using the layout mapping's preferred form.
    /// \param handle Data handle referencing tensor elements.
    /// \param exts Extents that describe the tensor shape.
    /// \param accessor Accessor that interacts with the data handle.
    TensorView(handle_type handle, extents_type const& exts, accessor_type accessor = accessor_type{})
        : TensorView(std::move(handle), construct_mapping(exts), std::move(accessor))
    {}

    /// \brief Construct from a handle, extents, and explicit strides.
    /// \param handle Data handle referencing tensor elements.
    /// \param exts Extents that describe the tensor shape.
    /// \param strides Stride specification for each tensor dimension.
    /// \param accessor Accessor that interacts with the data handle.
    TensorView(handle_type handle, extents_type const& exts,
               std::array<index_type, extents_type::rank()> const& strides, accessor_type accessor = accessor_type{})
        : TensorView(std::move(handle), mapping_type{exts, strides}, std::move(accessor))
    {}

    /// \brief Access via bracket-operator: t[i0,i1,…].
    /// \tparam Idx Index pack specifying the coordinates to access.
    /// \param idxs Coordinate pack enumerating each dimension index.
    /// \return Reference to the tensor element at the requested coordinates.
    template <typename... Idx>
      requires(sizeof...(Idx) == extents_type::rank())
    reference operator[](Idx... idxs) const noexcept
    {
      return this->accessor_.access(this->handle_, this->mapping_(static_cast<index_type>(idxs)...));
    }

    /// \brief Retrieve the mdspan view (mapping plus accessor).
    /// \return Mdspan object describing the tensor view.
    mdspan_type mdspan() const noexcept { return mdspan_type(handle_, mapping_, accessor_); }

    /// \brief Rank of the tensor.
    /// \return Static rank reported by the extents type.
    static constexpr size_type rank() noexcept { return extents_type::rank(); }

    /// \brief Number of elements, equivalent to required_span_size().
    /// \return Total number of elements addressable by the view.
    size_type size() const noexcept { return mapping_.required_span_size(); }

    /// \brief Handle to the buffer.
    /// \return Data handle supplied by the accessor.
    handle_type handle() const noexcept { return handle_; }

    /// \brief The extents (shape).
    /// \return Extents that describe the tensor dimensions.
    extents_type const& extents() const noexcept { return extents_; }

    /// \brief The layout mapping (holds strides plus extents).
    /// \return Mapping instance that translates coordinates to offsets.
    mapping_type const& mapping() const noexcept { return mapping_; }

    /// \brief The accessor in use.
    /// \return Accessor associated with the tensor view.
    accessor_type const& accessor() const noexcept { return accessor_; }

  protected:
    /// \brief Element buffer handle, which may or may not be owned.
    [[no_unique_address]] handle_type handle_{};
    /// \brief Layout mapping that translates indices to offsets.
    [[no_unique_address]] mapping_type mapping_{};
    /// \brief Cached extents derived from the mapping.
    [[no_unique_address]] extents_type extents_{};
    /// \brief Accessor used for const-qualified element access.
    [[no_unique_address]] accessor_type accessor_{};
};

/// \brief Tensor view specialisation providing mutable access.
/// \ingroup tensor
/// \tparam T Value type stored in the tensor.
/// \tparam Extents Extents type describing the tensor shape.
/// \tparam StoragePolicy Policy that identifies how storage is owned or referenced.
/// \tparam LayoutPolicy Layout policy that maps indices to offsets.
/// \tparam AccessorPolicy Accessor policy used to interact with the storage handle.
template <typename T, typename Extents, typename StoragePolicy, typename LayoutPolicy, typename AccessorPolicy>
class TensorView : public TensorView<T const, Extents, StoragePolicy, LayoutPolicy, AccessorPolicy> {
  public:
    /// \brief Alias for the const-qualified base view implementation.
    /// \ingroup internal
    using base_type = TensorView<T const, Extents, StoragePolicy, LayoutPolicy, AccessorPolicy>;

    /// \brief Element type exposed by the view.
    using element_type = T;
    /// \brief Value type alias for compatibility with standard containers.
    using value_type = T;
    using typename base_type::accessor_policy;
    using typename base_type::extents_type;
    using typename base_type::handle_type;
    using typename base_type::index_type;
    using typename base_type::layout_policy;
    using typename base_type::mapping_type;
    using typename base_type::size_type;

    /// \brief Accessor type providing mutable access semantics.
    using accessor_type = typename accessor_policy::template accessor_t<element_type>;
    /// \brief Accessor type exposing const access semantics.
    using const_accessor_type = const_accessor_t<accessor_type>;
    /// \brief Mdspan specialisation exposing mutable references.
    using mdspan_type = stdex::mdspan<element_type, extents_type, layout_policy, accessor_type>;
    /// \brief Reference type returned for mutable element access.
    using reference = typename accessor_type::reference;

    /// \brief Default-construct an empty mutable view.
    TensorView() = default;

    /// \brief Construct from a handle, mapping, and mutable accessor.
    /// \param handle Data handle referencing tensor elements.
    /// \param mapping Mapping that translates indices into offsets.
    /// \param accessor Mutable accessor that interacts with the data handle.
    TensorView(handle_type handle, mapping_type mapping, accessor_type accessor = accessor_type{})
        : base_type(handle, std::move(mapping), const_accessor(accessor)), accessor_mut_{std::move(accessor)}
    {}

    /// \brief Construct from a handle and extents, using the layout mapping's preferred form.
    /// \param handle Data handle referencing tensor elements.
    /// \param exts Extents that describe the tensor shape.
    /// \param accessor Mutable accessor that interacts with the data handle.
    TensorView(handle_type handle, extents_type const& exts, accessor_type accessor = accessor_type{})
        : TensorView(std::move(handle), base_type::construct_mapping(exts), std::move(accessor))
    {}

    /// \brief Construct from a handle, extents, and explicit strides.
    /// \param handle Data handle referencing tensor elements.
    /// \param exts Extents that describe the tensor shape.
    /// \param strides Stride specification for each tensor dimension.
    /// \param accessor Mutable accessor that interacts with the data handle.
    TensorView(handle_type handle, extents_type const& exts,
               std::array<index_type, extents_type::rank()> const& strides, accessor_type accessor = accessor_type{})
        : base_type(handle, mapping_type{exts, strides}, const_accessor(accessor)), accessor_mut_{std::move(accessor)}
    {}

    /// \brief Mutable bracket access for tensor elements.
    /// \tparam Idx Index pack specifying the coordinates to access.
    /// \param idxs Coordinate pack enumerating each dimension index.
    /// \return Mutable reference to the tensor element at the requested coordinates.
    template <typename... Idx>
      requires(sizeof...(Idx) == extents_type::rank())
    reference operator[](Idx... idxs) noexcept
    {
      return accessor_mut_.access(this->handle_, this->mapping_(static_cast<index_type>(idxs)...));
    }

    using base_type::operator[];

    /// \brief Retrieve a mutable mdspan view (const overload).
    /// \return Mdspan object that permits mutation through the accessor.
    mdspan_type mdspan() const noexcept { return mdspan_type(this->handle_, this->mapping_, accessor_mut_); }

    /// \brief Retrieve a mutable mdspan view (non-const overload).
    /// \return Mdspan object that permits mutation through the accessor.
    mdspan_type mutable_mdspan() noexcept { return mdspan_type(this->handle_, this->mapping_, accessor_mut_); }

    /// \brief Mutable accessor in use by the tensor view.
    /// \return Accessor that provides mutable access semantics.
    accessor_type const& accessor() const noexcept { return accessor_mut_; }

  private:
    // We need a mutable accessor. This is a pain - if it not stateless, then it is likely to contain
    // the same information as the TensorView<T const> version, but we cannot reuse it because it has
    // the wrong return type of .access().
    [[no_unique_address]] accessor_type accessor_mut_{};
};

} // namespace uni20
