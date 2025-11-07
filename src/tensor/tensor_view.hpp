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
#include <concepts>
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

/// \brief Trait bundle describing the policies required to build tensor views.
/// \ingroup tensor
/// \tparam Extents Extents type describing the tensor shape.
/// \tparam StoragePolicy Policy that identifies how storage is owned or referenced.
/// \tparam LayoutPolicy Layout policy that maps indices to offsets.
/// \tparam AccessorPolicy Accessor policy used to interact with the storage handle.
template <typename Extents, typename StoragePolicy = VectorStorage, typename LayoutPolicy = stdex::layout_stride,
          typename AccessorPolicy = DefaultAccessorFactory>
struct tensor_traits
{
    /// \brief Extents type representing the tensor shape.
    using extents_type = Extents;
    /// \brief Policy controlling storage ownership semantics.
    using storage_policy = StoragePolicy;
    /// \brief Layout policy that governs multidimensional index ordering.
    using layout_policy = LayoutPolicy;
    /// \brief Policy providing accessors for the underlying handle.
    using accessor_policy = AccessorPolicy;
};

/// \brief Forward declaration for the TensorView template using bundled traits.
/// \ingroup tensor
/// \tparam ElementType Value type viewed by the tensor.
/// \tparam Traits Trait bundle describing policies and extents.
template <typename ElementType, typename Traits>
class TensorView;

/// \brief Tensor view specialisation for const-qualified element access.
/// \ingroup tensor
/// \tparam T Base value type without const-qualification.
/// \tparam Traits Trait bundle describing policies and extents.
template <typename T, typename Traits>
class TensorView<T const, Traits>
{
  public:
    /// \brief Trait bundle type used by the tensor view.
    using traits_type = Traits;
    /// \brief Element type exposed by the view.
    using element_type = T const;
    /// \brief Value type without cv-qualification.
    using value_type = T;
    /// \brief Extents type representing the tensor shape.
    using extents_type = typename traits_type::extents_type;
    /// \brief Policy controlling storage ownership semantics.
    using storage_policy = typename traits_type::storage_policy;
    /// \brief Layout policy that governs multidimensional index ordering.
    using layout_policy = typename traits_type::layout_policy;
    /// \brief Policy providing accessors for the underlying handle.
    using accessor_policy = typename traits_type::accessor_policy;

    /// \brief Mapping type derived from the layout policy and extents.
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    /// \brief Accessor type that provides mutable access to the view elements.
    using mutable_accessor_type = typename accessor_policy::template accessor_t<value_type>;
    /// \brief Accessor type providing const access semantics.
    using const_accessor_type = typename accessor_policy::template accessor_t<element_type>;
    /// \brief Mutable handle type exposed by the accessor.
    using mutable_handle_type = typename mutable_accessor_type::data_handle_type;
    /// \brief Data handle type exposed by the const accessor.
    using handle_type = typename const_accessor_type::data_handle_type;
    /// \brief Mdspan specialisation used to present the tensor view.
    using mdspan_type = stdex::mdspan<element_type, extents_type, layout_policy, const_accessor_type>;
    /// \brief Index type used for addressing elements.
    using index_type = typename extents_type::index_type;
    /// \brief Size type representing the number of elements.
    using size_type = uni20::size_type;
    /// \brief Reference type returned by the accessor.
    using reference = typename const_accessor_type::reference;

    /// \brief Default-construct an empty view.
    TensorView() = default;

    /// \brief Construct from a handle, mapping, and accessor.
    /// \tparam Handle Handle type convertible to the const handle type.
    /// \param handle Data handle referencing tensor elements.
    /// \param mapping Mapping that translates indices into offsets.
    /// \param accessor Accessor that interacts with the data handle.
    template <typename Handle>
      requires std::convertible_to<Handle, handle_type>
    TensorView(Handle&& handle, mapping_type mapping, const_accessor_type accessor = const_accessor_type{})
        : handle_(static_cast<handle_type>(std::forward<Handle>(handle))),
          mapping_(std::move(mapping)),
          extents_(mapping_.extents()),
          accessor_(std::move(accessor))
    {}

    /// \brief Construct from a handle and extents, using the layout mapping's preferred form.
    /// \tparam Handle Handle type convertible to the const handle type.
    /// \param handle Data handle referencing tensor elements.
    /// \param exts Extents that describe the tensor shape.
    /// \param accessor Accessor that interacts with the data handle.
    template <typename Handle>
      requires std::convertible_to<Handle, handle_type>
    TensorView(Handle&& handle, extents_type const& exts, const_accessor_type accessor = const_accessor_type{})
        : TensorView(std::forward<Handle>(handle), construct_mapping(exts), std::move(accessor))
    {}

    /// \brief Construct from a handle, extents, and explicit strides.
    /// \tparam Handle Handle type convertible to the const handle type.
    /// \param handle Data handle referencing tensor elements.
    /// \param exts Extents that describe the tensor shape.
    /// \param strides Stride specification for each tensor dimension.
    /// \param accessor Accessor that interacts with the data handle.
    template <typename Handle>
      requires std::convertible_to<Handle, handle_type>
    TensorView(Handle&& handle, extents_type const& exts,
               std::array<index_type, extents_type::rank()> const& strides,
               const_accessor_type accessor = const_accessor_type{})
        : TensorView(std::forward<Handle>(handle), mapping_type{exts, strides}, std::move(accessor))
    {}

    /// \brief Access via bracket-operator: t[i0,i1,…].
    /// \tparam Idx Index pack specifying the coordinates to access.
    /// \param idxs Coordinate pack enumerating each dimension index.
    /// \return Reference to the tensor element at the requested coordinates.
    template <typename... Idx>
      requires(sizeof...(Idx) == extents_type::rank())
    reference operator[](Idx... idxs) const noexcept
    {
      return accessor_.access(handle_, mapping_(static_cast<index_type>(idxs)...));
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
    const_accessor_type const& accessor() const noexcept { return accessor_; }

  protected:
    /// \brief Construct a mapping from the supplied extents using the layout policy defaults.
    /// \param exts Extents that describe the tensor shape.
    /// \return Mapping instance that translates coordinates to offsets.
    static constexpr auto construct_mapping(extents_type const& exts) -> mapping_type
    {
      return mapping_type{exts};
    }

    /// \brief Element buffer handle, which may or may not be owned.
    [[no_unique_address]] handle_type handle_{};
    /// \brief Layout mapping that translates indices to offsets.
    [[no_unique_address]] mapping_type mapping_{};
    /// \brief Cached extents derived from the mapping.
    [[no_unique_address]] extents_type extents_{};
    /// \brief Accessor used for const-qualified element access.
    [[no_unique_address]] const_accessor_type accessor_{};
};

/// \brief Tensor view specialisation providing mutable access.
/// \ingroup tensor
/// \tparam T Value type stored in the tensor.
/// \tparam Traits Trait bundle describing policies and extents.
template <typename T, typename Traits>
class TensorView : public TensorView<T const, Traits>
{
  public:
    /// \brief Alias for the const-qualified base view implementation.
    /// \ingroup internal
    using base_type = TensorView<T const, Traits>;

    /// \brief Element type exposed by the view.
    using element_type = T;
    /// \brief Value type alias for compatibility with standard containers.
    using value_type = T;
    using typename base_type::accessor_policy;
    using typename base_type::const_accessor_type;
    using typename base_type::extents_type;
    using typename base_type::index_type;
    using typename base_type::layout_policy;
    using typename base_type::mapping_type;
    using typename base_type::mutable_accessor_type;
    using typename base_type::mutable_handle_type;
    using typename base_type::size_type;

    /// \brief Accessor type providing mutable access semantics.
    using accessor_type = mutable_accessor_type;
    /// \brief Mdspan specialisation exposing mutable references.
    using mdspan_type = stdex::mdspan<element_type, extents_type, layout_policy, accessor_type>;
    /// \brief Reference type returned for mutable element access.
    using reference = typename accessor_type::reference;
    /// \brief Mutable handle alias for compatibility with previous APIs.
    using handle_type = mutable_handle_type;
    /// \brief Const-qualified handle type exposed by the base implementation.
    using const_handle_type = typename base_type::handle_type;

    /// \brief Default-construct an empty mutable view.
    TensorView() = default;

    /// \brief Construct from a handle, mapping, and mutable accessor.
    /// \tparam Handle Handle type convertible to the mutable handle type.
    /// \param handle Data handle referencing tensor elements.
    /// \param mapping Mapping that translates indices into offsets.
    /// \param accessor Mutable accessor that interacts with the data handle.
    template <typename Handle>
      requires std::convertible_to<Handle, handle_type>
    TensorView(Handle&& handle, mapping_type mapping, accessor_type accessor = accessor_type{})
        : base_type(static_cast<const_handle_type>(handle), std::move(mapping),
                    make_const_accessor(accessor)),
          accessor_mut_{std::move(accessor)},
          mutable_handle_{std::forward<Handle>(handle)}
    {}

    /// \brief Construct from a handle and extents, using the layout mapping's preferred form.
    /// \tparam Handle Handle type convertible to the mutable handle type.
    /// \param handle Data handle referencing tensor elements.
    /// \param exts Extents that describe the tensor shape.
    /// \param accessor Mutable accessor that interacts with the data handle.
    template <typename Handle>
      requires std::convertible_to<Handle, handle_type>
    TensorView(Handle&& handle, extents_type const& exts, accessor_type accessor = accessor_type{})
        : TensorView(std::forward<Handle>(handle), base_type::construct_mapping(exts), std::move(accessor))
    {}

    /// \brief Construct from a handle, extents, and explicit strides.
    /// \tparam Handle Handle type convertible to the mutable handle type.
    /// \param handle Data handle referencing tensor elements.
    /// \param exts Extents that describe the tensor shape.
    /// \param strides Stride specification for each tensor dimension.
    /// \param accessor Mutable accessor that interacts with the data handle.
    template <typename Handle>
      requires std::convertible_to<Handle, handle_type>
    TensorView(Handle&& handle, extents_type const& exts,
               std::array<index_type, extents_type::rank()> const& strides,
               accessor_type accessor = accessor_type{})
        : TensorView(std::forward<Handle>(handle), mapping_type{exts, strides}, std::move(accessor))
    {}

    /// \brief Mutable bracket access for tensor elements.
    /// \tparam Idx Index pack specifying the coordinates to access.
    /// \param idxs Coordinate pack enumerating each dimension index.
    /// \return Mutable reference to the tensor element at the requested coordinates.
    template <typename... Idx>
      requires(sizeof...(Idx) == extents_type::rank())
    reference operator[](Idx... idxs) noexcept
    {
      return accessor_mut_.access(mutable_handle_, this->mapping_(static_cast<index_type>(idxs)...));
    }

    using base_type::operator[];

    using base_type::mdspan;

    /// \brief Retrieve a mutable mdspan view (non-const overload).
    /// \return Mdspan object that permits mutation through the accessor.
    mdspan_type mdspan() noexcept { return mdspan_type(mutable_handle_, this->mapping_, accessor_mut_); }

    /// \brief Retrieve a mutable mdspan view (non-const alias).
    /// \return Mdspan object that permits mutation through the accessor.
    mdspan_type mutable_mdspan() noexcept { return mdspan(); }

    /// \brief Mutable accessor in use by the tensor view.
    /// \return Accessor that provides mutable access semantics.
    accessor_type const& accessor() const noexcept { return accessor_mut_; }

    /// \brief Handle with mutable access semantics.
    /// \return Data handle that permits mutation of the underlying storage.
    handle_type mutable_handle() noexcept { return mutable_handle_; }

  private:
    [[no_unique_address]] accessor_type accessor_mut_{};
    [[no_unique_address]] handle_type mutable_handle_{};

    static constexpr auto make_const_accessor(accessor_type const& accessor) -> const_accessor_type
    {
      auto const converted = uni20::const_accessor(accessor);
      static_assert(std::is_convertible_v<decltype(converted), const_accessor_type>,
                    "Mutable accessor must be convertible to const accessor type.");
      return const_accessor_type{converted};
    }
};

} // namespace uni20
