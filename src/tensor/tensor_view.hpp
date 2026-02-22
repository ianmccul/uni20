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
#include "layout.hpp"
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
    /// \brief Adaptor to provide the actual storage type of the given ElementType
    template <typename ElementType> using storage_type = ElementType;
};

template <typename Extents, typename StoragePolicy = VectorStorage, typename LayoutPolicy = stdex::layout_stride,
          typename AccessorPolicy = DefaultAccessorFactory>
struct mutable_tensor_traits : tensor_traits<Extents, StoragePolicy, LayoutPolicy, AccessorPolicy>
{
    /// \brief Adaptor to provide the actual storage type of the given ElementType for mutable access patterns.
    template <typename ElementType> using storage_type = std::remove_cv_t<ElementType>;
};

/// \brief Trait bundle describing the policies required to build tensor views.
/// \ingroup tensor
/// \tparam Extents Extents type describing the tensor shape.
/// \tparam StoragePolicy Policy that identifies how storage is owned or referenced.
/// \tparam LayoutPolicy Layout policy that maps indices to offsets.
/// \tparam AccessorPolicy Accessor policy used to interact with the storage handle.
/// \brief Forward declaration for the TensorView template using bundled traits.
/// \ingroup tensor
/// \tparam ElementType Value type viewed by the tensor.
/// \tparam Traits Trait bundle describing policies and extents.
template <typename ElementType, typename Traits> class TensorView;

/// \brief Tensor view specialisation for const-qualified element access.
/// \ingroup tensor
/// \tparam T Base value type without const-qualification.
/// \tparam Traits Trait bundle describing policies and extents.
template <typename T, typename Traits> class TensorView<T const, Traits> {
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
    /// \brief Backend tag associated with the default storage policy.
    using default_tag = typename storage_policy::default_tag;
    /// \brief Layout policy that governs multidimensional index ordering.
    using layout_policy = typename traits_type::layout_policy;
    /// \brief Policy providing accessors for the underlying handle.
    using accessor_policy = typename traits_type::accessor_policy;

  protected:
    /// \brief The data type stored in the underlying buffer.
    using storage_type = typename traits_type::template storage_type<std::remove_const_t<value_type>>;

  public:
    /// \brief Mapping type derived from the layout policy and extents.
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    /// \brief Accessor type providing mutable access semantics.
    using mutable_accessor_type = typename accessor_policy::template accessor_t<storage_type>;
    /// \brief Accessor type providing const access semantics.
    using accessor_type = typename accessor_policy::template accessor_t<element_type>;
    /// \brief Mutable handle type exposed by the accessor.
    using mutable_handle_type = typename mutable_accessor_type::data_handle_type;
    /// \brief Data handle type exposed by the const accessor.
    using handle_type = typename accessor_type::data_handle_type;
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
    /// \tparam Handle Handle type convertible to the mutable handle type.
    /// \tparam Accessor Accessor type convertible to the mutable accessor type.
    /// \param handle Data handle referencing tensor elements.
    /// \param mapping Mapping that translates indices into offsets.
    /// \param accessor Accessor that interacts with the data handle.
    template <typename Handle, typename Accessor = mutable_accessor_type>
    requires((std::convertible_to<Handle, mutable_handle_type> ||
              std::convertible_to<Handle, handle_type>)&&std::convertible_to<Accessor, mutable_accessor_type>)
        TensorView(Handle&& handle, mapping_type mapping, Accessor&& accessor = Accessor{})
        : handle_(to_mutable_handle(std::forward<Handle>(handle))), mapping_(std::move(mapping)),
          extents_(mapping_.extents()), accessor_(std::forward<Accessor>(accessor))
    {}

    /// \brief Construct from a handle and extents, using the layout mapping's preferred form.
    /// \tparam Handle Handle type convertible to the mutable handle type.
    /// \tparam Accessor Accessor type convertible to the mutable accessor type.
    /// \param handle Data handle referencing tensor elements.
    /// \param exts Extents that describe the tensor shape.
    /// \param accessor Accessor that interacts with the data handle.
    template <typename Handle, typename Accessor = mutable_accessor_type>
    requires((std::convertible_to<Handle, mutable_handle_type> ||
              std::convertible_to<Handle, handle_type>)&&std::convertible_to<Accessor, mutable_accessor_type>)
        TensorView(Handle&& handle, extents_type const& exts, Accessor&& accessor = Accessor{})
        : TensorView(std::forward<Handle>(handle), layout::make_mapping<layout_policy>(exts),
                     std::forward<Accessor>(accessor))
    {}

    /// \brief Construct from a handle, extents, and explicit strides.
    /// \tparam Handle Handle type convertible to the mutable handle type.
    /// \tparam Accessor Accessor type convertible to the mutable accessor type.
    /// \param handle Data handle referencing tensor elements.
    /// \param exts Extents that describe the tensor shape.
    /// \param strides Stride specification for each tensor dimension.
    /// \param accessor Accessor that interacts with the data handle.
    template <typename Handle, typename Accessor = mutable_accessor_type>
    requires((std::convertible_to<Handle, mutable_handle_type> ||
              std::convertible_to<Handle, handle_type>)&&std::convertible_to<Accessor, mutable_accessor_type>)
        TensorView(Handle&& handle, extents_type const& exts,
                   std::array<index_type, extents_type::rank()> const& strides, Accessor&& accessor = Accessor{})
        : TensorView(std::forward<Handle>(handle), mapping_type{exts, strides}, std::forward<Accessor>(accessor))
    {}

    /// \brief Access via bracket-operator: t[i0,i1,…].
    /// \tparam Idx Index pack specifying the coordinates to access.
    /// \param idxs Coordinate pack enumerating each dimension index.
    /// \return Reference to the tensor element at the requested coordinates.
    template <typename... Idx>
    requires(sizeof...(Idx) == extents_type::rank()) reference operator[](Idx... idxs) const noexcept
    {
      return accessor_.access(handle_, mapping_(static_cast<index_type>(idxs)...));
    }

    /// \brief Retrieve the mdspan view (mapping plus accessor).
    /// \return Mdspan object describing the tensor view.
    auto mdspan() const noexcept -> mdspan_type { return mdspan_type(handle(), mapping_, accessor()); }

    /// \brief Rank of the tensor.
    /// \return Static rank reported by the extents type.
    static constexpr size_type rank() noexcept { return extents_type::rank(); }

    /// \brief Number of elements, equivalent to required_span_size().
    /// \return Total number of elements addressable by the view.
    size_type size() const noexcept { return mapping_.required_span_size(); }

    /// \brief Handle to the buffer.
    /// \return Data handle supplied by the accessor, with const qualifications applied.
    auto handle() const noexcept -> handle_type
    {
      static_assert(std::convertible_to<mutable_handle_type, handle_type>,
                    "Mutable handle must convert to const handle type.");
      return static_cast<handle_type>(handle_);
    }

    /// \brief The extents (shape).
    /// \return Extents that describe the tensor dimensions.
    auto extents() const noexcept -> extents_type const& { return extents_; }

    /// \brief Number of matrix rows for rank-2 tensor views.
    /// \return Count of the first extent when the tensor models a matrix.
    auto rows() const noexcept requires(extents_type::rank() == 2) { return extents().extent(0); }

    /// \brief Number of matrix columns for rank-2 tensor views.
    /// \return Count of the second extent when the tensor models a matrix.
    auto cols() const noexcept requires(extents_type::rank() == 2) { return extents().extent(1); }

    /// \brief The layout mapping (holds strides plus extents).
    /// \return Mapping instance that translates coordinates to offsets.
    auto mapping() const noexcept -> mapping_type const& { return mapping_; }

    /// \brief The accessor object with const semantics applied.
    /// \return Accessor associated with the tensor view.
    auto accessor() const noexcept -> accessor_type
    {
      if constexpr (std::is_same_v<mutable_accessor_type, accessor_type>)
      {
        return accessor_;
      }
      else
      {
        return accessor_type(accessor_);
      }
    }

  protected:
    template <typename Handle> static constexpr auto to_mutable_handle(Handle&& handle) -> mutable_handle_type
    {
      if constexpr (std::convertible_to<Handle, mutable_handle_type>)
      {
        return static_cast<mutable_handle_type>(std::forward<Handle>(handle));
      }
      else
      {
        static_assert(std::convertible_to<Handle, handle_type>,
                      "Handle must be convertible to either the mutable or const handle type.");
        if constexpr (std::is_pointer_v<mutable_handle_type> && std::is_pointer_v<handle_type>)
        {
          auto const_handle = static_cast<handle_type>(std::forward<Handle>(handle));
          using pointed_type = std::remove_pointer_t<mutable_handle_type>;
          return const_cast<pointed_type*>(const_handle);
        }
        else
        {
          static_assert(std::convertible_to<Handle, mutable_handle_type>,
                        "Non-pointer handles must be directly convertible to the mutable handle type.");
          return static_cast<mutable_handle_type>(std::forward<Handle>(handle));
        }
      }
    }

    /// \brief Mutable handle storage accessible to derived classes.
    auto mutable_handle_ref() noexcept -> mutable_handle_type& { return handle_; }
    /// \brief Mutable handle storage accessible in const contexts for derived classes.
    auto mutable_handle_ref() const noexcept -> mutable_handle_type const& { return handle_; }

    /// \brief Mutable accessor storage accessible to derived classes.
    auto mutable_accessor_ref() noexcept -> mutable_accessor_type& { return accessor_; }
    /// \brief Mutable accessor storage accessible in const contexts for derived classes.
    auto mutable_accessor_ref() const noexcept -> mutable_accessor_type const& { return accessor_; }

    [[no_unique_address]] mutable_handle_type handle_{};
    [[no_unique_address]] mapping_type mapping_{};
    [[no_unique_address]] extents_type extents_{};
    [[no_unique_address]] mutable_accessor_type accessor_{};
};

/// \brief Tensor view specialisation providing mutable access.
/// \ingroup tensor
/// \tparam T Value type stored in the tensor.
/// \tparam Traits Trait bundle describing policies and extents.
template <typename T, typename Traits> class TensorView : public TensorView<T const, Traits> {
  public:
    /// \brief Alias for the const-qualified base view implementation.
    using base_type = TensorView<T const, Traits>;
    /// \brief Trait bundle type used by the tensor view.
    using traits_type = Traits;
    /// \brief Element type exposed by the view.
    using element_type = T;
    /// \brief Value type without cv-qualification.
    using value_type = T;
    /// \brief Extents type representing the tensor shape.
    using extents_type = typename traits_type::extents_type;
    /// \brief Policy controlling storage ownership semantics.
    using storage_policy = typename traits_type::storage_policy;
    /// \brief Backend tag associated with the default storage policy.
    using default_tag = typename storage_policy::default_tag;
    /// \brief Layout policy that governs multidimensional index ordering.
    using layout_policy = typename traits_type::layout_policy;
    /// \brief Policy providing accessors for the underlying handle.
    using accessor_policy = typename traits_type::accessor_policy;

    using typename base_type::mutable_accessor_type;
    using typename base_type::mutable_handle_type;

  public:
    /// \brief Mapping type derived from the layout policy and extents.
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    /// \brief Accessor type providing mutable access semantics.
    using accessor_type = typename accessor_policy::template accessor_t<element_type>;
    /// \brief Data handle type exposed by the mutable accessor.
    using handle_type = typename accessor_type::data_handle_type;
    /// \brief Mdspan specialisation used to present the tensor view.
    using mdspan_type = stdex::mdspan<element_type, extents_type, layout_policy, accessor_type>;
    /// \brief Reference type returned by the accessor.
    using reference = typename accessor_type::reference;

    /// \brief Default-construct an empty mutable view.
    TensorView() = default;

    /// \brief Construct from a handle, mapping, and mutable accessor.
    /// \tparam Handle Handle type convertible to the mutable handle type.
    /// \tparam Accessor Accessor type convertible to the mutable accessor type.
    /// \param handle Data handle referencing tensor elements.
    /// \param mapping Mapping that translates indices into offsets.
    /// \param accessor Mutable accessor that interacts with the data handle.
    template <typename Handle, typename Accessor = accessor_type>
    requires(std::convertible_to<Handle, mutable_handle_type>&& std::convertible_to<Accessor, accessor_type>)
        TensorView(Handle&& handle, mapping_type mapping, Accessor&& accessor = Accessor{})
        : base_type(std::forward<Handle>(handle), std::move(mapping), std::forward<Accessor>(accessor))
    {}

    /// \brief Construct from a handle and extents, using the layout mapping's preferred form.
    /// \tparam Handle Handle type convertible to the mutable handle type.
    /// \tparam Accessor Accessor type convertible to the mutable accessor type.
    /// \param handle Data handle referencing tensor elements.
    /// \param exts Extents that describe the tensor shape.
    /// \param accessor Mutable accessor that interacts with the data handle.
    template <typename Handle, typename Accessor = accessor_type>
    requires(std::convertible_to<Handle, mutable_handle_type>&& std::convertible_to<Accessor, accessor_type>)
        TensorView(Handle&& handle, extents_type const& exts, Accessor&& accessor = Accessor{})
        : TensorView(std::forward<Handle>(handle), layout::make_mapping<layout_policy>(exts),
                     std::forward<Accessor>(accessor))
    {}

    /// \brief Construct from a handle, extents, and explicit strides.
    /// \tparam Handle Handle type convertible to the mutable handle type.
    /// \tparam Accessor Accessor type convertible to the mutable accessor type.
    /// \param handle Data handle referencing tensor elements.
    /// \param exts Extents that describe the tensor shape.
    /// \param strides Stride specification for each tensor dimension.
    /// \param accessor Mutable accessor that interacts with the data handle.
    template <typename Handle, typename Accessor = accessor_type>
    requires(std::convertible_to<Handle, mutable_handle_type>&& std::convertible_to<Accessor, accessor_type>)
        TensorView(Handle&& handle, extents_type const& exts,
                   std::array<typename base_type::index_type, extents_type::rank()> const& strides,
                   Accessor&& accessor = Accessor{})
        : TensorView(std::forward<Handle>(handle), mapping_type{exts, strides}, std::forward<Accessor>(accessor))
    {}

    /// \brief Mutable bracket access for tensor elements.
    /// \tparam Idx Index pack specifying the coordinates to access.
    /// \param idxs Coordinate pack enumerating each dimension index.
    /// \return Mutable reference to the tensor element at the requested coordinates.
    template <typename... Idx>
    requires(sizeof...(Idx) == extents_type::rank()) reference operator[](Idx... idxs) noexcept
    {
      return this->mutable_accessor_ref().access(this->mutable_handle_ref(),
                                                 this->mapping_(static_cast<typename base_type::index_type>(idxs)...));
    }

    using base_type::operator[];

    /// \brief Retrieve the mdspan view exposed by the mutable accessor.
    /// \return Mdspan object providing mutable access to the tensor elements.
    auto mutable_mdspan() noexcept -> mdspan_type
    {
      return mdspan_type(this->mutable_handle_ref(), this->mapping_, this->mutable_accessor_ref());
    }

    /// \brief Provide mutable access to the underlying handle.
    /// \return Mutable data handle supplied by the accessor.
    auto mutable_handle() noexcept -> mutable_handle_type { return this->mutable_handle_ref(); }

    /// \brief Retrieve the mutable accessor in use by the tensor view.
    /// \return Accessor that provides mutable access semantics.
    auto accessor() noexcept -> accessor_type& { return this->mutable_accessor_ref(); }

    /// \brief Retrieve the mutable accessor in const contexts.
    /// \return Accessor providing mutable semantics, exposed as a const reference.
    auto accessor() const noexcept -> accessor_type const& { return this->mutable_accessor_ref(); }

    using base_type::cols;
    using base_type::rows;
};

} // namespace uni20
