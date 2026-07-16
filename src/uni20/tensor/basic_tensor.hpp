#pragma once

/**
 * \file basic_tensor.hpp
 * \ingroup tensor
 * \brief Owning tensor implementation and configurable extents alias.
 */

#include "concepts.hpp"
#include "layout.hpp"

#include <uni20/common/trace.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/storage/vectorstorage.hpp>
#include <uni20/tensor/copy_into.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <class RequestedLayout, class InputMdspan>
using materialized_layout_t =
    std::conditional_t<std::is_void_v<RequestedLayout>,
                       std::conditional_t<std::same_as<typename InputMdspan::layout_type, stdex::layout_left> ||
                                              std::same_as<typename InputMdspan::layout_type, stdex::layout_right>,
                                          typename InputMdspan::layout_type, stdex::layout_left>,
                       RequestedLayout>;

} // namespace detail

/// \brief Factory that provides default accessors for tensor storage containers.
struct DefaultAccessorFactory
{
    template <typename ElementType> using accessor_t = stdex::default_accessor<ElementType>;

    template <typename ElementType, typename Storage>
    [[nodiscard]] constexpr auto make_accessor(Storage const&) const noexcept -> accessor_t<ElementType>
    {
      return accessor_t<ElementType>{};
    }
};

/// \brief General-purpose owning tensor that exposes mdspan-based access.
/// \ingroup tensor
/// \tparam ElementType Value type stored by the tensor.
/// \tparam Rank Static rank of the tensor.
/// \tparam StoragePolicy Policy controlling ownership and allocation of the buffer.
/// \tparam LayoutPolicy Layout policy that determines index ordering and stride computation.
/// \tparam AccessorFactory Factory that produces accessors for the storage handle.
/// \tparam Extents Extents type describing the tensor shape; fully dynamic by default.
template <typename ElementType, std::size_t Rank, typename StoragePolicy = VectorStorage,
          typename LayoutPolicy = stdex::layout_left, typename AccessorFactory = DefaultAccessorFactory,
          typename Extents = stdex::dextents<index_type, Rank>>
class Tensor {
  public:
    static_assert(Rank == Extents::rank());

    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using storage_policy = StoragePolicy;
    using layout_policy = LayoutPolicy;
    using layout_type = layout_policy;
    using accessor_factory_type = AccessorFactory;
    using accessor_type = typename accessor_factory_type::template accessor_t<element_type>;
    using const_accessor_type = typename accessor_factory_type::template accessor_t<element_type const>;
    using extents_type = Extents;
    using handle_type = typename const_accessor_type::data_handle_type;
    using mutable_handle_type = typename accessor_type::data_handle_type;
    using index_type = typename extents_type::index_type;
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    using size_type = uni20::size_type;
    using backend_selector_type = typename storage_policy::backend_selector_type;
    using mdspan_type = stdex::mdspan<element_type, extents_type, layout_policy, accessor_type>;
    using const_mdspan_type = stdex::mdspan<element_type const, extents_type, layout_policy, const_accessor_type>;

    using storage_type = typename storage_policy::template storage_t<element_type>;

    /// \brief Rebind this owning tensor configuration to another layout policy.
    template <typename NewLayoutPolicy>
    using rebind_layout_type =
        Tensor<element_type, Rank, storage_policy, NewLayoutPolicy, accessor_factory_type, extents_type>;

    /// \brief Rebind this owning tensor configuration to another extents type.
    template <typename NewExtents>
    using rebind_extents_type =
        Tensor<element_type, NewExtents::rank(), storage_policy, layout_policy, accessor_factory_type, NewExtents>;

    /// \brief Default-construct an empty tensor without allocated storage.
    Tensor() = default;

    /// \brief Copy-construct a tensor with independent owned storage.
    /// \details Tensor elements, mapping, and accessor-factory state are copied.
    ///          Resolved mdspans are constructed on demand from this tensor's
    ///          own storage.
    /// \param other Source tensor to copy.
    Tensor(Tensor const& other) = default;

    /// \brief Move-construct an owning tensor.
    /// \details Existing non-owning views and mdspans into the source must not
    ///          be used after ownership is transferred.
    /// \param other Source tensor to move from.
    Tensor(Tensor&& other) = default;

    /// \brief Copy-assign tensor storage and descriptor state.
    /// \param other Source tensor to copy.
    /// \return Reference to `*this`.
    Tensor& operator=(Tensor const& other) = default;

    /// \brief Move-assign tensor storage and descriptor state.
    /// \details Existing non-owning views and mdspans into either tensor must
    ///          not be used after ownership is transferred.
    /// \param other Source tensor to move from.
    /// \return Reference to `*this`.
    Tensor& operator=(Tensor&& other) = default;

    /// \brief Materialize an independent owning tensor from a readable tensor view.
    /// \details The destination's scalar, extents, storage, layout, and accessor
    ///          policies are fixed by this specialization. Values are copied
    ///          through the ordinary backend-dispatch path.
    /// \tparam InputTensor Readable tensor-level source with matching static rank.
    /// \param input Source tensor view whose values are materialized.
    template <TensorView InputTensor>
      requires(tensor_mdspan_t<InputTensor>::rank() == extents_type::rank() &&
               std::default_initializable<accessor_factory_type>)
    explicit Tensor(InputTensor const& input) : Tensor(detail::convert_tensor_extents<extents_type>(input.extents()))
    {
      copy(*this, input);
    }

    /// \brief Construct a tensor with default layout and accessor factory.
    /// \param exts Extents that describe the tensor shape.
    /// \param accessor_factory Factory used to create the accessor for the storage handle.
    explicit Tensor(extents_type const& exts, accessor_factory_type accessor_factory = accessor_factory_type{})
        : Tensor(internal_tag{}, make_payload(make_default_mapping(exts), std::move(accessor_factory)))
    {}

    /// \brief Construct a fully dynamic tensor from one extent per axis.
    /// \details This convenience form is available when every extent is
    ///          dynamic and keeps rank-specific aliases ergonomic without
    ///          adding wrapper classes solely for constructors.
    /// \tparam DynamicExtents Integral extent arguments, one per tensor axis.
    /// \param dynamic_extents Tensor extents in axis order.
    template <std::integral... DynamicExtents>
      requires(extents_type::rank() > 0 && extents_type::rank_dynamic() == extents_type::rank() &&
               sizeof...(DynamicExtents) == extents_type::rank())
    explicit Tensor(DynamicExtents... dynamic_extents)
        : Tensor(extents_type{static_cast<index_type>(dynamic_extents)...})
    {}

    /// \brief Construct a tensor using a custom mapping builder.
    /// \tparam MappingBuilder Callable that returns a mapping compatible with the layout policy.
    /// \param exts Extents that describe the tensor shape.
    /// \param mapping_builder Builder used to derive the mapping from the extents.
    /// \param accessor_factory Factory used to create the accessor for the storage handle.
    template <typename MappingBuilder>
      requires(layout::mapping_builder_for<MappingBuilder, layout_policy, extents_type> &&
               (!std::same_as<std::remove_cvref_t<MappingBuilder>, accessor_factory_type>))
    explicit Tensor(extents_type const& exts, MappingBuilder&& mapping_builder,
                    accessor_factory_type accessor_factory = accessor_factory_type{})
        : Tensor(internal_tag{},
                 make_payload(std::forward<MappingBuilder>(mapping_builder)(exts), std::move(accessor_factory)))
    {}

    /// \brief Construct a tensor from explicit extents and strides.
    /// \param exts Extents that describe the tensor shape.
    /// \param strides Stride specification per dimension for the layout mapping.
    /// \param accessor_factory Factory used to create the accessor for the storage handle.
    explicit Tensor(extents_type const& exts, std::array<index_type, extents_type::rank()> const& strides,
                    accessor_factory_type accessor_factory = accessor_factory_type{})
      requires std::constructible_from<mapping_type, extents_type const&,
                                       std::array<index_type, extents_type::rank()> const&>
        : Tensor(internal_tag{}, make_payload(mapping_type{exts, strides}, std::move(accessor_factory)))
    {}

    /// \brief Replace the tensor shape and discard its current values.
    /// \details The replacement uses the storage policy's default mapping for
    ///          the new extents and preserves accessor-factory state. The
    ///          replacement is constructed before the current tensor changes;
    ///          supported tensor state must be nothrow-swappable so the update
    ///          provides the strong exception guarantee.
    /// \param exts New tensor extents.
    void reset_shape(extents_type const& exts)
      requires(std::copy_constructible<accessor_factory_type> && std::is_nothrow_swappable_v<mapping_type> &&
               std::is_nothrow_swappable_v<storage_type> && std::is_nothrow_swappable_v<accessor_factory_type>)
    {
      Tensor replacement(exts, accessor_factory_);
      this->swap_state(replacement);
    }

    /// \brief Access the owned storage container.
    /// \return Mutable reference to the underlying storage.
    [[nodiscard]] storage_type& storage() noexcept { return data_; }

    /// \brief Access the owned storage container.
    /// \return Constant reference to the underlying storage.
    [[nodiscard]] storage_type const& storage() const noexcept { return data_; }

    /// \brief Transfer the underlying storage container out of this tensor.
    /// \details The mapping and accessor-factory state remain in the moved-from
    ///          tensor. Existing views and mdspans into this tensor must not be
    ///          used after the transfer.
    /// \return The concrete storage container selected by `storage_policy`.
    [[nodiscard]] storage_type release_storage() && noexcept(std::is_nothrow_move_constructible_v<storage_type>)
    {
      return std::move(data_);
    }

    /// \brief Transfer the accessor-factory state out of this tensor.
    /// \details This accompanies storage transfer when an owning operation
    ///          rebinds the allocation to a different tensor descriptor.
    /// \return The factory used to resolve accessors for the transferred storage.
    [[nodiscard]] accessor_factory_type
    release_accessor_factory() && noexcept(std::is_nothrow_move_constructible_v<accessor_factory_type>)
    {
      return std::move(accessor_factory_);
    }

    /// \brief Adopt an existing storage container without reallocating it.
    /// \details The storage may contain padding or an unused tail, but its
    ///          logical size must cover the mapping's required span. The
    ///          caller is responsible for ensuring that the mapping and
    ///          accessor factory are valid for the transferred allocation.
    /// \param mapping Mapping to expose through the adopted tensor.
    /// \param storage Storage container whose ownership is transferred.
    /// \param accessor_factory Factory used to resolve mdspan accessors.
    /// \return An owning tensor over the transferred storage.
    [[nodiscard]] static Tensor adopt_storage(mapping_type mapping, storage_type storage,
                                              accessor_factory_type accessor_factory = accessor_factory_type{})
      requires requires(storage_type const& value) { value.size(); }
    {
      ERROR_IF(std::cmp_less(storage.size(), mapping.required_span_size()),
               "adopted tensor storage is smaller than the mapping's required span");
      return Tensor(internal_tag{}, ctor_payload{std::move(mapping), std::move(storage), std::move(accessor_factory)});
    }

    /// \brief Replace the tensor mapping without moving or reallocating storage.
    /// \details Existing values are reinterpreted through the replacement
    ///          mapping. The mapping's required span must fit the current
    ///          allocation; higher-level operations establish any stronger
    ///          ordering or contiguity requirements.
    /// \param mapping Replacement mapping over the existing allocation.
    void replace_mapping(mapping_type mapping)
      requires requires(storage_type const& value) { value.size(); }
    {
      ERROR_IF(std::cmp_less(data_.size(), mapping.required_span_size()),
               "replacement tensor mapping exceeds the owned storage");
      mapping_ = std::move(mapping);
    }

    /// \brief Return the default backend selector associated with this tensor's storage.
    /// \return Ordered backend selector value for tensor-level dispatch.
    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return storage_policy::backend_selector();
    }

    /// \brief Resolve a writable mdspan over the owned storage.
    /// \return Mutable mdspan preserving the tensor mapping and accessor semantics.
    [[nodiscard]] auto mdspan() noexcept -> mdspan_type
    {
      return mdspan_type(this->mutable_handle(), mapping_, this->accessor());
    }

    /// \brief Resolve a read-only mdspan over the owned storage.
    /// \return Const mdspan preserving the tensor mapping and accessor semantics.
    [[nodiscard]] auto mdspan() const noexcept -> const_mdspan_type
    {
      return const_mdspan_type(this->handle(), mapping_, this->accessor());
    }

    /// \brief Return the tensor's const storage handle.
    /// \return Handle addressing the beginning of the owned storage.
    [[nodiscard]] auto handle() const noexcept -> handle_type { return storage_policy::make_handle(data_); }

    /// \brief Return the tensor's mutable storage handle.
    /// \return Mutable handle addressing the beginning of the owned storage.
    [[nodiscard]] auto mutable_handle() noexcept -> mutable_handle_type { return storage_policy::make_handle(data_); }

    /// \brief Return the tensor mapping.
    /// \return Mapping containing extents and strides.
    [[nodiscard]] auto mapping() const noexcept -> mapping_type const& { return mapping_; }

    /// \brief Return the tensor extents.
    /// \return Extents contained in the mapping.
    [[nodiscard]] auto extents() const noexcept -> extents_type const& { return mapping_.extents(); }

    /// \brief Return one tensor extent.
    /// \param axis Axis whose extent is requested.
    /// \return Extent of the requested axis.
    [[nodiscard]] auto extent(size_type axis) const noexcept { return this->extents().extent(axis); }

    /// \brief Construct the tensor's mutable accessor.
    /// \return Accessor used to construct writable resolved mdspans.
    [[nodiscard]] auto accessor() noexcept -> accessor_type
    {
      return accessor_factory_.template make_accessor<element_type>(data_);
    }

    /// \brief Construct the tensor's read-only accessor.
    /// \return Accessor used to construct read-only resolved mdspans.
    [[nodiscard]] auto accessor() const noexcept -> const_accessor_type
    {
      return accessor_factory_.template make_accessor<element_type const>(data_);
    }

    /// \brief Return the number of elements in the mapped storage span.
    /// \return Required span size of the tensor mapping.
    [[nodiscard]] auto size() const noexcept -> size_type { return mapping_.required_span_size(); }

    /// \brief Return the tensor rank.
    /// \return Static rank of the extents type.
    static constexpr size_type rank() noexcept { return extents_type::rank(); }

    /// \brief Return the number of dynamic extents.
    /// \return Static dynamic-rank count of the extents type.
    static constexpr size_type rank_dynamic() noexcept { return extents_type::rank_dynamic(); }

    /// \brief Return the matrix row count for rank-two tensors.
    /// \return First tensor extent.
    [[nodiscard]] auto rows() const noexcept
      requires(extents_type::rank() == 2)
    {
      return this->extent(0);
    }

    /// \brief Return the matrix column count for rank-two tensors.
    /// \return Second tensor extent.
    [[nodiscard]] auto cols() const noexcept
      requires(extents_type::rank() == 2)
    {
      return this->extent(1);
    }

    /// \brief Access a mutable tensor element.
    template <class... Index>
      requires(sizeof...(Index) == extents_type::rank())
    decltype(auto) operator[](Index... indices) noexcept
    {
      return this->mdspan()[indices...];
    }

    /// \brief Access a const tensor element.
    template <class... Index>
      requires(sizeof...(Index) == extents_type::rank())
    decltype(auto) operator[](Index... indices) const noexcept
    {
      return this->mdspan()[indices...];
    }

    /// \brief Access a mutable tensor element through an index array.
    /// \param indices Coordinates for every tensor axis.
    /// \return Mutable element reference.
    decltype(auto) operator[](std::array<index_type, extents_type::rank()> const& indices) noexcept
    {
      auto span = this->mdspan();
      return [&]<std::size_t... I>(std::index_sequence<I...>) -> decltype(auto) {
        return span[indices[I]...];
      }(std::make_index_sequence<extents_type::rank()>{});
    }

    /// \brief Access a const tensor element through an index array.
    /// \param indices Coordinates for every tensor axis.
    /// \return Const element reference.
    decltype(auto) operator[](std::array<index_type, extents_type::rank()> const& indices) const noexcept
    {
      auto span = this->mdspan();
      return [&]<std::size_t... I>(std::index_sequence<I...>) -> decltype(auto) {
        return span[indices[I]...];
      }(std::make_index_sequence<extents_type::rank()>{});
    }

  private:
    struct internal_tag
    {};

    struct ctor_payload
    {
        mapping_type mapping;
        storage_type storage;
        accessor_factory_type accessor_factory;
    };

    Tensor(internal_tag, ctor_payload payload)
        : mapping_(std::move(payload.mapping)), data_(std::move(payload.storage)),
          accessor_factory_(std::move(payload.accessor_factory))
    {}

    static ctor_payload make_payload(mapping_type mapping, accessor_factory_type accessor_factory)
    {
      auto storage = make_storage(mapping);
      return ctor_payload{std::move(mapping), std::move(storage), std::move(accessor_factory)};
    }

    static storage_type make_storage(mapping_type const& mapping)
    {
      auto const span_size = static_cast<size_type>(mapping.required_span_size());
      return create_storage(span_size);
    }

    static storage_type create_storage(size_type span_size)
    {
      auto const count = static_cast<std::size_t>(span_size);
      if constexpr (requires(storage_type& s) { s.resize(std::size_t{}); })
      {
        storage_type storage{};
        storage.resize(count);
        return storage;
      }
      else if constexpr (std::is_constructible_v<storage_type, std::size_t>)
      {
        return storage_type{count};
      }
      else if constexpr (std::is_constructible_v<storage_type, size_type>)
      {
        return storage_type{span_size};
      }
      else
      {
        static_assert(
            requires(storage_type& s) { s.resize(std::size_t{}); } ||
                std::is_constructible_v<storage_type, std::size_t> || std::is_constructible_v<storage_type, size_type>,
            "StoragePolicy::storage_t must be constructible from a size or provide resize().");
        return storage_type{};
      }
    }

    static constexpr auto make_default_mapping(extents_type const& exts)
    {
      if constexpr (requires { typename storage_policy::default_mapping_builder; })
      {
        using builder_type = typename storage_policy::default_mapping_builder;
        if constexpr (layout::mapping_builder_for<builder_type, layout_policy, extents_type>)
        {
          return builder_type{}(exts);
        }
        else
        {
          return layout::make_mapping<layout_policy>(exts);
        }
      }
      else
      {
        return layout::make_mapping<layout_policy>(exts);
      }
    }

    void swap_state(Tensor& other) noexcept
      requires(std::is_nothrow_swappable_v<mapping_type> && std::is_nothrow_swappable_v<storage_type> &&
               std::is_nothrow_swappable_v<accessor_factory_type>)
    {
      using std::swap;
      swap(mapping_, other.mapping_);
      swap(data_, other.data_);
      swap(accessor_factory_, other.accessor_factory_);
    }

    [[no_unique_address]] mapping_type mapping_{};
    storage_type data_{};
    [[no_unique_address]] accessor_factory_type accessor_factory_{};
};

template <typename ElementType, std::size_t Rank, typename StoragePolicy, typename LayoutPolicy,
          typename AccessorFactory, typename Extents>
inline constexpr bool
    enable_owning_tensor<Tensor<ElementType, Rank, StoragePolicy, LayoutPolicy, AccessorFactory, Extents>> = true;

/// \brief Deduce a runtime-extents host tensor that materializes a tensor view.
/// \details Canonical source layout is preserved. Sources without canonical
///          physical layout deduce the default column-major layout.
template <TensorView InputTensor>
Tensor(InputTensor const&)
    -> Tensor<tensor_element_t<InputTensor>, tensor_mdspan_t<InputTensor>::rank(), VectorStorage,
              detail::materialized_layout_t<void, tensor_mdspan_t<InputTensor>>, DefaultAccessorFactory>;

/// \brief Configurable owning tensor with an explicit mdspan extents type.
/// \ingroup tensor
/// \tparam ElementType Value type stored by the tensor.
/// \tparam Extents Extents type describing the tensor shape.
/// \tparam StoragePolicy Policy controlling ownership and allocation of the buffer.
/// \tparam LayoutPolicy Layout policy that determines index ordering and stride computation.
/// \tparam AccessorFactory Factory that produces accessors for the storage handle.
template <typename ElementType, typename Extents, typename StoragePolicy = VectorStorage,
          typename LayoutPolicy = stdex::layout_left, typename AccessorFactory = DefaultAccessorFactory>
using BasicTensor = Tensor<ElementType, Extents::rank(), StoragePolicy, LayoutPolicy, AccessorFactory, Extents>;

} // namespace uni20
