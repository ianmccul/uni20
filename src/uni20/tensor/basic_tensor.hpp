#pragma once

#include "layout.hpp"
#include "tensor_view.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Owning tensor that allocates storage and exposes mdspan-based access.
/// \ingroup tensor
/// \tparam ElementType Value type stored by the tensor.
/// \tparam Extents Extents type describing the tensor shape.
/// \tparam StoragePolicy Policy controlling ownership and allocation of the buffer.
/// \tparam LayoutPolicy Layout policy that determines index ordering and stride computation.
/// \tparam AccessorFactory Factory that produces accessors for the storage handle.
template <typename ElementType, typename Extents, typename StoragePolicy = VectorStorage,
          typename LayoutPolicy = stdex::layout_stride, typename AccessorFactory = DefaultAccessorFactory>
class BasicTensor {
  private:
    using traits_type = mutable_tensor_traits<Extents, StoragePolicy, LayoutPolicy, AccessorFactory>;
    using const_traits = tensor_traits<Extents, StoragePolicy, LayoutPolicy, AccessorFactory>;
    using mutable_view_type = BasicTensorView<ElementType, traits_type>;
    using const_view_type = BasicTensorView<ElementType const, const_traits>;

  public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using storage_policy = StoragePolicy;
    using layout_policy = LayoutPolicy;
    using layout_type = layout_policy;
    using accessor_factory_type = AccessorFactory;
    using accessor_policy = typename traits_type::accessor_policy;
    using accessor_type = typename mutable_view_type::accessor_type;
    using const_accessor_type = typename const_view_type::accessor_type;
    using extents_type = Extents;
    using handle_type = typename const_view_type::handle_type;
    using mutable_handle_type = typename mutable_view_type::handle_type;
    using index_type = typename extents_type::index_type;
    using mapping_type = typename layout_policy::template mapping<extents_type>;
    using size_type = uni20::size_type;
    using backend_selector_type = typename storage_policy::backend_selector_type;
    using default_tag = typename storage_policy::default_tag;

    using storage_type = typename storage_policy::template storage_t<element_type>;

    /// \brief Default-construct an empty tensor without allocated storage.
    BasicTensor() = default;

    /// \brief Copy-construct a tensor with independent owned storage.
    /// \details Tensor elements, mapping, and accessor state are copied. Resolved
    ///          views are constructed on demand from this tensor's own storage.
    /// \param other Source tensor to copy.
    BasicTensor(BasicTensor const& other) = default;

    /// \brief Move-construct an owning tensor.
    /// \param other Source tensor to move from.
    BasicTensor(BasicTensor&& other) = default;

    /// \brief Copy-assign tensor storage and descriptor state.
    /// \param other Source tensor to copy.
    /// \return Reference to `*this`.
    BasicTensor& operator=(BasicTensor const& other) = default;

    /// \brief Move-assign tensor storage and descriptor state.
    /// \param other Source tensor to move from.
    /// \return Reference to `*this`.
    BasicTensor& operator=(BasicTensor&& other) = default;

    /// \brief Construct a tensor with default layout and accessor factory.
    /// \param exts Extents that describe the tensor shape.
    /// \param accessor_factory Factory used to create the accessor for the storage handle.
    explicit BasicTensor(extents_type const& exts, accessor_factory_type accessor_factory = accessor_factory_type{})
        : BasicTensor(internal_tag{}, make_payload(make_default_mapping(exts), std::move(accessor_factory)))
    {}

    /// \brief Construct a tensor using a custom mapping builder.
    /// \tparam MappingBuilder Callable that returns a mapping compatible with the layout policy.
    /// \param exts Extents that describe the tensor shape.
    /// \param mapping_builder Builder used to derive the mapping from the extents.
    /// \param accessor_factory Factory used to create the accessor for the storage handle.
    template <typename MappingBuilder>
      requires(layout::mapping_builder_for<MappingBuilder, layout_policy, extents_type> &&
               (!std::same_as<std::remove_cvref_t<MappingBuilder>, accessor_factory_type>))
    explicit BasicTensor(extents_type const& exts, MappingBuilder&& mapping_builder,
                         accessor_factory_type accessor_factory = accessor_factory_type{})
        : BasicTensor(internal_tag{},
                      make_payload(std::forward<MappingBuilder>(mapping_builder)(exts), std::move(accessor_factory)))
    {}

    /// \brief Construct a tensor from explicit extents and strides.
    /// \param exts Extents that describe the tensor shape.
    /// \param strides Stride specification per dimension for the layout mapping.
    /// \param accessor_factory Factory used to create the accessor for the storage handle.
    explicit BasicTensor(extents_type const& exts, std::array<index_type, extents_type::rank()> const& strides,
                         accessor_factory_type accessor_factory = accessor_factory_type{})
        : BasicTensor(internal_tag{}, make_payload(mapping_type{exts, strides}, std::move(accessor_factory)))
    {}

    /// \brief Access the owned storage container.
    /// \return Mutable reference to the underlying storage.
    [[nodiscard]] storage_type& storage() noexcept { return data_; }

    /// \brief Access the owned storage container.
    /// \return Constant reference to the underlying storage.
    [[nodiscard]] storage_type const& storage() const noexcept { return data_; }

    /// \brief Return the default backend selector associated with this tensor's storage.
    /// \return Ordered backend selector value for tensor-level dispatch.
    [[nodiscard]] constexpr auto backend_selector() const noexcept -> backend_selector_type
    {
      return storage_policy::backend_selector();
    }

    /// \brief Create a mutable tensor view referencing the owned storage.
    /// \return BasicTensorView exposing mutable element access with the current mapping and accessor.
    [[nodiscard]] auto view() noexcept -> mutable_view_type
    {
      return mutable_view_type(mutable_handle(), mapping_, accessor_);
    }

    /// \brief Create a const tensor view referencing the owned storage.
    /// \return BasicTensorView exposing read-only access with the current mapping and accessor.
    [[nodiscard]] auto view() const noexcept -> const_view_type
    {
      return const_view_type(handle(), mapping_, accessor_);
    }

    /// \brief Create a const tensor view alias for readability.
    /// \return BasicTensorView exposing read-only access with the current mapping and accessor.
    [[nodiscard]] auto const_view() const noexcept -> const_view_type { return view(); }

    /// \brief Resolve a writable mdspan over the owned storage.
    /// \return Mutable mdspan preserving the tensor mapping and accessor semantics.
    [[nodiscard]] auto mdspan() noexcept { return this->view().mdspan(); }

    /// \brief Resolve a read-only mdspan over the owned storage.
    /// \return Const mdspan preserving the tensor mapping and accessor semantics.
    [[nodiscard]] auto mdspan() const noexcept { return this->const_view().mdspan(); }

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

    /// \brief Return the tensor's mutable accessor.
    /// \return Accessor used to construct writable resolved views.
    [[nodiscard]] auto accessor() noexcept -> accessor_type& { return accessor_; }

    /// \brief Return the tensor's accessor state.
    /// \return Accessor used to construct resolved views.
    [[nodiscard]] auto accessor() const noexcept -> accessor_type const& { return accessor_; }

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
      return this->view()[indices];
    }

    /// \brief Access a const tensor element through an index array.
    /// \param indices Coordinates for every tensor axis.
    /// \return Const element reference.
    decltype(auto) operator[](std::array<index_type, extents_type::rank()> const& indices) const noexcept
    {
      return this->const_view()[indices];
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

    BasicTensor(internal_tag, ctor_payload payload)
        : mapping_(std::move(payload.mapping)), data_(std::move(payload.storage)),
          accessor_(payload.accessor_factory.template make_accessor<element_type>(data_))
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

    [[no_unique_address]] mapping_type mapping_{};
    storage_type data_{};
    [[no_unique_address]] accessor_type accessor_{};
};

} // namespace uni20
