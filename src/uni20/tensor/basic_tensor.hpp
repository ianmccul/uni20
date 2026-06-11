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
class BasicTensor
    : public TensorView<ElementType, mutable_tensor_traits<Extents, StoragePolicy, LayoutPolicy, AccessorFactory>> {
  private:
    using traits_type = mutable_tensor_traits<Extents, StoragePolicy, LayoutPolicy, AccessorFactory>;
    using const_traits = tensor_traits<Extents, StoragePolicy, LayoutPolicy, AccessorFactory>;
    using base_type = TensorView<ElementType, traits_type>;

  public:
    using element_type = ElementType;
    using storage_policy = StoragePolicy;
    using layout_policy = LayoutPolicy;
    using accessor_factory_type = AccessorFactory;

    using typename base_type::accessor_policy;
    using typename base_type::accessor_type;
    using typename base_type::extents_type;
    using typename base_type::handle_type;
    using typename base_type::index_type;
    using typename base_type::mapping_type;
    using typename base_type::size_type;

    using storage_type = typename storage_policy::template storage_t<element_type>;

    using base_type::mdspan;
    using base_type::mutable_mdspan;

    /// \brief Default-construct an empty tensor without allocated storage.
    BasicTensor() = default;

    /// \brief Copy-construct a tensor with independent owned storage.
    /// \details The inherited view is rebound to this tensor's copied storage;
    ///          tensor elements are copied through `storage_type`.
    /// \param other Source tensor to copy.
    BasicTensor(BasicTensor const& other) : base_type(), data_(other.data_)
    {
      this->reset_view(other.mapping(), other.accessor());
    }

    /// \brief Move-construct a tensor and rebind its inherited view.
    /// \details The inherited view is rebound to this tensor's moved storage so
    ///          it never aliases the moved-from object's storage descriptor.
    /// \param other Source tensor to move from.
    BasicTensor(BasicTensor&& other) : base_type(), data_(std::move(other.data_))
    {
      this->reset_view(other.mapping(), other.accessor());
      other.reset_view_to_owned_storage();
    }

    /// \brief Copy-assign tensor storage and rebind the inherited view.
    /// \param other Source tensor to copy.
    /// \return Reference to `*this`.
    BasicTensor& operator=(BasicTensor const& other)
    {
      if (this != &other)
      {
        data_ = other.data_;
        this->reset_view(other.mapping(), other.accessor());
      }
      return *this;
    }

    /// \brief Move-assign tensor storage and rebind the inherited view.
    /// \param other Source tensor to move from.
    /// \return Reference to `*this`.
    BasicTensor& operator=(BasicTensor&& other)
    {
      if (this != &other)
      {
        data_ = std::move(other.data_);
        this->reset_view(other.mapping(), other.accessor());
        other.reset_view_to_owned_storage();
      }
      return *this;
    }

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

    /// \brief Create a mutable tensor view referencing the owned storage.
    /// \return TensorView exposing mutable element access with the current mapping and accessor.
    [[nodiscard]] auto view() noexcept -> TensorView<element_type, traits_type>
    {
      return TensorView<element_type, traits_type>(storage_policy::make_handle(data_), this->mapping(),
                                                   this->accessor());
    }

    /// \brief Create a const tensor view referencing the owned storage.
    /// \return TensorView exposing read-only access with the current mapping and accessor.
    [[nodiscard]] auto view() const noexcept -> TensorView<element_type const, const_traits>
    {
      return TensorView<element_type const, const_traits>(storage_policy::make_handle(const_cast<storage_type&>(data_)),
                                                          this->mapping(), this->accessor());
    }

    /// \brief Create a const tensor view alias for readability.
    /// \return TensorView exposing read-only access with the current mapping and accessor.
    [[nodiscard]] auto const_view() const noexcept -> TensorView<element_type const, const_traits> { return view(); }

  private:
    struct internal_tag
    {};

    struct ctor_payload
    {
        mapping_type mapping;
        storage_type storage;
        accessor_factory_type accessor_factory;
    };

    BasicTensor(internal_tag, ctor_payload payload) : base_type(), data_(std::move(payload.storage))
    {
      this->reset_view(std::move(payload.mapping),
                       payload.accessor_factory.template make_accessor<element_type>(data_));
    }

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

    void reset_view(mapping_type mapping, accessor_type accessor)
    {
      static_cast<base_type&>(*this) =
          base_type(storage_policy::make_handle(data_), std::move(mapping), std::move(accessor));
    }

    void reset_view_to_owned_storage() { this->reset_view(this->mapping(), this->accessor()); }

    storage_type data_{};
};

} // namespace uni20
