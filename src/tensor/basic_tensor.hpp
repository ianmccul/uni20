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

template <typename ElementType, typename Extents, typename StoragePolicy = VectorStorage,
          typename LayoutPolicy = stdex::layout_stride,
          typename AccessorFactory = DefaultAccessorFactory>
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

    BasicTensor() = default;

    explicit BasicTensor(extents_type const& exts, accessor_factory_type accessor_factory = accessor_factory_type{})
        : BasicTensor(internal_tag{},
                      make_payload(make_default_mapping(exts), std::move(accessor_factory)))
    {}

    template <typename MappingBuilder>
      requires(layout::mapping_builder_for<MappingBuilder, layout_policy, extents_type> &&
               (!std::same_as<std::remove_cvref_t<MappingBuilder>, accessor_factory_type>))
    explicit BasicTensor(extents_type const& exts, MappingBuilder&& mapping_builder,
                         accessor_factory_type accessor_factory = accessor_factory_type{})
        : BasicTensor(internal_tag{},
                      make_payload(std::forward<MappingBuilder>(mapping_builder)(exts),
                                   std::move(accessor_factory)))
    {}

    explicit BasicTensor(extents_type const& exts, std::array<index_type, extents_type::rank()> const& strides,
                         accessor_factory_type accessor_factory = accessor_factory_type{})
        : BasicTensor(internal_tag{}, make_payload(mapping_type{exts, strides}, std::move(accessor_factory)))
    {}

    storage_type& storage() noexcept { return data_; }
    storage_type const& storage() const noexcept { return data_; }

    auto view() noexcept -> TensorView<element_type, traits_type>
    {
      return TensorView<element_type, traits_type>(storage_policy::make_handle(data_), this->mapping(),
                                                   this->accessor());
    }

    auto view() const noexcept -> TensorView<element_type const, const_traits>
    {
      return TensorView<element_type const, const_traits>(
          storage_policy::make_handle(const_cast<storage_type&>(data_)), this->mapping(), this->accessor());
    }

    auto const_view() const noexcept -> TensorView<element_type const, const_traits>
    {
      return view();
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
        : base_type(storage_policy::make_handle(payload.storage), payload.mapping,
                    payload.accessor_factory.template make_accessor<element_type>(payload.storage)),
          data_(std::move(payload.storage))
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
      if constexpr (requires(storage_type & s) { s.resize(std::size_t{}); })
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
            requires(storage_type & s) { s.resize(std::size_t{}); } ||
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

    storage_type data_{};
};

} // namespace uni20
