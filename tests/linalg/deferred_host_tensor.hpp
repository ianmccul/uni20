#pragma once

#include <uni20/tensor/tensor.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::test
{

template <class Storage> struct HostStorageDescriptor
{
    Storage* storage = nullptr;
};

class HostAccessState {
  public:
    HostAccessState() = default;
    HostAccessState(HostAccessState const&) = delete;
    HostAccessState& operator=(HostAccessState const&) = delete;

    HostAccessState(HostAccessState&& other) noexcept : active_(std::exchange(other.active_, false)) {}

    HostAccessState& operator=(HostAccessState&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        active_ = std::exchange(other.active_, false);
      }
      return *this;
    }

    void release() noexcept { active_ = false; }

  private:
    bool active_ = true;
};

template <class Element, class Extents, class Layout, class Accessor, class Storage>
  requires std::is_const_v<Element>
[[nodiscard]] auto
acquire_host_read_access_sync(mdspec<Element, Extents, Layout, Accessor, HostStorageDescriptor<Storage>> const& span)
{
  using mdspan_type = stdex::mdspan<Element, Extents, Layout, Accessor>;
  using lease_type = read_mdspan_lease<mdspan_type, HostAccessState>;
  return lease_type{
      HostAccessState{},
      mdspan_type{span.data_descriptor().storage->data(), span.mapping(), span.accessor()},
  };
}

template <class Element, class Extents, class Layout, class Accessor, class Storage>
  requires(!std::is_const_v<Element>)
[[nodiscard]] auto
acquire_host_write_access_sync(mdspec<Element, Extents, Layout, Accessor, HostStorageDescriptor<Storage>>& span)
{
  using mdspan_type = stdex::mdspan<Element, Extents, Layout, Accessor>;
  using lease_type = write_mdspan_lease<mdspan_type, HostAccessState>;
  return lease_type{
      HostAccessState{},
      mdspan_type{span.data_descriptor().storage->data(), span.mapping(), span.accessor()},
  };
}

template <class Element, std::size_t Rank, class Layout = stdex::layout_left> class DeferredHostTensor {
  public:
    using element_type = Element;
    using value_type = std::remove_cv_t<element_type>;
    using extents_type = stdex::dextents<index_type, Rank>;
    using layout_type = Layout;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    using storage_type = std::vector<element_type>;
    using storage_policy = VectorStorage;
    using backend_selector_type = linalg::backend_list<linalg::CpuReferenceBackend>;
    using mutable_accessor_type = stdex::default_accessor<element_type>;
    using const_accessor_type = stdex::default_accessor<element_type const>;
    using mutable_descriptor_type = HostStorageDescriptor<storage_type>;
    using const_descriptor_type = HostStorageDescriptor<storage_type const>;
    using mutable_mdspec_type =
        uni20::mdspec<element_type, extents_type, layout_type, mutable_accessor_type, mutable_descriptor_type>;
    using const_mdspec_type =
        uni20::mdspec<element_type const, extents_type, layout_type, const_accessor_type, const_descriptor_type>;
    using mutable_mdspan_type = stdex::mdspan<element_type, extents_type, layout_type, mutable_accessor_type>;
    using const_mdspan_type = stdex::mdspan<element_type const, extents_type, layout_type, const_accessor_type>;

    explicit DeferredHostTensor(extents_type extents)
        : mapping_(std::move(extents)), storage_(static_cast<std::size_t>(mapping_.required_span_size()))
    {}

    template <std::integral... Extent>
      requires(sizeof...(Extent) == Rank)
    explicit DeferredHostTensor(Extent... extent) : DeferredHostTensor(extents_type{static_cast<index_type>(extent)...})
    {}

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return backend_selector_type{linalg::CpuReferenceBackend{}};
    }

    [[nodiscard]] auto mdspec() noexcept -> mutable_mdspec_type
    {
      return mutable_mdspec_type{mutable_descriptor_type{&storage_}, mapping_, mutable_accessor_type{}};
    }

    [[nodiscard]] auto mdspec() const noexcept -> const_mdspec_type
    {
      return const_mdspec_type{const_descriptor_type{&storage_}, mapping_, const_accessor_type{}};
    }

    [[nodiscard]] auto extents() const noexcept -> extents_type const& { return mapping_.extents(); }

    [[nodiscard]] index_type extent(std::size_t axis) const noexcept { return this->extents().extent(axis); }

    [[nodiscard]] auto mapping() const noexcept -> mapping_type const& { return mapping_; }

    [[nodiscard]] auto storage() noexcept -> storage_type& { return storage_; }

    [[nodiscard]] auto storage() const noexcept -> storage_type const& { return storage_; }

  private:
    mapping_type mapping_;
    storage_type storage_;
};

template <class Element, std::size_t Rank, class Layout>
[[nodiscard]] auto acquire_host_read_access_sync(DeferredHostTensor<Element, Rank, Layout> const& tensor)
{
  using tensor_type = DeferredHostTensor<Element, Rank, Layout>;
  using lease_type =
      read_tensor_lease<typename tensor_type::const_mdspan_type, HostAccessState,
                        typename tensor_type::backend_selector_type, typename tensor_type::storage_policy>;
  return lease_type{HostAccessState{},
                    typename tensor_type::const_mdspan_type{tensor.storage().data(), tensor.mapping(),
                                                            typename tensor_type::const_accessor_type{}},
                    tensor.backend_selector()};
}

template <class Element, std::size_t Rank, class Layout>
[[nodiscard]] auto acquire_host_write_access_sync(DeferredHostTensor<Element, Rank, Layout>& tensor)
{
  using tensor_type = DeferredHostTensor<Element, Rank, Layout>;
  using lease_type =
      write_tensor_lease<typename tensor_type::mutable_mdspan_type, typename tensor_type::const_mdspan_type,
                         HostAccessState, typename tensor_type::backend_selector_type,
                         typename tensor_type::storage_policy>;
  return lease_type{HostAccessState{},
                    typename tensor_type::mutable_mdspan_type{tensor.storage().data(), tensor.mapping(),
                                                              typename tensor_type::mutable_accessor_type{}},
                    typename tensor_type::const_mdspan_type{tensor.storage().data(), tensor.mapping(),
                                                            typename tensor_type::const_accessor_type{}},
                    tensor.backend_selector()};
}

static_assert(TensorView<DeferredHostTensor<double, 2>>);
static_assert(MutableTensorView<DeferredHostTensor<double, 2>>);
static_assert(!ImmediateTensorView<DeferredHostTensor<double, 2>>);
static_assert(HostReadableMdspec<typename DeferredHostTensor<double, 2>::const_mdspec_type>);
static_assert(HostWritableMdspec<typename DeferredHostTensor<double, 2>::mutable_mdspec_type>);
static_assert(HostReadableTensor<DeferredHostTensor<double, 2>>);
static_assert(HostWritableTensor<DeferredHostTensor<double, 2>>);

} // namespace uni20::test

namespace uni20
{

template <class Element, std::size_t Rank, class Layout>
inline constexpr bool enable_owning_tensor<test::DeferredHostTensor<Element, Rank, Layout>> = true;

static_assert(OwningTensor<test::DeferredHostTensor<double, 2>>);

} // namespace uni20
