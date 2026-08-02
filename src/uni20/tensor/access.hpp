/**
 * \file access.hpp
 * \ingroup tensor
 * \brief RAII leases that resolve device descriptors and tensor views to mdspans.
 */

#pragma once

#include <uni20/common/trace.hpp>
#include <uni20/tensor/concepts.hpp>

#include <concepts>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

namespace uni20
{
/// \brief Movable state retaining one acquired tensor data-handle lifetime.
/// \details `release()` must be idempotent: its first call ends active access,
///          and later calls have no effect. Moving an active state transfers
///          that responsibility and leaves the source inactive, so releasing
///          or destroying the moved-from state has no effect. Move construction
///          must not throw because leases transfer this state together with the
///          resolved mdspan that depends on it.
/// \tparam State Access-state type under test.
template <class State>
concept TensorAccessState = std::move_constructible<State> && std::is_nothrow_move_constructible_v<State> &&
                            requires(State& state) {
  { state.release() } noexcept;
};

namespace detail
{

struct immediate_mdspan_access_state
{
    void release() noexcept {}
};

/// \brief Pointer-only read lease for an immediately accessible tensor view.
template <ImmediateTensorView Tensor> class borrowed_read_tensor_lease {
  public:
    using tensor_type = std::remove_cvref_t<Tensor>;
    using mdspan_type = std::remove_cvref_t<decltype(std::declval<tensor_type const&>().mdspan())>;
    using storage_policy = tensor_storage_policy_t<tensor_type>;
    using extents_type = typename mdspan_type::extents_type;
    using index_type = typename mdspan_type::index_type;

    explicit borrowed_read_tensor_lease(tensor_type const& tensor) noexcept : tensor_(&tensor) {}

    borrowed_read_tensor_lease(borrowed_read_tensor_lease const&) = delete;
    borrowed_read_tensor_lease& operator=(borrowed_read_tensor_lease const&) = delete;

    borrowed_read_tensor_lease(borrowed_read_tensor_lease&& other) noexcept
        : tensor_(std::exchange(other.tensor_, nullptr))
    {}

    borrowed_read_tensor_lease& operator=(borrowed_read_tensor_lease&& other) noexcept
    {
      if (this != &other) tensor_ = std::exchange(other.tensor_, nullptr);
      return *this;
    }

    /// \brief Return the source tensor's read-only mdspan.
    [[nodiscard]] decltype(auto) mdspan() const& { return this->source().mdspan(); }

    void mdspan() && = delete;

    /// \brief Return the source tensor's backend selector.
    [[nodiscard]] decltype(auto) backend_selector() const { return this->source().backend_selector(); }

    /// \brief Return storage when the source tensor exposes it.
    [[nodiscard]] decltype(auto) storage() const
      requires requires(tensor_type const& tensor) { tensor.storage(); }
    {
      return this->source().storage();
    }

    /// \brief Return the source tensor extents.
    [[nodiscard]] decltype(auto) extents() const { return this->source().extents(); }

    /// \brief Return one source tensor extent.
    [[nodiscard]] index_type extent(std::size_t axis) const { return this->source().extent(axis); }

    /// \brief Return the static tensor rank.
    [[nodiscard]] static constexpr std::size_t rank() noexcept { return mdspan_type::rank(); }

    /// \brief End this no-op borrowed access scope.
    /// \details Repeated calls have no effect.
    void release() noexcept { tensor_ = nullptr; }

  private:
    [[nodiscard]] tensor_type const& source() const
    {
      CHECK(tensor_ != nullptr, "cannot use a released borrowed tensor lease");
      return *tensor_;
    }

    tensor_type const* tensor_ = nullptr;
};

/// \brief Pointer-only write lease for an immediately accessible tensor view.
template <MutableImmediateTensorView Tensor> class borrowed_write_tensor_lease {
  public:
    using tensor_type = std::remove_cvref_t<Tensor>;
    using mdspan_type = std::remove_cvref_t<decltype(std::declval<tensor_type&>().mdspan())>;
    using const_mdspan_type = std::remove_cvref_t<decltype(std::declval<tensor_type const&>().mdspan())>;
    using storage_policy = tensor_storage_policy_t<tensor_type>;
    using extents_type = typename mdspan_type::extents_type;
    using index_type = typename mdspan_type::index_type;

    explicit borrowed_write_tensor_lease(tensor_type& tensor) noexcept : tensor_(&tensor) {}

    borrowed_write_tensor_lease(borrowed_write_tensor_lease const&) = delete;
    borrowed_write_tensor_lease& operator=(borrowed_write_tensor_lease const&) = delete;

    borrowed_write_tensor_lease(borrowed_write_tensor_lease&& other) noexcept
        : tensor_(std::exchange(other.tensor_, nullptr))
    {}

    borrowed_write_tensor_lease& operator=(borrowed_write_tensor_lease&& other) noexcept
    {
      if (this != &other) tensor_ = std::exchange(other.tensor_, nullptr);
      return *this;
    }

    /// \brief Return the source tensor's writable mdspan.
    [[nodiscard]] decltype(auto) mdspan() & { return this->source().mdspan(); }

    /// \brief Return the source tensor's read-only mdspan.
    [[nodiscard]] decltype(auto) mdspan() const& { return std::as_const(this->source()).mdspan(); }

    void mdspan() && = delete;

    /// \brief Return the source tensor's backend selector.
    [[nodiscard]] decltype(auto) backend_selector() const { return this->source().backend_selector(); }

    /// \brief Return the source tensor extents.
    [[nodiscard]] decltype(auto) extents() const { return this->source().extents(); }

    /// \brief Return one source tensor extent.
    [[nodiscard]] index_type extent(std::size_t axis) const { return this->source().extent(axis); }

    /// \brief Return the static tensor rank.
    [[nodiscard]] static constexpr std::size_t rank() noexcept { return mdspan_type::rank(); }

    /// \brief End this no-op borrowed access scope.
    /// \details Repeated calls have no effect.
    void release() noexcept { tensor_ = nullptr; }

  private:
    [[nodiscard]] tensor_type& source() const
    {
      CHECK(tensor_ != nullptr, "cannot use a released borrowed tensor lease");
      return *tensor_;
    }

    tensor_type* tensor_ = nullptr;
};

} // namespace detail

/// \brief Move-only RAII lease exposing one resolved read-only mdspan.
/// \details This lower-level lease intentionally carries no tensor storage
///          policy or backend selector. The access state is stored before the
///          mdspan so the mdspan is destroyed before access ends.
/// \tparam Mdspan Read-only resolved mdspan type.
/// \tparam AccessState RAII state retaining the acquired data handle.
template <MdspanLike Mdspan, TensorAccessState AccessState>
  requires std::is_const_v<typename Mdspan::element_type> && std::is_nothrow_move_constructible_v<Mdspan>
class read_mdspan_lease {
  public:
    using mdspan_type = Mdspan;
    using access_state_type = AccessState;

    /// \brief Construct an active read lease.
    read_mdspan_lease(access_state_type state, mdspan_type mdspan) noexcept
        : state_(std::move(state)), mdspan_(std::move(mdspan))
    {}

    read_mdspan_lease(read_mdspan_lease const&) = delete;
    read_mdspan_lease& operator=(read_mdspan_lease const&) = delete;

    read_mdspan_lease(read_mdspan_lease&& other) noexcept
        : state_(std::move(other.state_)), mdspan_(std::move(other.mdspan_))
    {
      other.mdspan_.reset();
    }

    read_mdspan_lease&
    operator=(read_mdspan_lease&& other) noexcept
      requires(std::is_nothrow_move_assignable_v<access_state_type> && std::is_nothrow_move_assignable_v<mdspan_type>)
    {
      if (this != &other)
      {
        this->release();
        state_ = std::move(other.state_);
        mdspan_ = std::move(other.mdspan_);
        other.mdspan_.reset();
      }
      return *this;
    }

    ~read_mdspan_lease() { this->release(); }

    /// \brief Return the resolved read-only mdspan.
    [[nodiscard]] mdspan_type& mdspan() &
    {
      CHECK(mdspan_.has_value(), "cannot use an inactive mdspan lease");
      return *mdspan_;
    }

    /// \brief Return the resolved read-only mdspan.
    [[nodiscard]] mdspan_type const& mdspan() const&
    {
      CHECK(mdspan_.has_value(), "cannot use an inactive mdspan lease");
      return *mdspan_;
    }

    mdspan_type mdspan() && = delete;

    /// \brief End the access scope.
    /// \details Repeated calls have no effect.
    void release() noexcept
    {
      if (!mdspan_) return;
      mdspan_.reset();
      state_.release();
    }

  private:
    [[no_unique_address]] access_state_type state_;
    std::optional<mdspan_type> mdspan_;
};

/// \brief Move-only RAII lease exposing one resolved writable mdspan.
/// \details This lower-level lease intentionally carries no tensor storage
///          policy or backend selector. The access state is stored before the
///          mdspan so the mdspan is destroyed before access ends.
/// \tparam Mdspan Writable resolved mdspan type.
/// \tparam AccessState RAII state retaining the acquired data handle.
template <MutableMdspanLike Mdspan, TensorAccessState AccessState>
  requires std::is_nothrow_move_constructible_v<Mdspan>
class write_mdspan_lease {
  public:
    using mdspan_type = Mdspan;
    using access_state_type = AccessState;

    /// \brief Construct an active write lease.
    write_mdspan_lease(access_state_type state, mdspan_type mdspan) noexcept
        : state_(std::move(state)), mdspan_(std::move(mdspan))
    {}

    write_mdspan_lease(write_mdspan_lease const&) = delete;
    write_mdspan_lease& operator=(write_mdspan_lease const&) = delete;

    write_mdspan_lease(write_mdspan_lease&& other) noexcept
        : state_(std::move(other.state_)), mdspan_(std::move(other.mdspan_))
    {
      other.mdspan_.reset();
    }

    write_mdspan_lease&
    operator=(write_mdspan_lease&& other) noexcept
      requires(std::is_nothrow_move_assignable_v<access_state_type> && std::is_nothrow_move_assignable_v<mdspan_type>)
    {
      if (this != &other)
      {
        this->release();
        state_ = std::move(other.state_);
        mdspan_ = std::move(other.mdspan_);
        other.mdspan_.reset();
      }
      return *this;
    }

    ~write_mdspan_lease() { this->release(); }

    /// \brief Return the resolved writable mdspan.
    [[nodiscard]] mdspan_type& mdspan() &
    {
      CHECK(mdspan_.has_value(), "cannot use an inactive mdspan lease");
      return *mdspan_;
    }

    /// \brief Return the resolved writable mdspan without changing lease ownership.
    [[nodiscard]] mdspan_type const& mdspan() const&
    {
      CHECK(mdspan_.has_value(), "cannot use an inactive mdspan lease");
      return *mdspan_;
    }

    mdspan_type mdspan() && = delete;

    /// \brief End the access scope.
    /// \details Repeated calls have no effect.
    void release() noexcept
    {
      if (!mdspan_) return;
      mdspan_.reset();
      state_.release();
    }

  private:
    [[no_unique_address]] access_state_type state_;
    std::optional<mdspan_type> mdspan_;
};

/// \brief Move-only RAII lease exposing a read-only ImmediateTensorView.
/// \details The access state is stored before the mdspan so the mdspan is
///          destroyed before the state that makes its data handle usable.
///          Independent lease types may model `ReadTensorLease` without using
///          this class template.
/// \tparam Mdspan Read-only resolved mdspan type.
/// \tparam AccessState RAII state retaining the acquired data handle.
/// \tparam BackendSelector Backend list copied from the source tensor view.
/// \tparam StoragePolicy Storage policy associated with the source tensor view.
template <MdspanLike Mdspan, TensorAccessState AccessState, class BackendSelector, class StoragePolicy>
  requires std::is_const_v<typename Mdspan::element_type> && std::is_nothrow_move_constructible_v<Mdspan> &&
           std::is_nothrow_move_constructible_v<BackendSelector>
class read_tensor_lease {
  public:
    using mdspan_type = Mdspan;
    using access_state_type = AccessState;
    using backend_selector_type = BackendSelector;
    using storage_policy = StoragePolicy;
    using extents_type = typename mdspan_type::extents_type;
    using index_type = typename mdspan_type::index_type;

    /// \brief Construct an active read lease.
    read_tensor_lease(access_state_type state, mdspan_type mdspan, backend_selector_type selector) noexcept
        : state_(std::move(state)), mdspan_(std::move(mdspan)), selector_(std::move(selector))
    {}

    read_tensor_lease(read_tensor_lease const&) = delete;
    read_tensor_lease& operator=(read_tensor_lease const&) = delete;

    read_tensor_lease(read_tensor_lease&& other) noexcept
        : state_(std::move(other.state_)), mdspan_(std::move(other.mdspan_)), selector_(std::move(other.selector_))
    {
      other.mdspan_.reset();
    }

    read_tensor_lease&
    operator=(read_tensor_lease&& other) noexcept
      requires(std::is_nothrow_move_assignable_v<access_state_type> && std::is_nothrow_move_assignable_v<mdspan_type> &&
               std::is_nothrow_move_assignable_v<backend_selector_type>)
    {
      if (this != &other)
      {
        this->release();
        state_ = std::move(other.state_);
        mdspan_ = std::move(other.mdspan_);
        selector_ = std::move(other.selector_);
        other.mdspan_.reset();
      }
      return *this;
    }

    ~read_tensor_lease() { this->release(); }

    /// \brief Return the resolved read-only mdspan.
    [[nodiscard]] mdspan_type& mdspan() &
    {
      CHECK(mdspan_.has_value(), "cannot resolve mdspan through an inactive tensor lease");
      return *mdspan_;
    }

    /// \brief Return the resolved read-only mdspan.
    [[nodiscard]] mdspan_type const& mdspan() const&
    {
      CHECK(mdspan_.has_value(), "cannot resolve mdspan through an inactive tensor lease");
      return *mdspan_;
    }

    mdspan_type mdspan() && = delete;

    /// \brief Return the backend selector retained from the source tensor view.
    [[nodiscard]] backend_selector_type const& backend_selector() const noexcept { return selector_; }

    /// \brief Return the storage retained by access state when it exposes one.
    [[nodiscard]] decltype(auto) storage() const
      requires requires(access_state_type const& state) { state.storage(); }
    {
      return state_.storage();
    }

    /// \brief Return the resolved tensor extents.
    [[nodiscard]] extents_type const& extents() const { return this->mdspan().extents(); }

    /// \brief Return one resolved tensor extent.
    [[nodiscard]] index_type extent(std::size_t axis) const { return this->mdspan().extent(axis); }

    /// \brief Return the static tensor rank.
    [[nodiscard]] static constexpr std::size_t rank() noexcept { return mdspan_type::rank(); }

    /// \brief End the access scope and make this lease inactive.
    /// \details Invalidates the resolved mdspan before releasing the access
    ///          state. Repeated calls have no effect.
    void release() noexcept
    {
      if (!mdspan_) return;
      mdspan_.reset();
      state_.release();
    }

  private:
    [[no_unique_address]] access_state_type state_;
    std::optional<mdspan_type> mdspan_;
    [[no_unique_address]] backend_selector_type selector_;
};

/// \brief Move-only RAII lease exposing mutable and const immediate TensorViews.
/// \details A non-const lease resolves a writable mdspan. Its const interface
///          resolves a distinct const-element mdspan, preserving ImmediateTensorView
///          shallow-const rules.
/// \tparam MutableMdspan Writable resolved mdspan type.
/// \tparam ConstMdspan Read-only resolved mdspan type.
/// \tparam AccessState RAII state retaining exclusive access.
/// \tparam BackendSelector Backend list copied from the source tensor view.
/// \tparam StoragePolicy Storage policy associated with the source tensor view.
template <MutableMdspanLike MutableMdspan, MdspanLike ConstMdspan, TensorAccessState AccessState, class BackendSelector,
          class StoragePolicy>
  requires std::is_const_v<typename ConstMdspan::element_type> &&
           std::is_nothrow_move_constructible_v<MutableMdspan> &&
           std::is_nothrow_move_constructible_v<ConstMdspan> &&
           std::is_nothrow_move_constructible_v<BackendSelector>
class write_tensor_lease {
  public:
    using mdspan_type = MutableMdspan;
    using const_mdspan_type = ConstMdspan;
    using access_state_type = AccessState;
    using backend_selector_type = BackendSelector;
    using storage_policy = StoragePolicy;
    using extents_type = typename mdspan_type::extents_type;
    using index_type = typename mdspan_type::index_type;

    /// \brief Construct an active write lease.
    write_tensor_lease(access_state_type state, mdspan_type mdspan, const_mdspan_type const_mdspan,
                       backend_selector_type selector) noexcept
        : state_(std::move(state)), mdspan_(std::move(mdspan)), const_mdspan_(std::move(const_mdspan)),
          selector_(std::move(selector))
    {}

    write_tensor_lease(write_tensor_lease const&) = delete;
    write_tensor_lease& operator=(write_tensor_lease const&) = delete;

    write_tensor_lease(write_tensor_lease&& other) noexcept
        : state_(std::move(other.state_)), mdspan_(std::move(other.mdspan_)),
          const_mdspan_(std::move(other.const_mdspan_)), selector_(std::move(other.selector_))
    {
      other.const_mdspan_.reset();
      other.mdspan_.reset();
    }

    write_tensor_lease&
    operator=(write_tensor_lease&& other) noexcept
      requires(std::is_nothrow_move_assignable_v<access_state_type> && std::is_nothrow_move_assignable_v<mdspan_type> &&
               std::is_nothrow_move_assignable_v<const_mdspan_type> &&
               std::is_nothrow_move_assignable_v<backend_selector_type>)
    {
      if (this != &other)
      {
        this->release();
        state_ = std::move(other.state_);
        mdspan_ = std::move(other.mdspan_);
        const_mdspan_ = std::move(other.const_mdspan_);
        selector_ = std::move(other.selector_);
        other.const_mdspan_.reset();
        other.mdspan_.reset();
      }
      return *this;
    }

    ~write_tensor_lease() { this->release(); }

    /// \brief Return the resolved writable mdspan.
    [[nodiscard]] mdspan_type& mdspan() &
    {
      CHECK(mdspan_.has_value(), "cannot resolve mdspan through an inactive tensor lease");
      return *mdspan_;
    }

    /// \brief Return the resolved read-only mdspan.
    [[nodiscard]] const_mdspan_type const& mdspan() const&
    {
      CHECK(const_mdspan_.has_value(), "cannot resolve mdspan through an inactive tensor lease");
      return *const_mdspan_;
    }

    mdspan_type mdspan() && = delete;

    /// \brief Return the backend selector retained from the source tensor view.
    [[nodiscard]] backend_selector_type const& backend_selector() const noexcept { return selector_; }

    /// \brief Return the resolved tensor extents.
    [[nodiscard]] extents_type const& extents() const { return this->mdspan().extents(); }

    /// \brief Return one resolved tensor extent.
    [[nodiscard]] index_type extent(std::size_t axis) const { return this->mdspan().extent(axis); }

    /// \brief Return the static tensor rank.
    [[nodiscard]] static constexpr std::size_t rank() noexcept { return mdspan_type::rank(); }

    /// \brief End the access scope and make this lease inactive.
    /// \details Invalidates both resolved mdspans before releasing the access
    ///          state. Repeated calls have no effect.
    void release() noexcept
    {
      if (!mdspan_) return;
      const_mdspan_.reset();
      mdspan_.reset();
      state_.release();
    }

  private:
    [[no_unique_address]] access_state_type state_;
    std::optional<mdspan_type> mdspan_;
    std::optional<const_mdspan_type> const_mdspan_;
    [[no_unique_address]] backend_selector_type selector_;
};

/// \brief Move-only lease exposing a resolved read-only mdspan.
template <class Lease>
concept ReadMdspanLease =
    std::move_constructible<Lease> && (!std::copy_constructible<Lease>) && requires(Lease& lease) {
      requires MdspanLike<decltype(lease.mdspan())>;
      requires std::is_const_v<typename std::remove_cvref_t<decltype(lease.mdspan())>::element_type>;
      { lease.release() } noexcept;
    };

/// \brief Move-only lease exposing a resolved writable mdspan.
template <class Lease>
concept WriteMdspanLease =
    std::move_constructible<Lease> && (!std::copy_constructible<Lease>) && requires(Lease& lease) {
      requires MutableMdspanLike<decltype(lease.mdspan())>;
      { lease.release() } noexcept;
    };

/// \brief Read mdspan lease whose resolved accessor is valid in host code.
template <class Lease>
concept HostReadMdspanLease = ReadMdspanLease<Lease> && HostAccessibleMdspan<decltype(std::declval<Lease&>().mdspan())>;

/// \brief Write mdspan lease whose resolved accessor is valid in host code.
template <class Lease>
concept HostWriteMdspanLease =
    WriteMdspanLease<Lease> && HostAccessibleMdspan<decltype(std::declval<Lease&>().mdspan())>;

/// \brief Read mdspan lease whose resolved accessor is valid in CUDA code.
template <class Lease>
concept CudaReadMdspanLease = ReadMdspanLease<Lease> && CudaAccessibleMdspan<decltype(std::declval<Lease&>().mdspan())>;

/// \brief Write mdspan lease whose resolved accessor is valid in CUDA code.
template <class Lease>
concept CudaWriteMdspanLease =
    WriteMdspanLease<Lease> && CudaAccessibleMdspan<decltype(std::declval<Lease&>().mdspan())>;

/// \brief ImmediateTensorView that owns a move-only read access lifetime.
/// \details `release()` must be `noexcept` and idempotent. Moving an active
///          lease transfers its access lifetime and leaves the source inactive.
template <class Lease>
concept ReadTensorLease = ImmediateTensorView<Lease> && std::move_constructible<Lease> &&
                          (!std::copy_constructible<Lease>) && requires(Lease& lease) {
                            { lease.release() } noexcept;
                          };

/// \brief MutableImmediateTensorView that owns a move-only write access lifetime.
template <class Lease>
concept WriteTensorLease = ReadTensorLease<Lease> && MutableImmediateTensorView<Lease>;

/// \brief Read lease whose resolved mdspan is directly host-accessible.
template <class Lease>
concept HostReadTensorLease = ReadTensorLease<Lease> && HostAccessibleMdspan<decltype(std::declval<Lease&>().mdspan())>;

/// \brief Write lease whose resolved mutable mdspan is directly host-accessible.
template <class Lease>
concept HostWriteTensorLease =
    WriteTensorLease<Lease> && HostAccessibleMdspan<decltype(std::declval<Lease&>().mdspan())>;

/// \brief Read lease whose resolved mdspan is directly CUDA-accessible.
template <class Lease>
concept CudaReadTensorLease = ReadTensorLease<Lease> && CudaAccessibleMdspan<decltype(std::declval<Lease&>().mdspan())>;

/// \brief Write lease whose resolved mutable mdspan is directly CUDA-accessible.
template <class Lease>
concept CudaWriteTensorLease =
    WriteTensorLease<Lease> && CudaAccessibleMdspan<decltype(std::declval<Lease&>().mdspan())>;

/// \brief Always-ready awaitable that transfers one access lease.
template <class Lease> class ready_access {
  public:
    using lease_type = Lease;

    explicit ready_access(lease_type lease) : lease_(std::move(lease)) {}

    ready_access(ready_access const&) = delete;
    ready_access& operator=(ready_access const&) = delete;
    ready_access(ready_access&&) = default;
    ready_access& operator=(ready_access&&) = default;

    [[nodiscard]] bool await_ready() const noexcept { return true; }

    template <class Task> void await_suspend(Task&&) const noexcept {}

    [[nodiscard]] lease_type await_resume() { return std::move(lease_); }

  private:
    lease_type lease_;
};

/// \brief Acquire a directly host-accessible read-only mdspan.
template <HostAccessibleMdspan Mdspan>
  requires std::is_const_v<typename std::remove_cvref_t<Mdspan>::element_type>
[[nodiscard]] auto acquire_host_read_access_sync(Mdspan const& mdspan)
{
  using mdspan_type = std::remove_cvref_t<Mdspan>;
  return read_mdspan_lease<mdspan_type, detail::immediate_mdspan_access_state>{detail::immediate_mdspan_access_state{},
                                                                               mdspan};
}

/// \brief Acquire a directly host-accessible writable mdspan.
template <MutableMdspanLike Mdspan>
  requires HostAccessibleMdspan<Mdspan>
[[nodiscard]] auto acquire_host_write_access_sync(Mdspan& mdspan)
{
  using mdspan_type = std::remove_cvref_t<Mdspan>;
  return write_mdspan_lease<mdspan_type, detail::immediate_mdspan_access_state>{detail::immediate_mdspan_access_state{},
                                                                                mdspan};
}

/// \brief Acquire an immediately host-accessible tensor view for read-only use.
template <ImmediateTensorView Tensor>
  requires HostAccessibleMdspan<immediate_tensor_mdspan_t<Tensor>>
[[nodiscard]] auto acquire_host_read_access_sync(Tensor& tensor)
{
  using tensor_type = std::remove_cvref_t<Tensor>;
  return detail::borrowed_read_tensor_lease<tensor_type>{tensor};
}

/// \brief Acquire an immediately host-accessible mutable tensor view.
template <MutableImmediateTensorView Tensor>
  requires HostAccessibleMdspan<mutable_immediate_tensor_mdspan_t<Tensor>>
[[nodiscard]] auto acquire_host_write_access_sync(Tensor& tensor)
{
  using tensor_type = std::remove_cvref_t<Tensor>;
  return detail::borrowed_write_tensor_lease<tensor_type>{tensor};
}

/// \brief Return an immediately-ready host read acquisition.
template <ImmediateTensorView Tensor>
  requires HostAccessibleMdspan<immediate_tensor_mdspan_t<Tensor>>
[[nodiscard]] auto acquire_host_read_access_async(Tensor& tensor)
{
  return ready_access{acquire_host_read_access_sync(tensor)};
}

/// \brief Return an immediately-ready host write acquisition.
template <MutableImmediateTensorView Tensor>
  requires HostAccessibleMdspan<mutable_immediate_tensor_mdspan_t<Tensor>>
[[nodiscard]] auto acquire_host_write_access_async(Tensor& tensor)
{
  return ready_access{acquire_host_write_access_sync(tensor)};
}

namespace detail
{
template <class Mdspec>
using host_read_mdspan_lease_t = decltype(acquire_host_read_access_sync(std::declval<Mdspec const&>()));

template <class Mdspec>
using host_write_mdspan_lease_t = decltype(acquire_host_write_access_sync(std::declval<Mdspec&>()));
} // namespace detail

/// \brief Read-only mdspan type resolved by host descriptor acquisition.
template <class Mdspec>
using host_read_mdspan_t =
    std::remove_cvref_t<decltype(std::declval<detail::host_read_mdspan_lease_t<Mdspec>&>().mdspan())>;

/// \brief Writable mdspan type resolved by host descriptor acquisition.
template <class Mdspec>
using host_write_mdspan_t =
    std::remove_cvref_t<decltype(std::declval<detail::host_write_mdspan_lease_t<Mdspec>&>().mdspan())>;

/// \brief Mdspec metadata supporting host read acquisition.
template <class Mdspec>
concept HostReadableMdspec = MdspecLike<Mdspec> && requires(Mdspec const& mdspec) {
  { acquire_host_read_access_sync(mdspec) } -> HostReadMdspanLease;
};

/// \brief Mutable mdspec metadata supporting host write acquisition.
template <class Mdspec>
concept HostWritableMdspec = MutableMdspecLike<Mdspec> && requires(Mdspec& mdspec) {
  { acquire_host_write_access_sync(mdspec) } -> HostWriteMdspanLease;
};

namespace detail
{
template <class Tensor>
using host_read_tensor_lease_t = decltype(acquire_host_read_access_sync(std::declval<Tensor const&>()));

template <class Tensor>
using host_write_tensor_lease_t = decltype(acquire_host_write_access_sync(std::declval<Tensor&>()));
} // namespace detail

/// \brief Read-only mdspan type resolved by host tensor acquisition.
template <class Tensor>
using host_read_tensor_mdspan_t =
    std::remove_cvref_t<decltype(std::declval<detail::host_read_tensor_lease_t<Tensor>&>().mdspan())>;

/// \brief Writable mdspan type resolved by host tensor acquisition.
template <class Tensor>
using host_write_tensor_mdspan_t =
    std::remove_cvref_t<decltype(std::declval<detail::host_write_tensor_lease_t<Tensor>&>().mdspan())>;

/// \brief Tensor view supporting host read acquisition.
template <class Tensor>
concept HostReadableTensor = TensorView<Tensor> && requires(Tensor const& tensor) {
  { acquire_host_read_access_sync(tensor) } -> HostReadTensorLease;
};

/// \brief Mutable tensor view supporting host write acquisition.
template <class Tensor>
concept HostWritableTensor = MutableTensorView<Tensor> && requires(Tensor& tensor) {
  { acquire_host_write_access_sync(tensor) } -> HostWriteTensorLease;
};

} // namespace uni20
