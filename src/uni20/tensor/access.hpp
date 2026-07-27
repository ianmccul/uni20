/**
 * \file access.hpp
 * \ingroup tensor
 * \brief RAII leases that resolve device tensor views into immediate tensor views.
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
namespace detail
{

template <class State>
concept TensorAccessState = std::move_constructible<State> && requires(State& state) {
  { state.release() } noexcept;
};

template <class Tensor> class borrowed_tensor_access_state {
  public:
    explicit borrowed_tensor_access_state(Tensor& tensor) noexcept : tensor_(&tensor) {}

    borrowed_tensor_access_state(borrowed_tensor_access_state const&) = delete;
    borrowed_tensor_access_state& operator=(borrowed_tensor_access_state const&) = delete;

    borrowed_tensor_access_state(borrowed_tensor_access_state&& other) noexcept
        : tensor_(std::exchange(other.tensor_, nullptr))
    {}

    borrowed_tensor_access_state& operator=(borrowed_tensor_access_state&& other) noexcept
    {
      if (this != &other) tensor_ = std::exchange(other.tensor_, nullptr);
      return *this;
    }

    void release() noexcept { tensor_ = nullptr; }

    [[nodiscard]] decltype(auto) storage()
      requires requires(Tensor& tensor) { tensor.storage(); }
    {
      CHECK(tensor_ != nullptr, "cannot inspect storage through a released tensor lease");
      return tensor_->storage();
    }

    [[nodiscard]] decltype(auto) storage() const
      requires requires(Tensor const& tensor) { tensor.storage(); }
    {
      CHECK(tensor_ != nullptr, "cannot inspect storage through a released tensor lease");
      return std::as_const(*tensor_).storage();
    }

  private:
    Tensor* tensor_ = nullptr;
};

template <class Tensor>
concept ImmediatelyAccessibleTensorView =
    DeviceTensorView<Tensor> && SpanLike<normalized_const_device_mdspan_t<Tensor>> &&
    SpanLike<normalized_mutable_device_mdspan_t<Tensor>> &&
    requires(std::remove_reference_t<Tensor> const& tensor) { tensor.storage(); };

} // namespace detail

/// \brief Move-only RAII lease exposing a read-only immediate TensorView.
/// \details The access state is stored before the mdspan so the mdspan is
///          destroyed before the state that makes its data handle usable.
///          Independent lease types may model `ReadTensorLease` without using
///          this class template.
/// \tparam Mdspan Read-only resolved mdspan type.
/// \tparam AccessState RAII state retaining the acquired data handle.
/// \tparam BackendSelector Backend list copied from the source tensor view.
/// \tparam StoragePolicy Storage policy associated with the source tensor view.
template <SpanLike Mdspan, detail::TensorAccessState AccessState, class BackendSelector, class StoragePolicy>
  requires std::is_const_v<typename Mdspan::element_type>
class read_tensor_lease {
  public:
    using mdspan_type = Mdspan;
    using access_state_type = AccessState;
    using backend_selector_type = BackendSelector;
    using storage_policy = StoragePolicy;
    using extents_type = typename mdspan_type::extents_type;
    using index_type = typename mdspan_type::index_type;

    /// \brief Construct an active read lease.
    read_tensor_lease(access_state_type state, mdspan_type mdspan, backend_selector_type selector)
        : state_(std::move(state)), mdspan_(std::move(mdspan)), selector_(std::move(selector))
    {}

    read_tensor_lease(read_tensor_lease const&) = delete;
    read_tensor_lease& operator=(read_tensor_lease const&) = delete;

    read_tensor_lease(read_tensor_lease&& other) noexcept(std::is_nothrow_move_constructible_v<access_state_type> &&
                                                          std::is_nothrow_move_constructible_v<mdspan_type> &&
                                                          std::is_nothrow_move_constructible_v<backend_selector_type>)
        : state_(std::move(other.state_)), mdspan_(std::move(other.mdspan_)), selector_(std::move(other.selector_))
    {
      other.mdspan_.reset();
    }

    read_tensor_lease&
    operator=(read_tensor_lease&& other) noexcept(std::is_nothrow_move_assignable_v<access_state_type> &&
                                                  std::is_nothrow_move_assignable_v<mdspan_type> &&
                                                  std::is_nothrow_move_assignable_v<backend_selector_type>)
      requires(std::is_move_assignable_v<access_state_type> && std::is_move_assignable_v<mdspan_type> &&
               std::is_move_assignable_v<backend_selector_type>)
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
    void release() noexcept
    {
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
///          resolves a distinct const-element mdspan, preserving TensorView
///          shallow-const rules.
/// \tparam MutableMdspan Writable resolved mdspan type.
/// \tparam ConstMdspan Read-only resolved mdspan type.
/// \tparam AccessState RAII state retaining exclusive access.
/// \tparam BackendSelector Backend list copied from the source tensor view.
/// \tparam StoragePolicy Storage policy associated with the source tensor view.
template <MutableSpanLike MutableMdspan, SpanLike ConstMdspan, detail::TensorAccessState AccessState,
          class BackendSelector, class StoragePolicy>
  requires std::is_const_v<typename ConstMdspan::element_type>
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
                       backend_selector_type selector)
        : state_(std::move(state)), mdspan_(std::move(mdspan)), const_mdspan_(std::move(const_mdspan)),
          selector_(std::move(selector))
    {}

    write_tensor_lease(write_tensor_lease const&) = delete;
    write_tensor_lease& operator=(write_tensor_lease const&) = delete;

    write_tensor_lease(write_tensor_lease&& other) noexcept(std::is_nothrow_move_constructible_v<access_state_type> &&
                                                            std::is_nothrow_move_constructible_v<mdspan_type> &&
                                                            std::is_nothrow_move_constructible_v<const_mdspan_type> &&
                                                            std::is_nothrow_move_constructible_v<backend_selector_type>)
        : state_(std::move(other.state_)), mdspan_(std::move(other.mdspan_)),
          const_mdspan_(std::move(other.const_mdspan_)), selector_(std::move(other.selector_))
    {
      other.const_mdspan_.reset();
      other.mdspan_.reset();
    }

    write_tensor_lease&
    operator=(write_tensor_lease&& other) noexcept(std::is_nothrow_move_assignable_v<access_state_type> &&
                                                   std::is_nothrow_move_assignable_v<mdspan_type> &&
                                                   std::is_nothrow_move_assignable_v<const_mdspan_type> &&
                                                   std::is_nothrow_move_assignable_v<backend_selector_type>)
      requires(std::is_move_assignable_v<access_state_type> && std::is_move_assignable_v<mdspan_type> &&
               std::is_move_assignable_v<const_mdspan_type> && std::is_move_assignable_v<backend_selector_type>)
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

    /// \brief Return mutable storage retained by the access state.
    [[nodiscard]] decltype(auto) storage()
      requires requires(access_state_type& state) { state.storage(); }
    {
      return state_.storage();
    }

    /// \brief Return read-only storage retained by the access state.
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
    void release() noexcept
    {
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

/// \brief TensorView that owns a move-only read access lifetime.
template <class Lease>
concept ReadTensorLease = TensorView<Lease> && std::move_constructible<Lease> && (!std::copy_constructible<Lease>) &&
                          requires(Lease& lease, Lease const& const_lease) {
                            { lease.release() } noexcept;
                            const_lease.storage();
                          };

/// \brief MutableTensorView that owns a move-only write access lifetime.
template <class Lease>
concept WriteTensorLease =
    ReadTensorLease<Lease> && MutableTensorView<Lease> && requires(Lease& lease) { lease.storage(); };

/// \brief Always-ready awaitable that transfers one tensor access lease.
template <class Lease> class ready_tensor_access {
  public:
    using lease_type = Lease;

    explicit ready_tensor_access(lease_type lease) : lease_(std::move(lease)) {}

    ready_tensor_access(ready_tensor_access const&) = delete;
    ready_tensor_access& operator=(ready_tensor_access const&) = delete;
    ready_tensor_access(ready_tensor_access&&) = default;
    ready_tensor_access& operator=(ready_tensor_access&&) = default;

    [[nodiscard]] bool await_ready() const noexcept { return true; }

    template <class Task> void await_suspend(Task&&) const noexcept {}

    [[nodiscard]] lease_type await_resume() { return std::move(lease_); }

  private:
    lease_type lease_;
};

/// \brief Acquire an immediate tensor view for read-only use.
template <detail::ImmediatelyAccessibleTensorView Tensor> [[nodiscard]] auto blocking_read_access(Tensor const& tensor)
{
  using tensor_type = std::remove_reference_t<Tensor>;
  using state_type = detail::borrowed_tensor_access_state<tensor_type const>;
  using mdspan_type = std::remove_cvref_t<decltype(detail::tensor_device_mdspan(tensor))>;
  using selector_type = std::remove_cvref_t<decltype(tensor.backend_selector())>;
  using storage_policy = detail::tensor_storage_policy_t<tensor_type>;
  return read_tensor_lease<mdspan_type, state_type, selector_type, storage_policy>{
      state_type{tensor}, detail::tensor_device_mdspan(tensor), tensor.backend_selector()};
}

/// \brief Acquire an immediate mutable tensor view.
template <class Tensor>
  requires(detail::ImmediatelyAccessibleTensorView<Tensor> && MutableDeviceTensorView<Tensor> &&
           requires(std::remove_reference_t<Tensor>& tensor) { tensor.storage(); })
[[nodiscard]] auto blocking_write_access(Tensor& tensor)
{
  using tensor_type = std::remove_reference_t<Tensor>;
  using state_type = detail::borrowed_tensor_access_state<tensor_type>;
  using mdspan_type = std::remove_cvref_t<decltype(detail::tensor_device_mdspan(tensor))>;
  using const_mdspan_type = std::remove_cvref_t<decltype(detail::tensor_device_mdspan(std::as_const(tensor)))>;
  using selector_type = std::remove_cvref_t<decltype(tensor.backend_selector())>;
  using storage_policy = detail::tensor_storage_policy_t<tensor_type>;
  return write_tensor_lease<mdspan_type, const_mdspan_type, state_type, selector_type, storage_policy>{
      state_type{tensor}, detail::tensor_device_mdspan(tensor), detail::tensor_device_mdspan(std::as_const(tensor)),
      tensor.backend_selector()};
}

/// \brief Return an immediately-ready read acquisition for an immediate TensorView.
template <detail::ImmediatelyAccessibleTensorView Tensor> [[nodiscard]] auto read_access(Tensor const& tensor)
{
  return ready_tensor_access{blocking_read_access(tensor)};
}

/// \brief Return an immediately-ready write acquisition for an immediate mutable TensorView.
template <class Tensor>
  requires(detail::ImmediatelyAccessibleTensorView<Tensor> && MutableDeviceTensorView<Tensor> &&
           requires(std::remove_reference_t<Tensor>& tensor) { tensor.storage(); })
[[nodiscard]] auto write_access(Tensor& tensor)
{
  return ready_tensor_access{blocking_write_access(tensor)};
}

} // namespace uni20
