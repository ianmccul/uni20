#pragma once

/**
 * \file conjugate.hpp
 * \ingroup tensor
 * \brief Read-only lazy conjugation views for tensor-level objects.
 */

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>
#include <uni20/tensor/concepts.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{
template <class Tensor, class = void> struct TensorStoragePolicy
{
    using type = void;
};

template <class Tensor> struct TensorStoragePolicy<Tensor, std::void_t<typename Tensor::storage_policy>>
{
    using type = typename Tensor::storage_policy;
};

template <class Tensor> using tensor_storage_policy_t = typename TensorStoragePolicy<Tensor>::type;
} // namespace detail

/// \brief Non-owning read-only view of a tensor-level object.
template <TensorView Tensor> class ConstTensorView {
  public:
    using tensor_type = std::remove_cvref_t<Tensor>;
    using storage_policy = detail::tensor_storage_policy_t<tensor_type>;

    /// \brief Bind a read-only view to an existing tensor.
    explicit constexpr ConstTensorView(tensor_type const& tensor) noexcept : tensor_(std::addressof(tensor)) {}

    /// \brief Return the underlying tensor's backend selector.
    [[nodiscard]] constexpr decltype(auto) backend_selector() const
        noexcept(noexcept(std::declval<tensor_type const&>().backend_selector()))
    {
      return tensor_->backend_selector();
    }

    /// \brief Resolve the underlying tensor's read-only mdspan.
    [[nodiscard]] constexpr decltype(auto) mdspan() const
        noexcept(noexcept(std::declval<tensor_type const&>().mdspan()))
    {
      return tensor_->mdspan();
    }

    /// \brief Return the underlying tensor extents.
    [[nodiscard]] constexpr decltype(auto) extents() const
        noexcept(noexcept(std::declval<tensor_type const&>().extents()))
    {
      return tensor_->extents();
    }

    /// \brief Return one underlying tensor extent.
    [[nodiscard]] constexpr decltype(auto) extent(std::size_t axis) const
        noexcept(noexcept(std::declval<tensor_type const&>().extent(axis)))
    {
      return tensor_->extent(axis);
    }

    /// \brief Return the tensor referenced by this view.
    [[nodiscard]] constexpr tensor_type const& base() const noexcept { return *tensor_; }

  private:
    tensor_type const* tensor_;
};

/// \brief Non-owning tensor view whose resolved mdspan presents conjugated values.
/// \details The view preserves tensor metadata and backend selection while
///          delegating value transformation to `conjugated_accessor`. It is
///          read-only and may not outlive the referenced tensor.
template <TensorView Tensor> class ConjugatedTensorView {
  public:
    using tensor_type = std::remove_cvref_t<Tensor>;
    using storage_policy = detail::tensor_storage_policy_t<tensor_type>;

    /// \brief Bind a lazy conjugating view to an existing tensor.
    explicit constexpr ConjugatedTensorView(tensor_type const& tensor) noexcept : tensor_(std::addressof(tensor)) {}

    /// \brief Return the underlying tensor's backend selector.
    [[nodiscard]] constexpr decltype(auto) backend_selector() const
        noexcept(noexcept(std::declval<tensor_type const&>().backend_selector()))
    {
      return tensor_->backend_selector();
    }

    /// \brief Resolve the conjugating mdspan view.
    [[nodiscard]] constexpr auto mdspan() const
        noexcept(noexcept(uni20::conj(std::declval<tensor_type const&>().mdspan())))
    {
      return uni20::conj(tensor_->mdspan());
    }

    /// \brief Return the underlying tensor extents.
    [[nodiscard]] constexpr decltype(auto) extents() const
        noexcept(noexcept(std::declval<tensor_type const&>().extents()))
    {
      return tensor_->extents();
    }

    /// \brief Return one underlying tensor extent.
    [[nodiscard]] constexpr decltype(auto) extent(std::size_t axis) const
        noexcept(noexcept(std::declval<tensor_type const&>().extent(axis)))
    {
      return tensor_->extent(axis);
    }

    /// \brief Return the tensor referenced by this view.
    [[nodiscard]] constexpr tensor_type const& base() const noexcept { return *tensor_; }

  private:
    tensor_type const* tensor_;
};

/// \brief Return a lazy read-only conjugating view of a complex tensor lvalue.
template <TensorView Tensor>
  requires(std::is_lvalue_reference_v<Tensor &&> && Complex<tensor_element_t<Tensor>>)
[[nodiscard]] constexpr auto conj(Tensor&& tensor)
{
  return ConjugatedTensorView<std::remove_cvref_t<Tensor>>{tensor};
}

/// \brief Cancel a tensor conjugation view and return the read-only base tensor.
template <TensorView Tensor> [[nodiscard]] constexpr auto conj(ConjugatedTensorView<Tensor>& view) noexcept
{
  return ConstTensorView<typename ConjugatedTensorView<Tensor>::tensor_type>{view.base()};
}

/// \brief Cancel a const tensor conjugation view and return the read-only base tensor.
template <TensorView Tensor> [[nodiscard]] constexpr auto conj(ConjugatedTensorView<Tensor> const& view) noexcept
{
  return ConstTensorView<typename ConjugatedTensorView<Tensor>::tensor_type>{view.base()};
}

/// \brief Cancel a temporary tensor conjugation view and return its durable base tensor.
template <TensorView Tensor> [[nodiscard]] constexpr auto conj(ConjugatedTensorView<Tensor>&& view) noexcept
{
  return ConstTensorView<typename ConjugatedTensorView<Tensor>::tensor_type>{view.base()};
}

/// \brief Reapply conjugation to a read-only complex tensor view.
template <TensorView Tensor>
  requires Complex<tensor_element_t<Tensor>>
[[nodiscard]] constexpr auto conj(ConstTensorView<Tensor>& view) noexcept
{
  return ConjugatedTensorView<typename ConstTensorView<Tensor>::tensor_type>{view.base()};
}

/// \brief Reapply conjugation to a const read-only complex tensor view.
template <TensorView Tensor>
  requires Complex<tensor_element_t<Tensor>>
[[nodiscard]] constexpr auto conj(ConstTensorView<Tensor> const& view) noexcept
{
  return ConjugatedTensorView<typename ConstTensorView<Tensor>::tensor_type>{view.base()};
}

/// \brief Reapply conjugation to a temporary read-only complex tensor view.
template <TensorView Tensor>
  requires Complex<tensor_element_t<Tensor>>
[[nodiscard]] constexpr auto conj(ConstTensorView<Tensor>&& view) noexcept
{
  return ConjugatedTensorView<typename ConstTensorView<Tensor>::tensor_type>{view.base()};
}

/// \brief Return a read-only identity view of a non-complex tensor lvalue.
template <TensorView Tensor>
  requires(std::is_lvalue_reference_v<Tensor &&> && !Complex<tensor_element_t<Tensor>>)
[[nodiscard]] constexpr auto conj(Tensor&& tensor) noexcept -> std::remove_reference_t<Tensor> const&
{
  return tensor;
}

} // namespace uni20
