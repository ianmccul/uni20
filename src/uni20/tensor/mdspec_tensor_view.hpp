#pragma once

/**
 * \file mdspec_tensor_view.hpp
 * \ingroup tensor
 * \brief Concrete tensor-level views over owned mdspan or mdspec metadata.
 */

#include <uni20/tensor/concepts.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief TensorView materialization over writable and read-only mdspec values.
/// \details The view owns multidimensional metadata but not the described
///          element storage. Its storage policy supplies tensor-level backend
///          selection; access acquisition remains governed by the stored
///          mdspec data descriptor.
/// \tparam MutableMdspec Metadata exposed through the mutable interface.
/// \tparam ConstMdspec Read-only metadata exposed through the const interface.
/// \tparam StoragePolicy Dense storage policy associated with the metadata.
template <MdspecLike MutableMdspec, MdspecLike ConstMdspec, class StoragePolicy>
  requires(std::same_as<typename MutableMdspec::extents_type, typename ConstMdspec::extents_type> &&
           std::same_as<std::remove_const_t<typename MutableMdspec::element_type>,
                        std::remove_const_t<typename ConstMdspec::element_type>> &&
           std::is_const_v<typename ConstMdspec::element_type> && requires { StoragePolicy::backend_selector(); })
class MdspecTensorView {
  public:
    using mutable_mdspec_type = MutableMdspec;
    using const_mdspec_type = ConstMdspec;
    using storage_policy = StoragePolicy;
    using backend_selector_type = std::remove_cvref_t<decltype(storage_policy::backend_selector())>;
    using element_type = typename mutable_mdspec_type::element_type;
    using value_type = std::remove_cv_t<element_type>;
    using extents_type = typename mutable_mdspec_type::extents_type;
    using index_type = typename extents_type::index_type;

    /// \brief Construct a tensor view from writable and read-only metadata.
    constexpr MdspecTensorView(mutable_mdspec_type mutable_mdspec, const_mdspec_type const_mdspec)
        : mutable_mdspec_(std::move(mutable_mdspec)), const_mdspec_(std::move(const_mdspec))
    {}

    /// \brief Return the dense storage policy's default backend selector.
    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return storage_policy::backend_selector();
    }

    /// \brief Return writable immediate or descriptor-backed metadata.
    [[nodiscard]] constexpr auto mdspec() & -> mutable_mdspec_type { return mutable_mdspec_; }

    /// \brief Return read-only immediate or descriptor-backed metadata.
    [[nodiscard]] constexpr auto mdspec() const& -> const_mdspec_type { return const_mdspec_; }

    /// \brief Return the writable mdspan when the metadata is immediate.
    [[nodiscard]] constexpr auto mdspan() & -> mutable_mdspec_type
      requires MdspanLike<mutable_mdspec_type>
    {
      return mutable_mdspec_;
    }

    /// \brief Return the read-only mdspan when the metadata is immediate.
    [[nodiscard]] constexpr auto mdspan() const& -> const_mdspec_type
      requires MdspanLike<const_mdspec_type>
    {
      return const_mdspec_;
    }

    /// \brief Return the static dense-block rank.
    [[nodiscard]] static constexpr std::size_t rank() noexcept { return extents_type::rank(); }

    /// \brief Return the dense-block extents.
    [[nodiscard]] constexpr auto extents() const noexcept -> extents_type const& { return const_mdspec_.extents(); }

    /// \brief Return one dense-block extent.
    [[nodiscard]] constexpr auto extent(std::size_t axis) const noexcept -> index_type
    {
      return const_mdspec_.extent(axis);
    }

    /// \brief Return one dense-block stride when the mapping is strided.
    [[nodiscard]] constexpr auto stride(std::size_t axis) const -> index_type
      requires StridedMdspecLike<const_mdspec_type>
    {
      return const_mdspec_.stride(axis);
    }

    /// \brief Evaluate or assign one element when writable metadata is immediate.
    template <class... Index>
      requires MdspanLike<mutable_mdspec_type> && (sizeof...(Index) == extents_type::rank())
    constexpr decltype(auto) operator[](Index... indices)
    {
      return mutable_mdspec_[indices...];
    }

    /// \brief Evaluate one element when read-only metadata is immediate.
    template <class... Index>
      requires MdspanLike<const_mdspec_type> && (sizeof...(Index) == extents_type::rank())
    constexpr decltype(auto) operator[](Index... indices) const
    {
      return const_mdspec_[indices...];
    }

  private:
    [[no_unique_address]] mutable_mdspec_type mutable_mdspec_;
    [[no_unique_address]] const_mdspec_type const_mdspec_;
};

} // namespace uni20
