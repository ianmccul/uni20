#pragma once

/**
 * \file generated.hpp
 * \ingroup tensor
 * \brief Compact read-only tensors whose values are generated from their indices.
 */

#include <uni20/common/mdspan.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/mdspan/generated_accessor.hpp>
#include <uni20/storage/generated_storage.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/shape.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <class ElementType> struct FullGenerator
{
    ElementType value;

    template <class Indices> [[nodiscard]] constexpr ElementType operator()(Indices const&) const { return value; }
};

template <class ElementType> struct ZeroGenerator
{
    template <class Indices> [[nodiscard]] constexpr ElementType operator()(Indices const&) const
    {
      return ElementType{0};
    }
};

template <class ElementType> struct OneGenerator
{
    template <class Indices> [[nodiscard]] constexpr ElementType operator()(Indices const&) const
    {
      return ElementType{1};
    }
};

template <class ElementType> struct EyeGenerator
{
    template <class Indices> [[nodiscard]] constexpr ElementType operator()(Indices const& indices) const
    {
      if constexpr (std::tuple_size_v<std::remove_cvref_t<Indices>> > 1)
      {
        for (std::size_t axis = 1; axis < indices.size(); ++axis)
          if (indices[axis] != indices[0]) return ElementType{0};
      }
      return ElementType{1};
    }
};

} // namespace detail

/// \brief Compact read-only tensor evaluated from logical indices.
/// \details The tensor owns its extents and generator state but no dense
///          element allocation. Its generated storage is backend-neutral and
///          its resolved mdspan returns values by value.
template <Scalar ElementType, class Extents, class Generator> class GeneratedTensor {
  public:
    using element_type = ElementType;
    using value_type = element_type;
    using extents_type = Extents;
    using generator_type = Generator;
    using storage_policy = GeneratedStorage;
    using backend_selector_type = typename storage_policy::backend_selector_type;
    using index_type = typename extents_type::index_type;
    using layout_type = stdex::layout_right;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    using accessor_type = generated_accessor<element_type, extents_type, generator_type>;
    using mdspan_type = stdex::mdspan<element_type const, extents_type, layout_type, accessor_type>;

    /// \brief Construct from extents and compact generator state.
    constexpr GeneratedTensor(extents_type extents, generator_type generator)
        : extents_(std::move(extents)), generator_(std::move(generator))
    {}

    /// \brief Return the host fallback selector used when no concrete storage operand is present.
    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return storage_policy::backend_selector();
    }

    /// \brief Resolve a self-contained generated mdspan descriptor.
    [[nodiscard]] constexpr auto mdspan() const -> mdspan_type
    {
      return mdspan_type{generated_data_handle{}, mapping_type{extents_}, accessor_type{extents_, generator_}};
    }

    /// \brief Return the generated tensor extents.
    [[nodiscard]] constexpr auto extents() const noexcept -> extents_type const& { return extents_; }

    /// \brief Return one generated tensor extent.
    [[nodiscard]] constexpr auto extent(std::size_t axis) const noexcept -> index_type { return extents_.extent(axis); }

    /// \brief Return the tensor's static rank.
    [[nodiscard]] static constexpr std::size_t rank() noexcept { return extents_type::rank(); }

    /// \brief Evaluate one generated tensor element.
    template <class... Index>
      requires(sizeof...(Index) == extents_type::rank())
    [[nodiscard]] constexpr element_type operator[](Index... indices) const
    {
      return this->mdspan()[indices...];
    }

  private:
    [[no_unique_address]] extents_type extents_{};
    [[no_unique_address]] generator_type generator_{};
};

/// \brief Return a generated tensor filled with one scalar value.
template <Scalar ElementType, std::integral... Extents> [[nodiscard]] auto full(ElementType value, Extents... extents)
{
  auto shape = detail::make_tensor_extents(extents...);
  using shape_type = decltype(shape);
  using generator_type = detail::FullGenerator<ElementType>;
  return GeneratedTensor<ElementType, shape_type, generator_type>{std::move(shape), generator_type{std::move(value)}};
}

/// \brief Return a generated tensor filled with scalar zeros.
template <Scalar ElementType, std::integral... Extents> [[nodiscard]] auto zeros(Extents... extents)
{
  auto shape = detail::make_tensor_extents(extents...);
  using shape_type = decltype(shape);
  using generator_type = detail::ZeroGenerator<ElementType>;
  return GeneratedTensor<ElementType, shape_type, generator_type>{std::move(shape), generator_type{}};
}

/// \brief Return a generated tensor filled with scalar ones.
template <Scalar ElementType, std::integral... Extents> [[nodiscard]] auto ones(Extents... extents)
{
  auto shape = detail::make_tensor_extents(extents...);
  using shape_type = decltype(shape);
  using generator_type = detail::OneGenerator<ElementType>;
  return GeneratedTensor<ElementType, shape_type, generator_type>{std::move(shape), generator_type{}};
}

/// \brief Return a generated tensor equal to one exactly when all indices agree.
/// \details Rank-zero `eye` is scalar one and rank-one `eye` is an all-ones
///          vector. Extents need not be equal; the nonzero diagonal length is
///          the minimum extent.
template <Scalar ElementType, std::integral... Extents> [[nodiscard]] auto eye(Extents... extents)
{
  auto shape = detail::make_tensor_extents(extents...);
  using shape_type = decltype(shape);
  using generator_type = detail::EyeGenerator<ElementType>;
  return GeneratedTensor<ElementType, shape_type, generator_type>{std::move(shape), generator_type{}};
}

} // namespace uni20
