/**
 * \file morphism_boundary.hpp
 * \ingroup symmetry
 * \brief Defines ordered domain and codomain space values for tensor morphisms.
 */

#pragma once

#include <uni20/symmetry/space.hpp>

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Ordered domain spaces of a tensor morphism.
/// \details Space structure is exposed read-only. Leg labels may be changed
///          explicitly with `set_label<I>()`.
/// \tparam Spaces Concrete space value types in domain order.
template <Space... Spaces> class Domain {
  public:
    /// \brief Tuple containing the ordered domain-space types.
    using tuple_type = std::tuple<Spaces...>;

    /// \brief Type of one domain space.
    /// \tparam I Zero-based position in the domain.
    template <std::size_t I> using space_type = std::tuple_element_t<I, tuple_type>;

    /// \brief Construct a domain from ordered space values.
    /// \param spaces Space values in domain order.
    explicit Domain(Spaces... spaces) : spaces_(std::move(spaces)...) {}

    /// \brief Return the number of domain factors.
    /// \return Compile-time domain size.
    static constexpr auto size() noexcept -> std::size_t { return sizeof...(Spaces); }

    /// \brief Return whether the domain is the tensor unit.
    /// \return `true` when the domain contains no factors.
    static constexpr auto empty() noexcept -> bool { return sizeof...(Spaces) == 0; }

    /// \brief Return one domain space by position.
    /// \tparam I Zero-based position in the domain.
    /// \return Read-only space value.
    template <std::size_t I> constexpr auto space() const& noexcept -> space_type<I> const&
    {
      return std::get<I>(spaces_);
    }

    /// \brief Return the ordered domain-space tuple.
    /// \return Read-only tuple of space values.
    constexpr auto spaces() const& noexcept -> tuple_type const& { return spaces_; }

    /// \brief Replace one domain leg label without changing its structure.
    /// \tparam I Zero-based position in the domain.
    /// \param label New semantic leg label.
    template <std::size_t I> void set_label(std::string label) { std::get<I>(spaces_).set_label(std::move(label)); }

    /// \brief Compare ordered domain spaces for exact equality.
    /// \param other Other domain value.
    /// \return `true` when all space structures and labels are equal.
    auto operator==(Domain const& other) const -> bool = default;

  private:
    tuple_type spaces_;
};

/// \brief Deduce a domain's concrete value types.
template <class... Spaces> Domain(Spaces&&...) -> Domain<std::remove_cvref_t<Spaces>...>;

/// \brief Ordered codomain spaces of a tensor morphism.
/// \details Space structure is exposed read-only. Leg labels may be changed
///          explicitly with `set_label<I>()`.
/// \tparam Spaces Concrete space value types in codomain order.
template <Space... Spaces> class Codomain {
  public:
    /// \brief Tuple containing the ordered codomain-space types.
    using tuple_type = std::tuple<Spaces...>;

    /// \brief Type of one codomain space.
    /// \tparam I Zero-based position in the codomain.
    template <std::size_t I> using space_type = std::tuple_element_t<I, tuple_type>;

    /// \brief Construct a codomain from ordered space values.
    /// \param spaces Space values in codomain order.
    explicit Codomain(Spaces... spaces) : spaces_(std::move(spaces)...) {}

    /// \brief Return the number of codomain factors.
    /// \return Compile-time codomain size.
    static constexpr auto size() noexcept -> std::size_t { return sizeof...(Spaces); }

    /// \brief Return whether the codomain is the tensor unit.
    /// \return `true` when the codomain contains no factors.
    static constexpr auto empty() noexcept -> bool { return sizeof...(Spaces) == 0; }

    /// \brief Return one codomain space by position.
    /// \tparam I Zero-based position in the codomain.
    /// \return Read-only space value.
    template <std::size_t I> constexpr auto space() const& noexcept -> space_type<I> const&
    {
      return std::get<I>(spaces_);
    }

    /// \brief Return the ordered codomain-space tuple.
    /// \return Read-only tuple of space values.
    constexpr auto spaces() const& noexcept -> tuple_type const& { return spaces_; }

    /// \brief Replace one codomain leg label without changing its structure.
    /// \tparam I Zero-based position in the codomain.
    /// \param label New semantic leg label.
    template <std::size_t I> void set_label(std::string label) { std::get<I>(spaces_).set_label(std::move(label)); }

    /// \brief Compare ordered codomain spaces for exact equality.
    /// \param other Other codomain value.
    /// \return `true` when all space structures and labels are equal.
    auto operator==(Codomain const& other) const -> bool = default;

  private:
    tuple_type spaces_;
};

/// \brief Deduce a codomain's concrete value types.
template <class... Spaces> Codomain(Spaces&&...) -> Codomain<std::remove_cvref_t<Spaces>...>;

} // namespace uni20
