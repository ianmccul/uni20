/**
 * \file space.hpp
 * \ingroup symmetry
 * \brief Defines the common tensor-space concepts.
 */

#pragma once

#include <uni20/symmetry/symmetry.hpp>

#include <concepts>
#include <string>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Tensor-space value with a mutable semantic leg label.
/// \tparam T Candidate space type.
template <class T>
concept Space = std::copy_constructible<std::remove_cvref_t<T>> && std::equality_comparable<std::remove_cvref_t<T>> &&
                requires(std::remove_cvref_t<T>& space, std::remove_cvref_t<T> const& const_space, std::string label) {
                  { const_space.label() } -> std::same_as<std::string const&>;
                  { space.set_label(std::move(label)) } -> std::same_as<void>;
                };

/// \brief Space whose structure is interpreted in one explicit symmetry context.
/// \tparam T Candidate symmetry-space type.
template <class T>
concept SymmetrySpace = Space<T> && requires(std::remove_cvref_t<T> const& space) {
  { space.symmetry() } -> std::same_as<Symmetry>;
};

} // namespace uni20
