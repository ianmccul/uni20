/**
 * \file rabc_contraction_plan.hpp
 * \ingroup tensor_network
 * \brief Sparse logical plans for R/A/B/C block contractions.
 */

#pragma once

#include <uni20/core/scalar_concepts.hpp>

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::tensor_network
{

/// \brief One nonzero coefficient in `R_r += f(r,a,b,c) A_a B_b transpose(C_c)`.
/// \details Each index addresses an immutable logical-key table owned by the
///          surrounding `RabcContractionPlan`; it is not a storage ordinal.
/// \tparam Scalar Scalar coefficient type.
template <uni20::Scalar Scalar> struct RabcTerm
{
    std::size_t r_key_index;
    std::size_t a_key_index;
    std::size_t b_key_index;
    std::size_t c_key_index;
    Scalar coefficient;
};

namespace detail
{

template <uni20::Scalar Scalar> [[nodiscard]] constexpr auto rabc_term_key(RabcTerm<Scalar> const& term) noexcept
{
  return std::tuple{term.r_key_index, term.a_key_index, term.b_key_index, term.c_key_index};
}

template <class Key> void require_canonical_rabc_keys(std::vector<Key> const& keys)
{
  if (!std::ranges::is_sorted(keys) || std::ranges::adjacent_find(keys) != keys.end())
    throw std::invalid_argument("R/A/B/C logical key tables must be sorted and unique");
}

} // namespace detail

/// \brief Immutable sparse coefficient tensor for an R/A/B/C contraction.
/// \details The four sorted key tables provide logical block identity relative
///          to the symmetry and boundaries for which the caller constructs the
///          plan. Terms index those tables and are canonicalized by
///          `(r,a,b,c)`; duplicate coefficients are summed and exact zeros are
///          omitted. Storage ordinals and placement are resolved only during
///          backend preparation. The plan contains no left-first, right-first,
///          placement, or communication decisions. Binding a plan to operands
///          with different symmetry or boundary semantics violates the
///          contraction precondition; the plan deliberately does not retain or
///          compare that metadata at runtime.
/// \tparam Scalar Scalar coefficient type.
/// \tparam RKey Output logical block-key type.
/// \tparam AKey Left-environment logical block-key type.
/// \tparam BKey Input-center logical block-key type.
/// \tparam CKey Right-environment logical block-key type.
template <uni20::Scalar Scalar, class RKey, class AKey, class BKey, class CKey> class RabcContractionPlan {
  public:
    using scalar_type = Scalar;
    using r_key_type = RKey;
    using a_key_type = AKey;
    using b_key_type = BKey;
    using c_key_type = CKey;
    using term_type = RabcTerm<scalar_type>;

    /// \brief Construct an empty sparse coefficient plan.
    RabcContractionPlan() = default;

    /// \brief Retain logical key tables and canonicalize sparse coefficients.
    /// \param r_keys Sorted unique output logical keys.
    /// \param a_keys Sorted unique left-environment logical keys.
    /// \param b_keys Sorted unique input-center logical keys.
    /// \param c_keys Sorted unique right-environment logical keys.
    /// \param terms Possibly unordered terms with possible duplicate coordinates.
    RabcContractionPlan(std::vector<r_key_type> r_keys, std::vector<a_key_type> a_keys,
                        std::vector<b_key_type> b_keys, std::vector<c_key_type> c_keys,
                        std::vector<term_type> terms)
        : r_keys_(std::move(r_keys)), a_keys_(std::move(a_keys)), b_keys_(std::move(b_keys)),
          c_keys_(std::move(c_keys)), terms_(canonicalize(std::move(terms)))
    {
      detail::require_canonical_rabc_keys(r_keys_);
      detail::require_canonical_rabc_keys(a_keys_);
      detail::require_canonical_rabc_keys(b_keys_);
      detail::require_canonical_rabc_keys(c_keys_);
      for (auto const& term : terms_)
      {
        if (term.r_key_index >= r_keys_.size() || term.a_key_index >= a_keys_.size() ||
            term.b_key_index >= b_keys_.size() || term.c_key_index >= c_keys_.size())
          throw std::invalid_argument("R/A/B/C term has an out-of-range logical key index");
      }
    }

    /// \brief Return output logical keys addressed by term indices.
    [[nodiscard]] auto r_keys() const noexcept -> std::span<r_key_type const> { return r_keys_; }

    /// \brief Return left-environment logical keys addressed by term indices.
    [[nodiscard]] auto a_keys() const noexcept -> std::span<a_key_type const> { return a_keys_; }

    /// \brief Return input-center logical keys addressed by term indices.
    [[nodiscard]] auto b_keys() const noexcept -> std::span<b_key_type const> { return b_keys_; }

    /// \brief Return right-environment logical keys addressed by term indices.
    [[nodiscard]] auto c_keys() const noexcept -> std::span<c_key_type const> { return c_keys_; }

    /// \brief Return the canonical nonzero terms.
    [[nodiscard]] auto terms() const noexcept -> std::span<term_type const> { return terms_; }

    /// \brief Return the number of canonical nonzero terms.
    [[nodiscard]] auto term_count() const noexcept -> std::size_t { return terms_.size(); }

  private:
    [[nodiscard]] static auto canonicalize(std::vector<term_type> terms) -> std::vector<term_type>
    {
      std::ranges::stable_sort(terms, [](term_type const& lhs, term_type const& rhs) {
        return detail::rabc_term_key(lhs) < detail::rabc_term_key(rhs);
      });

      std::vector<term_type> result;
      result.reserve(terms.size());
      for (auto const& term : terms)
      {
        if (!result.empty() && detail::rabc_term_key(result.back()) == detail::rabc_term_key(term))
          result.back().coefficient += term.coefficient;
        else
          result.push_back(term);
      }
      std::erase_if(result, [](term_type const& term) { return term.coefficient == scalar_type{}; });
      return result;
    }

    std::vector<r_key_type> r_keys_;
    std::vector<a_key_type> a_keys_;
    std::vector<b_key_type> b_keys_;
    std::vector<c_key_type> c_keys_;
    std::vector<term_type> terms_;
};

template <class T> inline constexpr bool is_rabc_contraction_plan = false;

template <uni20::Scalar Scalar, class RKey, class AKey, class BKey, class CKey>
inline constexpr bool is_rabc_contraction_plan<RabcContractionPlan<Scalar, RKey, AKey, BKey, CKey>> = true;

/// \brief Concrete logical R/A/B/C coefficient-plan type.
template <class T>
concept RabcPlan = is_rabc_contraction_plan<std::remove_cvref_t<T>>;

} // namespace uni20::tensor_network
