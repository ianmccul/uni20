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
#include <tuple>
#include <utility>
#include <vector>

namespace uni20::tensor_network
{

/// \brief One nonzero coefficient in `R_r += f(r,a,b,c) A_a B_b transpose(C_c)`.
/// \tparam Scalar Scalar coefficient type.
template <uni20::Scalar Scalar> struct RabcTerm
{
    std::size_t r_ordinal;
    std::size_t a_ordinal;
    std::size_t b_ordinal;
    std::size_t c_ordinal;
    Scalar coefficient;
};

namespace detail
{

template <uni20::Scalar Scalar> [[nodiscard]] constexpr auto rabc_term_key(RabcTerm<Scalar> const& term) noexcept
{
  return std::tuple{term.r_ordinal, term.a_ordinal, term.b_ordinal, term.c_ordinal};
}

} // namespace detail

/// \brief Immutable sparse coefficient tensor for an R/A/B/C contraction.
/// \details Terms are canonicalized by `(r,a,b,c)`, duplicate coefficients
///          are summed, and exact zeros are omitted. The plan deliberately
///          contains no left-first, right-first, placement, or communication
///          decisions; execution backends derive those plans from this sparse
///          hypergraph.
/// \tparam Scalar Scalar coefficient type.
template <uni20::Scalar Scalar> class RabcContractionPlan {
  public:
    using scalar_type = Scalar;
    using term_type = RabcTerm<scalar_type>;

    /// \brief Construct an empty sparse coefficient plan.
    RabcContractionPlan() = default;

    /// \brief Canonicalize a sparse coefficient term list.
    /// \param terms Possibly unordered terms with possible duplicate coordinates.
    explicit RabcContractionPlan(std::vector<term_type> terms) : terms_(canonicalize(std::move(terms))) {}

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

    std::vector<term_type> terms_;
};

} // namespace uni20::tensor_network
