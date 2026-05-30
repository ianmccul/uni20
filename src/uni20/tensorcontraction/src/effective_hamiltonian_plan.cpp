#include <uni20/tensorcontraction/effective_hamiltonian_plan.hpp>

#include "Arranger.hpp"
#include "Swapper.hpp"
#include "Utils.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace uni20::tensorcontraction
{

struct EffectiveHamiltonianPlan::Impl
{
    MatrixFamily r_mats;
    MatrixFamily a_mats;
    MatrixFamily b_mats;
    MatrixFamily c_mats;
    std::vector<Term> terms;
    tensor::Swapper swapper;
    tensor::Arranger arranger;
    bool is_compiled = false;

    Impl(MatrixFamily r, MatrixFamily a, MatrixFamily b, MatrixFamily c, std::span<Term const> input_terms)
        : r_mats(std::move(r)), a_mats(std::move(a)), b_mats(std::move(b)), c_mats(std::move(c)),
          terms(input_terms.begin(), input_terms.end()), swapper(), arranger(swapper)
    {}

    ~Impl()
    {
      arranger.clear();
      swapper.clear();
    }
};

namespace
{

int checked_index(std::size_t value)
{
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    throw std::length_error("TensorContraction term index must fit in int");
  }
  return static_cast<int>(value);
}

void validate_index(std::size_t index, std::size_t size, char family)
{
  if (index >= size)
  {
    throw std::out_of_range(std::string("TensorContraction term references missing ") + family + " block");
  }
}

std::vector<tensor::TermTy> convert_terms(std::span<EffectiveHamiltonianPlan::Term const> terms)
{
  std::vector<tensor::TermTy> converted;
  converted.reserve(terms.size());
  for (auto const& term : terms)
  {
    converted.emplace_back(checked_index(term.r), checked_index(term.a), checked_index(term.b), checked_index(term.c),
                           term.coefficient);
  }
  return converted;
}

void validate_term_shapes(MatrixFamily const& r_mats, MatrixFamily const& a_mats, MatrixFamily const& b_mats,
                          MatrixFamily const& c_mats, std::span<EffectiveHamiltonianPlan::Term const> terms)
{
  for (auto const& term : terms)
  {
    validate_index(term.r, r_mats.size(), 'R');
    validate_index(term.a, a_mats.size(), 'A');
    validate_index(term.b, b_mats.size(), 'B');
    validate_index(term.c, c_mats.size(), 'C');

    auto const r = r_mats.block(term.r);
    auto const a = a_mats.block(term.a);
    auto const b = b_mats.block(term.b);
    auto const c = c_mats.block(term.c);

    if (b.cols != c.rows)
    {
      throw std::invalid_argument("TensorContraction term has incompatible B/C dimensions");
    }
    if (a.cols != b.rows)
    {
      throw std::invalid_argument("TensorContraction term has incompatible A/(B*C) dimensions");
    }
    if (r.rows != a.rows || r.cols != c.cols)
    {
      throw std::invalid_argument("TensorContraction term result dimensions do not match R block");
    }
  }
}

} // namespace

EffectiveHamiltonianPlan::EffectiveHamiltonianPlan(MatrixFamily r_mats, MatrixFamily a_mats, MatrixFamily b_mats,
                                                   MatrixFamily c_mats, std::span<Term const> terms)
    : impl_(nullptr)
{
  validate_term_shapes(r_mats, a_mats, b_mats, c_mats, terms);
  impl_ = std::make_unique<Impl>(std::move(r_mats), std::move(a_mats), std::move(b_mats), std::move(c_mats), terms);
}

EffectiveHamiltonianPlan::EffectiveHamiltonianPlan(EffectiveHamiltonianPlan&&) noexcept = default;
EffectiveHamiltonianPlan& EffectiveHamiltonianPlan::operator=(EffectiveHamiltonianPlan&&) noexcept = default;
EffectiveHamiltonianPlan::~EffectiveHamiltonianPlan() = default;

std::size_t EffectiveHamiltonianPlan::term_count() const noexcept { return impl_->terms.size(); }

bool EffectiveHamiltonianPlan::compiled() const noexcept { return impl_->is_compiled; }

void EffectiveHamiltonianPlan::compile()
{
  if (impl_->is_compiled)
  {
    return;
  }

  validate_term_shapes(impl_->r_mats, impl_->a_mats, impl_->b_mats, impl_->c_mats, impl_->terms);
  auto terms = convert_terms(impl_->terms);
  auto const& r = raw_matrices(impl_->r_mats);
  auto const& a = raw_matrices(impl_->a_mats);
  auto const& b = raw_matrices(impl_->b_mats);
  auto const& c = raw_matrices(impl_->c_mats);

  impl_->arranger.analyzeComputation(r, a, b, c, terms);
  impl_->arranger.compileWorklists(r, a, b, c);
  impl_->is_compiled = true;
}

} // namespace uni20::tensorcontraction
