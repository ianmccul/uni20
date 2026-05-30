#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>

#include "Arranger.hpp"
#include "Swapper.hpp"
#include "Utils.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uni20::tensorcontraction
{

struct EffectiveHamiltonianOperator::Impl
{
    MatrixFamily r_mats;
    MatrixFamily a_mats;
    MatrixFamily b_mats;
    MatrixFamily c_mats;
    std::vector<MatrixFamily::Block> input_blocks;
    std::vector<MatrixFamily::Block> output_blocks;
    std::vector<Term> terms;
    tensor::Swapper swapper;
    tensor::Arranger arranger;
    bool is_compiled = false;

    Impl(MatrixFamily a, MatrixFamily b, std::span<MatrixFamily::Block const> input,
         std::span<MatrixFamily::Block const> output, std::span<Term const> input_terms)
        : r_mats(output), a_mats(std::move(a)), b_mats(std::move(b)), c_mats(input),
          input_blocks(input.begin(), input.end()), output_blocks(output.begin(), output.end()),
          terms(input_terms.begin(), input_terms.end()), swapper(), arranger(swapper)
    {}

    ~Impl() { swapper.clear(); }
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

std::vector<tensor::TermTy> convert_terms(std::span<EffectiveHamiltonianOperator::Term const> terms)
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
                          MatrixFamily const& c_mats, std::span<EffectiveHamiltonianOperator::Term const> terms)
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

void validate_family_shape(MatrixFamily const& actual, std::span<MatrixFamily::Block const> expected,
                           char const* family_name)
{
  if (actual.blocks().size() != expected.size())
  {
    throw std::invalid_argument(std::string("TensorContraction ") + family_name + " vector has the wrong block count");
  }
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    if (actual.block(i) != expected[i])
    {
      throw std::invalid_argument(std::string("TensorContraction ") + family_name +
                                  " vector has incompatible block shapes");
    }
  }
}

} // namespace

EffectiveHamiltonianOperator::EffectiveHamiltonianOperator(MatrixFamily a_mats, MatrixFamily b_mats,
                                                           std::span<MatrixFamily::Block const> input_blocks,
                                                           std::span<MatrixFamily::Block const> output_blocks,
                                                           std::span<Term const> terms)
    : impl_(nullptr)
{
  MatrixFamily r_mats(output_blocks);
  MatrixFamily c_mats(input_blocks);
  validate_term_shapes(r_mats, a_mats, b_mats, c_mats, terms);
  impl_ = std::make_unique<Impl>(std::move(a_mats), std::move(b_mats), input_blocks, output_blocks, terms);
}

EffectiveHamiltonianOperator::EffectiveHamiltonianOperator(EffectiveHamiltonianOperator&&) noexcept = default;
EffectiveHamiltonianOperator&
EffectiveHamiltonianOperator::operator=(EffectiveHamiltonianOperator&&) noexcept = default;
EffectiveHamiltonianOperator::~EffectiveHamiltonianOperator() = default;

std::size_t EffectiveHamiltonianOperator::term_count() const noexcept { return impl_->terms.size(); }

bool EffectiveHamiltonianOperator::compiled() const noexcept { return impl_->is_compiled; }

MatrixFamily EffectiveHamiltonianOperator::make_input_vector() const { return MatrixFamily(impl_->input_blocks); }

MatrixFamily EffectiveHamiltonianOperator::make_output_vector() const { return MatrixFamily(impl_->output_blocks); }

void EffectiveHamiltonianOperator::compile()
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

  impl_->arranger.resetWork();
  impl_->arranger.analyzeComputation(r, a, b, c, terms);
  impl_->arranger.compileWorklists(r, a, b, c);
  impl_->is_compiled = true;
}

void EffectiveHamiltonianOperator::apply(MatrixFamily const& x, MatrixFamily& y)
{
  validate_family_shape(x, impl_->input_blocks, "input");
  validate_family_shape(y, impl_->output_blocks, "output");

  if (!impl_->is_compiled)
  {
    compile();
  }

  impl_->c_mats.assign(x);
  impl_->r_mats.fill(0.0);
  impl_->arranger.doContraction(raw_matrices(impl_->r_mats), raw_matrices(impl_->a_mats), raw_matrices(impl_->b_mats),
                                raw_matrices(impl_->c_mats));
  y.assign(impl_->r_mats);
}

} // namespace uni20::tensorcontraction
