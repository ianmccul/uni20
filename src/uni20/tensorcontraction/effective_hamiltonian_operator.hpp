#pragma once

#include <uni20/tensorcontraction/effective_hamiltonian_plan.hpp>
#include <uni20/tensorcontraction/matrix_family.hpp>

#include <cstddef>
#include <memory>
#include <span>

namespace uni20::tensorcontraction
{

class EffectiveHamiltonianOperator {
  public:
    using Term = EffectiveHamiltonianPlan::Term;

    EffectiveHamiltonianOperator(MatrixFamily a_mats, MatrixFamily b_mats,
                                 std::span<MatrixFamily::Block const> input_blocks,
                                 std::span<MatrixFamily::Block const> output_blocks, std::span<Term const> terms);
    EffectiveHamiltonianOperator(EffectiveHamiltonianOperator&&) noexcept;
    EffectiveHamiltonianOperator& operator=(EffectiveHamiltonianOperator&&) noexcept;
    EffectiveHamiltonianOperator(EffectiveHamiltonianOperator const&) = delete;
    EffectiveHamiltonianOperator& operator=(EffectiveHamiltonianOperator const&) = delete;
    ~EffectiveHamiltonianOperator();

    [[nodiscard]] std::size_t term_count() const noexcept;
    [[nodiscard]] bool compiled() const noexcept;
    [[nodiscard]] MatrixFamily make_input_vector() const;
    [[nodiscard]] MatrixFamily make_output_vector() const;

    void compile();
    void apply(MatrixFamily const& x, MatrixFamily& y);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace uni20::tensorcontraction
