#pragma once

#include <uni20/tensorcontraction/effective_hamiltonian_plan.hpp>
#include <uni20/tensorcontraction/matrix_family.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace uni20::tensorcontraction
{

class VectorAlgebraEngine;
struct RabcLanczosFixture;

class EffectiveHamiltonianOperator {
  public:
    using Term = EffectiveHamiltonianPlan::Term;

    EffectiveHamiltonianOperator(MatrixFamily a_mats, MatrixFamily b_mats,
                                 std::span<MatrixFamily::Block const> input_blocks,
                                 std::span<MatrixFamily::Block const> output_blocks, std::span<Term const> terms);
    static auto variable_middle(MatrixFamily a_mats, MatrixFamily c_mats,
                                std::span<MatrixFamily::Block const> input_blocks,
                                std::span<MatrixFamily::Block const> output_blocks,
                                std::span<Term const> terms) -> EffectiveHamiltonianOperator;
    EffectiveHamiltonianOperator(EffectiveHamiltonianOperator&&) noexcept;
    EffectiveHamiltonianOperator& operator=(EffectiveHamiltonianOperator&&) noexcept;
    EffectiveHamiltonianOperator(EffectiveHamiltonianOperator const&) = delete;
    EffectiveHamiltonianOperator& operator=(EffectiveHamiltonianOperator const&) = delete;
    ~EffectiveHamiltonianOperator();

    [[nodiscard]] std::size_t term_count() const noexcept;
    [[nodiscard]] bool compiled() const noexcept;
    [[nodiscard]] MatrixFamily make_input_vector() const;
    [[nodiscard]] MatrixFamily make_output_vector() const;

    /// \brief Append one value-free term-structure record (the contraction
    /// f-hypergraph plus per-block dimensions) to a JSONL file at \p path.
    /// \details Emits a `kind:"rabc_matvec"` record with `terms[]` carrying
    /// r/a/b/c, coefficient, block dimensions, and derived bc_flops /
    /// accumulate_flops / intermediate_bytes — no matrix-element values, no GPU
    /// apply, and no timing. Intended for cheaply harvesting block dimensions at
    /// large bond dimension. The layout is reported as a single device.
    /// \param path JSONL output path (opened in append mode).
    /// \param index Record index written into the JSON.
    void write_term_structure(std::string const& path, std::uint64_t index) const;

    void compile();
    void apply(MatrixFamily const& x, MatrixFamily& y);
    void apply_resident(MatrixFamily const& x, MatrixFamily& y, VectorAlgebraEngine& algebra);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit EffectiveHamiltonianOperator(std::unique_ptr<Impl> impl);

    friend auto capture_variable_middle_rabc_fixture(EffectiveHamiltonianOperator const& op,
                                                     MatrixFamily const& input_vector) -> RabcLanczosFixture;
};

[[nodiscard]] auto capture_variable_middle_rabc_fixture(EffectiveHamiltonianOperator const& op,
                                                        MatrixFamily const& input_vector) -> RabcLanczosFixture;

} // namespace uni20::tensorcontraction
