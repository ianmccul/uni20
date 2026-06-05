/**
 * \file rabc_lanczos_fixture.hpp
 * \brief Binary fixtures for replaying resident R/A/B/C Lanczos solves.
 */

#pragma once

#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>
#include <uni20/tensorcontraction/matrix_family.hpp>

#include <string>
#include <vector>

namespace uni20::tensorcontraction
{

/// \brief Captured variable-middle R/A/B/C problem for standalone Lanczos replay.
/// \details The fixture stores static `A` and `C` families, the active `B` input
///          vector, output block shapes, and the `F` term list. It intentionally
///          captures TensorContraction-level data, not the higher-level MPS/MPO
///          symmetry metadata.
struct RabcLanczosFixture
{
    MatrixFamily a_mats;
    MatrixFamily c_mats;
    MatrixFamily input_vector;
    std::vector<MatrixFamily::Block> output_blocks;
    std::vector<EffectiveHamiltonianOperator::Term> terms;

    /// \brief Reconstruct the variable-middle effective Hamiltonian operator.
    /// \return Resident-capable R/A/B/C operator.
    [[nodiscard]] auto make_operator() -> EffectiveHamiltonianOperator;
};

/// \brief Capture a variable-middle R/A/B/C fixture from an operator and input vector.
/// \throws std::invalid_argument If `op` is not a variable-middle operator or `input_vector` has incompatible blocks.
/// \param op Effective Hamiltonian operator.
/// \param input_vector Active `B` vector to use as the Lanczos initial vector.
/// \return Self-contained fixture with copied host values.
[[nodiscard]] auto capture_variable_middle_rabc_fixture(EffectiveHamiltonianOperator const& op,
                                                        MatrixFamily const& input_vector) -> RabcLanczosFixture;

/// \brief Capture and write a variable-middle R/A/B/C fixture.
/// \throws std::runtime_error If the file cannot be written.
/// \param path Output fixture path.
/// \param op Effective Hamiltonian operator.
/// \param input_vector Active `B` vector to use as the Lanczos initial vector.
void write_variable_middle_rabc_fixture(std::string const& path, EffectiveHamiltonianOperator const& op,
                                        MatrixFamily const& input_vector);

/// \brief Write a captured R/A/B/C Lanczos fixture to disk.
/// \throws std::runtime_error If the file cannot be written.
/// \param path Output fixture path.
/// \param fixture Fixture to write.
void write_rabc_lanczos_fixture(std::string const& path, RabcLanczosFixture const& fixture);

/// \brief Read a captured R/A/B/C Lanczos fixture from disk.
/// \throws std::runtime_error If the file is malformed or cannot be read.
/// \param path Input fixture path.
/// \return Fixture ready for standalone replay.
[[nodiscard]] auto read_rabc_lanczos_fixture(std::string const& path) -> RabcLanczosFixture;

} // namespace uni20::tensorcontraction
