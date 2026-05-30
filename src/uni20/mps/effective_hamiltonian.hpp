/**
 * \file effective_hamiltonian.hpp
 * \brief Compile dense two-site effective Hamiltonians for the first DMRG prototype.
 */

#pragma once

#include <uni20/mps/environment.hpp>
#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace uni20
{

struct TwoSiteEffectiveHamiltonianLayout
{
    std::size_t left_bond_dim = 0;
    std::size_t left_physical_dim = 0;
    std::size_t right_physical_dim = 0;
    std::size_t right_bond_dim = 0;

    [[nodiscard]] std::size_t vector_size() const
    {
      return left_bond_dim * left_physical_dim * right_physical_dim * right_bond_dim;
    }

    [[nodiscard]] std::size_t index(std::size_t left_bond, std::size_t left_phys, std::size_t right_phys,
                                    std::size_t right_bond) const
    {
      return (((left_bond * left_physical_dim + left_phys) * right_physical_dim + right_phys) * right_bond_dim) +
             right_bond;
    }
};

struct TwoSiteEffectiveHamiltonian
{
    tensorcontraction::EffectiveHamiltonianOperator op;
    TwoSiteEffectiveHamiltonianLayout input_layout;
    TwoSiteEffectiveHamiltonianLayout output_layout;
};

inline void validate_two_site_effective_hamiltonian_inputs(MpoEnvironment const& left_env,
                                                           OperatorComponent const& left_component,
                                                           OperatorComponent const& right_component,
                                                           MpoEnvironment const& right_env)
{
  if (left_env.virtual_space() != left_component.left_virtual_space())
  {
    throw std::invalid_argument("two-site effective Hamiltonian left environment does not match left MPO site");
  }
  if (left_component.right_virtual_space() != right_component.left_virtual_space())
  {
    throw std::invalid_argument("two-site effective Hamiltonian MPO virtual spaces do not match");
  }
  if (right_env.virtual_space() != right_component.right_virtual_space())
  {
    throw std::invalid_argument("two-site effective Hamiltonian right environment does not match right MPO site");
  }
  if (left_component.local_bra_space().size() != left_component.local_ket_space().size() ||
      right_component.local_bra_space().size() != right_component.local_ket_space().size())
  {
    throw std::invalid_argument("two-site effective Hamiltonian currently requires square local spaces");
  }
}

inline auto make_two_site_vector_layout(MpoEnvironment const& left_env, OperatorComponent const& left_component,
                                        OperatorComponent const& right_component,
                                        MpoEnvironment const& right_env) -> TwoSiteEffectiveHamiltonianLayout
{
  return TwoSiteEffectiveHamiltonianLayout{.left_bond_dim = left_env.bond_dim(),
                                           .left_physical_dim = left_component.local_ket_space().size(),
                                           .right_physical_dim = right_component.local_ket_space().size(),
                                           .right_bond_dim = right_env.bond_dim()};
}

inline auto make_two_site_effective_hamiltonian(MpoEnvironment const& left_env, OperatorComponent const& left_component,
                                                OperatorComponent const& right_component,
                                                MpoEnvironment const& right_env) -> TwoSiteEffectiveHamiltonian
{
  validate_two_site_effective_hamiltonian_inputs(left_env, left_component, right_component, right_env);

  auto const input_layout = make_two_site_vector_layout(left_env, left_component, right_component, right_env);
  auto const output_layout = input_layout;
  auto const size = input_layout.vector_size();
  if (size == 0)
  {
    throw std::invalid_argument("two-site effective Hamiltonian requires a non-empty vector space");
  }

  std::array dense_block{tensorcontraction::MatrixFamily::Block{size, size}};
  tensorcontraction::MatrixFamily hamiltonian(dense_block);
  auto h_values = hamiltonian.values(0);

  for (std::size_t mpo_left = 0; mpo_left < left_component.rows(); ++mpo_left)
  {
    auto const left_env_values = left_env.values(mpo_left);
    for (auto const& left_mpo_entry : left_component.data().row(mpo_left))
    {
      auto const mpo_middle = left_mpo_entry.column;
      auto const& left_op = left_mpo_entry.value;
      for (auto const& right_mpo_entry : right_component.data().row(mpo_middle))
      {
        auto const mpo_right = right_mpo_entry.column;
        auto const right_env_values = right_env.values(mpo_right);
        auto const& right_op = right_mpo_entry.value;

        for (std::size_t left_bra = 0; left_bra < left_env.bond_dim(); ++left_bra)
        {
          for (std::size_t left_ket = 0; left_ket < left_env.bond_dim(); ++left_ket)
          {
            double const left_env_value = left_env_values[left_bra * left_env.bond_dim() + left_ket];
            if (left_env_value == 0.0)
            {
              continue;
            }
            for (std::size_t left_phys_bra = 0; left_phys_bra < left_op.rows(); ++left_phys_bra)
            {
              for (auto const& left_op_entry : left_op.coefficients().row(left_phys_bra))
              {
                auto const left_phys_ket = left_op_entry.column;
                double const left_weight = left_env_value * left_op_entry.value;
                for (std::size_t right_phys_bra = 0; right_phys_bra < right_op.rows(); ++right_phys_bra)
                {
                  for (auto const& right_op_entry : right_op.coefficients().row(right_phys_bra))
                  {
                    auto const right_phys_ket = right_op_entry.column;
                    double const operator_weight = left_weight * right_op_entry.value;
                    for (std::size_t right_bra = 0; right_bra < right_env.bond_dim(); ++right_bra)
                    {
                      for (std::size_t right_ket = 0; right_ket < right_env.bond_dim(); ++right_ket)
                      {
                        double const right_env_value = right_env_values[right_bra * right_env.bond_dim() + right_ket];
                        if (right_env_value == 0.0)
                        {
                          continue;
                        }
                        auto const row = output_layout.index(left_bra, left_phys_bra, right_phys_bra, right_bra);
                        auto const col = input_layout.index(left_ket, left_phys_ket, right_phys_ket, right_ket);
                        h_values[row * size + col] += operator_weight * right_env_value;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  tensorcontraction::MatrixFamily identity(dense_block);
  auto id_values = identity.values(0);
  for (std::size_t i = 0; i < size; ++i)
  {
    id_values[i * size + i] = 1.0;
  }

  std::array vector_blocks{tensorcontraction::MatrixFamily::Block{size, 1}};
  std::array terms{tensorcontraction::EffectiveHamiltonianOperator::Term{0, 0, 0, 0, 1.0}};
  return TwoSiteEffectiveHamiltonian{
      .op = tensorcontraction::EffectiveHamiltonianOperator(std::move(identity), std::move(hamiltonian), vector_blocks,
                                                            vector_blocks, terms),
      .input_layout = input_layout,
      .output_layout = output_layout,
  };
}

inline void assign_two_site_matrix_to_vector(tensorcontraction::MatrixFamily const& source,
                                             tensorcontraction::MatrixFamily& target,
                                             TwoSiteEffectiveHamiltonianLayout const& layout)
{
  if (source.size() != 1 || target.size() != 1)
  {
    throw std::invalid_argument("two-site vector assignment requires single-block MatrixFamily values");
  }
  auto const source_block = source.block(0);
  if (source_block.rows != layout.left_bond_dim * layout.left_physical_dim ||
      source_block.cols != layout.right_physical_dim * layout.right_bond_dim)
  {
    throw std::invalid_argument("two-site source matrix shape does not match the vector layout");
  }
  if (target.block(0) != tensorcontraction::MatrixFamily::Block{layout.vector_size(), 1})
  {
    throw std::invalid_argument("two-site target vector shape does not match the vector layout");
  }
  std::copy(source.values(0).begin(), source.values(0).end(), target.values(0).begin());
}

inline auto make_two_site_vector(TwoSiteWavefunction const& theta,
                                 TwoSiteEffectiveHamiltonianLayout const& layout) -> tensorcontraction::MatrixFamily
{
  std::array block{tensorcontraction::MatrixFamily::Block{layout.vector_size(), 1}};
  tensorcontraction::MatrixFamily result(block);
  assign_two_site_matrix_to_vector(theta.matrix_family(), result, layout);
  return result;
}

} // namespace uni20
