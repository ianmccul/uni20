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

    [[nodiscard]] std::size_t block_count() const { return left_physical_dim * right_physical_dim; }

    [[nodiscard]] std::size_t block_index(std::size_t left_phys, std::size_t right_phys) const
    {
      return left_phys * right_physical_dim + right_phys;
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

  std::vector<tensorcontraction::MatrixFamily::Block> center_blocks(input_layout.block_count());
  std::fill(center_blocks.begin(), center_blocks.end(),
            tensorcontraction::MatrixFamily::Block{left_env.bond_dim(), right_env.bond_dim()});

  struct BlockData
  {
      tensorcontraction::MatrixFamily::Block block;
      std::vector<double> values;
  };

  std::vector<BlockData> a_blocks;
  std::vector<BlockData> c_blocks;
  std::vector<tensorcontraction::EffectiveHamiltonianOperator::Term> terms;

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

        for (std::size_t left_phys_bra = 0; left_phys_bra < left_op.rows(); ++left_phys_bra)
        {
          for (auto const& left_op_entry : left_op.coefficients().row(left_phys_bra))
          {
            auto const left_phys_ket = left_op_entry.column;

            BlockData a_data{.block = tensorcontraction::MatrixFamily::Block{left_env.bond_dim(), left_env.bond_dim()},
                             .values = std::vector<double>(left_env.bond_dim() * left_env.bond_dim(), 0.0)};
            for (std::size_t left_bra = 0; left_bra < left_env.bond_dim(); ++left_bra)
            {
              for (std::size_t left_ket = 0; left_ket < left_env.bond_dim(); ++left_ket)
              {
                a_data.values[left_bra * left_env.bond_dim() + left_ket] =
                    left_env_values[left_bra * left_env.bond_dim() + left_ket] * left_op_entry.value;
              }
            }
            auto const a_index = a_blocks.size();
            a_blocks.push_back(std::move(a_data));

            for (std::size_t right_phys_bra = 0; right_phys_bra < right_op.rows(); ++right_phys_bra)
            {
              for (auto const& right_op_entry : right_op.coefficients().row(right_phys_bra))
              {
                auto const right_phys_ket = right_op_entry.column;
                BlockData c_data{.block =
                                     tensorcontraction::MatrixFamily::Block{right_env.bond_dim(), right_env.bond_dim()},
                                 .values = std::vector<double>(right_env.bond_dim() * right_env.bond_dim(), 0.0)};
                for (std::size_t right_ket = 0; right_ket < right_env.bond_dim(); ++right_ket)
                {
                  for (std::size_t right_bra = 0; right_bra < right_env.bond_dim(); ++right_bra)
                  {
                    c_data.values[right_ket * right_env.bond_dim() + right_bra] =
                        right_op_entry.value * right_env_values[right_bra * right_env.bond_dim() + right_ket];
                  }
                }
                auto const c_index = c_blocks.size();
                c_blocks.push_back(std::move(c_data));
                terms.push_back(tensorcontraction::EffectiveHamiltonianOperator::Term{
                    .r = output_layout.block_index(left_phys_bra, right_phys_bra),
                    .a = a_index,
                    .b = input_layout.block_index(left_phys_ket, right_phys_ket),
                    .c = c_index,
                    .coefficient = 1.0});
              }
            }
          }
        }
      }
    }
  }

  std::vector<tensorcontraction::MatrixFamily::Block> a_family_blocks;
  std::vector<tensorcontraction::MatrixFamily::Block> c_family_blocks;
  a_family_blocks.reserve(a_blocks.size());
  c_family_blocks.reserve(c_blocks.size());
  for (auto const& block : a_blocks)
  {
    a_family_blocks.push_back(block.block);
  }
  for (auto const& block : c_blocks)
  {
    c_family_blocks.push_back(block.block);
  }
  tensorcontraction::MatrixFamily a_mats(a_family_blocks);
  tensorcontraction::MatrixFamily c_mats(c_family_blocks);
  for (std::size_t i = 0; i < a_blocks.size(); ++i)
  {
    a_mats.assign(i, a_blocks[i].values);
  }
  for (std::size_t i = 0; i < c_blocks.size(); ++i)
  {
    c_mats.assign(i, c_blocks[i].values);
  }

  return TwoSiteEffectiveHamiltonian{
      .op = tensorcontraction::EffectiveHamiltonianOperator::variable_middle(std::move(a_mats), std::move(c_mats),
                                                                             center_blocks, center_blocks, terms),
      .input_layout = input_layout,
      .output_layout = output_layout,
  };
}

inline void assign_two_site_matrix_to_vector(tensorcontraction::MatrixFamily const& source,
                                             tensorcontraction::MatrixFamily& target,
                                             TwoSiteEffectiveHamiltonianLayout const& layout)
{
  if (source.size() != 1)
  {
    throw std::invalid_argument("two-site vector assignment requires a single-block source MatrixFamily");
  }
  auto const source_block = source.block(0);
  if (source_block.rows != layout.left_bond_dim * layout.left_physical_dim ||
      source_block.cols != layout.right_physical_dim * layout.right_bond_dim)
  {
    throw std::invalid_argument("two-site source matrix shape does not match the vector layout");
  }
  if (target.size() != layout.block_count())
  {
    throw std::invalid_argument("two-site target vector block count does not match the vector layout");
  }
  auto const source_values = source.values(0);
  for (std::size_t left_phys = 0; left_phys < layout.left_physical_dim; ++left_phys)
  {
    for (std::size_t right_phys = 0; right_phys < layout.right_physical_dim; ++right_phys)
    {
      auto const block_index = layout.block_index(left_phys, right_phys);
      if (target.block(block_index) !=
          tensorcontraction::MatrixFamily::Block{layout.left_bond_dim, layout.right_bond_dim})
      {
        throw std::invalid_argument("two-site target vector block shape does not match the vector layout");
      }
      auto target_values = target.values(block_index);
      for (std::size_t left_bond = 0; left_bond < layout.left_bond_dim; ++left_bond)
      {
        auto const source_row = left_bond * layout.left_physical_dim + left_phys;
        for (std::size_t right_bond = 0; right_bond < layout.right_bond_dim; ++right_bond)
        {
          auto const source_col = right_phys * layout.right_bond_dim + right_bond;
          target_values[left_bond * layout.right_bond_dim + right_bond] =
              source_values[source_row * source_block.cols + source_col];
        }
      }
    }
  }
}

inline auto make_two_site_vector(TwoSiteWavefunction const& theta,
                                 TwoSiteEffectiveHamiltonianLayout const& layout) -> tensorcontraction::MatrixFamily
{
  std::vector<tensorcontraction::MatrixFamily::Block> blocks(layout.block_count());
  std::fill(blocks.begin(), blocks.end(),
            tensorcontraction::MatrixFamily::Block{layout.left_bond_dim, layout.right_bond_dim});
  tensorcontraction::MatrixFamily result(blocks);
  assign_two_site_matrix_to_vector(theta.matrix_family(), result, layout);
  return result;
}

} // namespace uni20
