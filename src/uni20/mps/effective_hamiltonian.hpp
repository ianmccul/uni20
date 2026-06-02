/**
 * \file effective_hamiltonian.hpp
 * \brief Compile dense two-site effective Hamiltonians for the first DMRG prototype.
 */

#pragma once

#include <uni20/mps/environment.hpp>
#include <uni20/tensorcontraction/effective_hamiltonian_operator.hpp>
#include <uni20/tensorcontraction/vector_algebra.hpp>

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

  tensorcontraction::MatrixFamily a_mats(
      detail::repeated_blocks(left_env.virtual_dim(), left_env.bond_dim(), left_env.bond_dim()));
  tensorcontraction::MatrixFamily c_mats(
      detail::repeated_blocks(right_env.virtual_dim(), right_env.bond_dim(), right_env.bond_dim()));
  std::vector<tensorcontraction::EffectiveHamiltonianOperator::Term> terms;

  for (std::size_t mpo_left = 0; mpo_left < left_env.virtual_dim(); ++mpo_left)
  {
    a_mats.assign(mpo_left, left_env.values(mpo_left));
  }
  for (std::size_t mpo_right = 0; mpo_right < right_env.virtual_dim(); ++mpo_right)
  {
    auto const source = right_env.values(mpo_right);
    auto target = c_mats.values(mpo_right);
    for (std::size_t right_bra = 0; right_bra < right_env.bond_dim(); ++right_bra)
    {
      for (std::size_t right_ket = 0; right_ket < right_env.bond_dim(); ++right_ket)
      {
        target[right_ket * right_env.bond_dim() + right_bra] = source[right_bra * right_env.bond_dim() + right_ket];
      }
    }
  }

  for (std::size_t mpo_left = 0; mpo_left < left_component.rows(); ++mpo_left)
  {
    for (auto const& left_mpo_entry : left_component.data().row(mpo_left))
    {
      auto const mpo_middle = left_mpo_entry.column;
      auto const& left_op = left_mpo_entry.value;
      for (auto const& right_mpo_entry : right_component.data().row(mpo_middle))
      {
        auto const mpo_right = right_mpo_entry.column;
        auto const& right_op = right_mpo_entry.value;

        for (std::size_t left_phys_bra = 0; left_phys_bra < left_op.rows(); ++left_phys_bra)
        {
          for (auto const& left_op_entry : left_op.coefficients().row(left_phys_bra))
          {
            auto const left_phys_ket = left_op_entry.column;

            for (std::size_t right_phys_bra = 0; right_phys_bra < right_op.rows(); ++right_phys_bra)
            {
              for (auto const& right_op_entry : right_op.coefficients().row(right_phys_bra))
              {
                auto const right_phys_ket = right_op_entry.column;
                terms.push_back(tensorcontraction::EffectiveHamiltonianOperator::Term{
                    .r = output_layout.block_index(left_phys_bra, right_phys_bra),
                    .a = mpo_left,
                    .b = input_layout.block_index(left_phys_ket, right_phys_ket),
                    .c = mpo_right,
                    .coefficient = left_op_entry.value * right_op_entry.value});
              }
            }
          }
        }
      }
    }
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

inline auto make_two_site_vector(FiniteMPS const& psi, std::size_t left_site,
                                 TwoSiteEffectiveHamiltonianLayout const& layout) -> tensorcontraction::MatrixFamily
{
  if (left_site + 1 >= psi.size())
  {
    throw std::out_of_range("make_two_site_vector requires two adjacent MPS sites");
  }

  auto const& left = psi[left_site];
  auto const& right = psi[left_site + 1];
  if (left.right_bond_space() != right.left_bond_space())
  {
    throw std::invalid_argument("make_two_site_vector adjacent bond spaces do not match");
  }
  if (layout.left_bond_dim != left.left_dim() || layout.left_physical_dim != left.physical_dim() ||
      layout.right_physical_dim != right.physical_dim() || layout.right_bond_dim != right.right_dim())
  {
    throw std::invalid_argument("make_two_site_vector layout does not match the MPS sites");
  }

  auto const shared_bond_dim = left.right_dim();
  std::vector<tensorcontraction::MatrixFamily::Block> left_blocks(layout.left_physical_dim);
  std::vector<tensorcontraction::MatrixFamily::Block> right_blocks(layout.right_physical_dim);
  std::vector<tensorcontraction::MatrixFamily::Block> result_blocks(layout.block_count());
  std::vector<std::size_t> left_block_for_result(layout.block_count());
  std::vector<std::size_t> right_block_for_result(layout.block_count());
  std::fill(left_blocks.begin(), left_blocks.end(),
            tensorcontraction::MatrixFamily::Block{layout.left_bond_dim, shared_bond_dim});
  std::fill(right_blocks.begin(), right_blocks.end(),
            tensorcontraction::MatrixFamily::Block{shared_bond_dim, layout.right_bond_dim});
  for (std::size_t left_phys = 0; left_phys < layout.left_physical_dim; ++left_phys)
  {
    for (std::size_t right_phys = 0; right_phys < layout.right_physical_dim; ++right_phys)
    {
      auto const block = layout.block_index(left_phys, right_phys);
      result_blocks[block] = tensorcontraction::MatrixFamily::Block{layout.left_bond_dim, layout.right_bond_dim};
      left_block_for_result[block] = left_phys;
      right_block_for_result[block] = right_phys;
    }
  }

  tensorcontraction::MatrixFamily left_operands(left_blocks);
  tensorcontraction::MatrixFamily right_operands(right_blocks);
  tensorcontraction::MatrixFamily result(result_blocks);
  for (std::size_t left_phys = 0; left_phys < layout.left_physical_dim; ++left_phys)
  {
    left_operands.assign(left_phys, left.values(left_phys));
  }
  for (std::size_t right_phys = 0; right_phys < layout.right_physical_dim; ++right_phys)
  {
    right_operands.assign(right_phys, right.values(right_phys));
  }

  // The first dense prototype has one block per physical-pair sector.  Future
  // symmetry-aware code should replace these block GEMMs with the same
  // operation over only the allowed fusion/F-move output sectors.
  tensorcontraction::gemm_selected(left_operands, right_operands, result, left_block_for_result,
                                   right_block_for_result);
  return result;
}

inline auto
make_two_site_vector_resident(FiniteMPS const& psi, std::size_t left_site,
                              TwoSiteEffectiveHamiltonianLayout const& layout,
                              tensorcontraction::VectorAlgebraEngine& algebra) -> tensorcontraction::MatrixFamily
{
  if (left_site + 1 >= psi.size())
  {
    throw std::out_of_range("make_two_site_vector requires two adjacent MPS sites");
  }

  auto const& left = psi[left_site];
  auto const& right = psi[left_site + 1];
  if (left.right_bond_space() != right.left_bond_space())
  {
    throw std::invalid_argument("make_two_site_vector adjacent bond spaces do not match");
  }
  if (layout.left_bond_dim != left.left_dim() || layout.left_physical_dim != left.physical_dim() ||
      layout.right_physical_dim != right.physical_dim() || layout.right_bond_dim != right.right_dim())
  {
    throw std::invalid_argument("make_two_site_vector layout does not match the MPS sites");
  }

  auto const shared_bond_dim = left.right_dim();
  std::vector<tensorcontraction::MatrixFamily::Block> left_blocks(layout.left_physical_dim);
  std::vector<tensorcontraction::MatrixFamily::Block> right_blocks(layout.right_physical_dim);
  std::vector<tensorcontraction::MatrixFamily::Block> result_blocks(layout.block_count());
  std::vector<std::size_t> left_block_for_result(layout.block_count());
  std::vector<std::size_t> right_block_for_result(layout.block_count());
  std::fill(left_blocks.begin(), left_blocks.end(),
            tensorcontraction::MatrixFamily::Block{layout.left_bond_dim, shared_bond_dim});
  std::fill(right_blocks.begin(), right_blocks.end(),
            tensorcontraction::MatrixFamily::Block{shared_bond_dim, layout.right_bond_dim});
  for (std::size_t left_phys = 0; left_phys < layout.left_physical_dim; ++left_phys)
  {
    for (std::size_t right_phys = 0; right_phys < layout.right_physical_dim; ++right_phys)
    {
      auto const block = layout.block_index(left_phys, right_phys);
      result_blocks[block] = tensorcontraction::MatrixFamily::Block{layout.left_bond_dim, layout.right_bond_dim};
      left_block_for_result[block] = left_phys;
      right_block_for_result[block] = right_phys;
    }
  }

  tensorcontraction::MatrixFamily left_operands(left_blocks);
  tensorcontraction::MatrixFamily right_operands(right_blocks);
  tensorcontraction::MatrixFamily result(result_blocks);
  for (std::size_t left_phys = 0; left_phys < layout.left_physical_dim; ++left_phys)
  {
    left_operands.assign(left_phys, left.values(left_phys));
  }
  for (std::size_t right_phys = 0; right_phys < layout.right_physical_dim; ++right_phys)
  {
    right_operands.assign(right_phys, right.values(right_phys));
  }

  // Host MPS storage is still the authority for the site tensors, but the
  // resulting center is consumed immediately by Lanczos.  Keep that output in
  // the same TensorContraction runtime instead of materializing it to host and
  // uploading it again.
  algebra.gemm_selected_to_resident(left_operands, right_operands, result, left_block_for_result,
                                    right_block_for_result);
  return result;
}

} // namespace uni20
