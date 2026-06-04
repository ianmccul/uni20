/**
 * \file two_site_solve.hpp
 * \brief Local two-site DMRG solve wrapper for the first prototype.
 */

#pragma once

#include <uni20/mps/effective_hamiltonian.hpp>
#include <uni20/tensorcontraction/lanczos.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>

namespace uni20
{

struct TwoSiteSolveResult
{
    tensorcontraction::LanczosResult lanczos;
    TwoSiteEffectiveHamiltonianLayout layout;
    tensorcontraction::MatrixFamily optimized_vector;
    std::unique_ptr<tensorcontraction::VectorAlgebraEngine> resident_algebra;
    tensorcontraction::MatrixFamily optimized_matrix;
};

inline void assign_two_site_vector_to_matrix(tensorcontraction::MatrixFamily const& source,
                                             tensorcontraction::MatrixFamily& target,
                                             TwoSiteEffectiveHamiltonianLayout const& layout)
{
  if (target.size() != 1)
  {
    throw std::invalid_argument("two-site matrix assignment requires a single-block target MatrixFamily");
  }
  if (source.size() != layout.block_count())
  {
    throw std::invalid_argument("two-site source vector block count does not match the layout");
  }
  if (target.block(0) != tensorcontraction::MatrixFamily::Block{layout.left_bond_dim * layout.left_physical_dim,
                                                                layout.right_physical_dim * layout.right_bond_dim})
  {
    throw std::invalid_argument("two-site target matrix shape does not match the layout");
  }
  auto target_values = target.values(0);
  auto const target_cols = layout.right_physical_dim * layout.right_bond_dim;
  for (std::size_t left_phys = 0; left_phys < layout.left_physical_dim; ++left_phys)
  {
    for (std::size_t right_phys = 0; right_phys < layout.right_physical_dim; ++right_phys)
    {
      auto const block_index = layout.block_index(left_phys, right_phys);
      if (source.block(block_index) !=
          tensorcontraction::MatrixFamily::Block{layout.left_bond_dim, layout.right_bond_dim})
      {
        throw std::invalid_argument("two-site source vector block shape does not match the layout");
      }
      auto const source_values = source.values(block_index);
      for (std::size_t left_bond = 0; left_bond < layout.left_bond_dim; ++left_bond)
      {
        auto const target_row = left_bond * layout.left_physical_dim + left_phys;
        for (std::size_t right_bond = 0; right_bond < layout.right_bond_dim; ++right_bond)
        {
          auto const target_col = right_phys * layout.right_bond_dim + right_bond;
          target_values[target_row * target_cols + target_col] =
              source_values[left_bond * layout.right_bond_dim + right_bond];
        }
      }
    }
  }
}

inline auto make_two_site_matrix(tensorcontraction::MatrixFamily const& vector,
                                 TwoSiteEffectiveHamiltonianLayout const& layout) -> tensorcontraction::MatrixFamily
{
  std::array block{tensorcontraction::MatrixFamily::Block{layout.left_bond_dim * layout.left_physical_dim,
                                                          layout.right_physical_dim * layout.right_bond_dim}};
  tensorcontraction::MatrixFamily result(block);
  assign_two_site_vector_to_matrix(vector, result, layout);
  return result;
}

inline auto solve_two_site(FiniteMPS const& psi, FiniteTriangularMPO const& mpo, std::size_t left_site,
                           MpoEnvironment const& left_env, MpoEnvironment const& right_env,
                           tensorcontraction::LanczosOptions options = {}) -> TwoSiteSolveResult
{
  if (left_site + 1 >= psi.size() || left_site + 1 >= mpo.size())
  {
    throw std::out_of_range("solve_two_site requires two adjacent MPS and MPO sites");
  }

  auto effective_hamiltonian =
      make_two_site_effective_hamiltonian(left_env, mpo[left_site], mpo[left_site + 1], right_env);
  auto algebra = std::make_unique<tensorcontraction::VectorAlgebraEngine>();
  auto optimized_vector = make_two_site_vector_resident(psi, left_site, effective_hamiltonian.input_layout, *algebra);
  options.synchronize_result_to_host = false;
  auto lanczos =
      tensorcontraction::lanczos_lowest_resident(optimized_vector, effective_hamiltonian.op, *algebra, options);

  return TwoSiteSolveResult{.lanczos = lanczos,
                            .layout = effective_hamiltonian.output_layout,
                            .optimized_vector = std::move(optimized_vector),
                            .resident_algebra = std::move(algebra),
                            .optimized_matrix = tensorcontraction::MatrixFamily{}};
}

inline auto materialize_two_site_solution_matrix(TwoSiteSolveResult& solution) -> tensorcontraction::MatrixFamily&
{
  if (solution.optimized_matrix.empty())
  {
    if (solution.resident_algebra != nullptr && !solution.resident_algebra->uses_host_backend())
    {
      solution.resident_algebra->synchronize(solution.optimized_vector);
    }
    // The uni20 prototype currently uses MPI ranks as replicated host-state
    // workers rather than TensorContraction's original distributed R-owner model.
    // Rank-local contraction matvecs can drift at roundoff level, so treat rank 0
    // as the authority when materializing host state.
    tensorcontraction::broadcast_values_from_rank_zero(solution.optimized_vector);
    solution.optimized_matrix = make_two_site_matrix(solution.optimized_vector, solution.layout);
  }
  return solution.optimized_matrix;
}

} // namespace uni20
