/**
 * \file two_site_dmrg.hpp
 * \ingroup tensor_network
 * \brief Performs directional finite-chain two-site DMRG updates.
 */

#pragma once

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/krylov/block_tensor_vector.hpp>
#include <uni20/krylov/symmetric_lanczos.hpp>
#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/tensor_network/environment_cache.hpp>
#include <uni20/tensor_network/two_site_effective_hamiltonian.hpp>
#include <uni20/tensor_network/two_site_split.hpp>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::tensor_network
{

/// \brief Numerical policies for one finite-chain two-site DMRG update.
/// \details The first implementation targets one smallest-algebraic Ritz
///          vector and requires that local solve to converge before changing
///          the MPS. The default truncation retains at least one singular state.
/// \tparam Real Real component type of the MPS scalar.
template <uni20::Real Real> struct TwoSiteDmrgOptions
{
    /// \brief Native Hermitian Lanczos parameters for each local solve.
    krylov::SymmetricEigenParams<Real> eigensolver = {
        .eigenvalue_count = 1,
        .spectrum = krylov::SpectrumPart::SmallestAlgebraic,
        .compute_eigenvectors = true,
    };
    /// \brief Global block-SVD state-selection policy for each updated bond.
    linalg::SvdTruncationPolicy<Real> truncation = {
        .minimum_retained_extent = 1,
    };
};

/// \brief Diagnostics and selected bond data from one DMRG bond update.
/// \tparam Real Real component type of the MPS scalar.
/// \tparam InstalledBond Diagonal spectrum and truncation result type.
template <uni20::Real Real, class InstalledBond> struct TwoSiteDmrgStepResult
{
    /// \brief Index of the left site in the optimized pair.
    std::size_t first_site;
    /// \brief Direction in which the canonical center moved.
    MpsSweepDirection direction;
    /// \brief Lowest local Ritz value before SVD truncation.
    Real local_energy;
    /// \brief Residual bound reported for the installed Ritz vector.
    Real residual_bound;
    /// \brief Number of Lanczos iterations or restart cycles.
    int iteration_count;
    /// \brief Number of effective-Hamiltonian applications.
    int matvec_count;
    /// \brief Selected Schmidt spectrum and truncation diagnostics.
    InstalledBond bond;
};

namespace detail
{

inline void validate_two_site_dmrg_direction(MpsSweepDirection direction)
{
  if (direction != MpsSweepDirection::left_to_right && direction != MpsSweepDirection::right_to_left)
    throw std::invalid_argument("two-site DMRG received an invalid sweep direction");
}

template <uni20::Real Real> void validate_two_site_dmrg_options(TwoSiteDmrgOptions<Real> const& options)
{
  if (options.eigensolver.eigenvalue_count != 1 ||
      options.eigensolver.spectrum != krylov::SpectrumPart::SmallestAlgebraic ||
      !options.eigensolver.compute_eigenvectors)
  {
    throw std::invalid_argument("two-site DMRG requires one computed smallest-algebraic local eigenvector");
  }
}

template <class Center, class EffectiveHamiltonian, uni20::Real Real>
auto solve_two_site_ground_state(Center const& initial, EffectiveHamiltonian effective_hamiltonian,
                                 krylov::SymmetricEigenParams<Real> const& params)
{
  using scalar_type = block_tensor_value_t<Center>;
  krylov::BlockTensorMatrixFreeOps ops(initial, std::move(effective_hamiltonian));
  std::size_t const dimension = ops.problem_dimension();
  int const krylov_dimension = krylov::effective_symmetric_krylov_dimension(params, dimension);

  auto result = static_cast<std::size_t>(krylov_dimension) < dimension && krylov_dimension > params.eigenvalue_count
                    ? krylov::symmetric_lanczos_restarted_standard<scalar_type>(ops, initial, params)
                    : krylov::symmetric_lanczos_standard<scalar_type>(ops, initial, params);
  if (result.status != 0 || result.eigenvalues.size() != 1 || result.residual_bounds.size() != 1 ||
      result.eigenvectors.size() != 1)
  {
    throw std::runtime_error("two-site DMRG local eigensolver did not converge to one ground-state vector");
  }
  return result;
}

} // namespace detail

/// \brief Optimize and replace one adjacent MPS pair with a ground-state DMRG step.
/// \details The operation forms the current two-site center, obtains the two
///          reusable environments outside the pair, solves the fixed sparse
///          effective Hamiltonian, performs staged block-SVD truncation, and
///          installs the directional split. The existing internal-bond label
///          is preserved. After replacement, only the completed-side
///          environment needed by the next step is rebuilt.
/// \tparam MpsChain Concrete finite MPS owner type.
/// \tparam MpoChain Concrete finite MPO owner type.
/// \tparam EnvironmentStorage Immediate host storage used by the cache.
/// \param mps Mutable MPS whose selected pair is replaced after convergence.
/// \param mpo Immutable MPO defining the effective Hamiltonian.
/// \param cache Environment cache attached to exactly \p mps and \p mpo.
/// \param first_site Index of the left site in the active pair.
/// \param direction Direction in which the canonical center moves.
/// \param options Local eigensolver and global SVD-truncation policies.
/// \return Local energy, residual and work diagnostics, and selected bond data.
/// \throws std::invalid_argument If the pair, cache, direction, or options are invalid.
/// \throws std::runtime_error If the local eigensolver does not converge.
template <class MpsChain, class MpoChain, SparseBlockStorage EnvironmentStorage>
[[nodiscard]] auto
optimize_two_site_dmrg_bond(MpsChain& mps, MpoChain const& mpo,
                            MpoEnvironmentCache<MpsChain, MpoChain, EnvironmentStorage>& cache, std::size_t first_site,
                            MpsSweepDirection direction,
                            TwoSiteDmrgOptions<make_real_t<typename MpsChain::value_type>> const& options = {})
{
  using scalar_type = typename MpsChain::value_type;
  using real_type = make_real_t<scalar_type>;
  static_assert(uni20::LapackRealOrComplex<scalar_type>,
                "two-site DMRG requires a real or complex scalar with LAPACK support");

  detail::validate_two_site_dmrg_direction(direction);
  detail::validate_two_site_dmrg_options(options);
  if (!cache.is_attached_to(mps, mpo))
    throw std::invalid_argument("two-site DMRG cache is attached to different chain objects");
  if (first_site >= mps.size() || first_site + 1 >= mps.size())
    throw std::out_of_range("two-site DMRG pair index is out of range");

  auto current = contract_adjacent<1>(mps.site(first_site), mps.site(first_site + 1));
  using center_type = std::remove_cvref_t<decltype(current)>;
  center_type initial(current.symmetry(), current.domain(), current.codomain(), current.legal_block_keys());
  copy(initial, current);
  auto eigensystem = [&]() {
    auto const& left_environment = cache.left_environment(first_site);
    auto const& right_environment = cache.right_environment(first_site + 2);
    TwoSiteEffectiveHamiltonian effective_hamiltonian(initial, left_environment, mpo.site(first_site),
                                                      mpo.site(first_site + 1), right_environment);
    return detail::solve_two_site_ground_state(initial, std::move(effective_hamiltonian), options.eigensolver);
  }();

  real_type const local_energy = eigensystem.eigenvalues.front();
  real_type const residual_bound = eigensystem.residual_bounds.front();
  int const iteration_count = eigensystem.iteration_count;
  int const matvec_count = eigensystem.matvec_count;
  auto decomposition = decompose_two_site_center(eigensystem.eigenvectors.front());
  auto selection = select_svd_states(decomposition.spectrum(), options.truncation);
  auto installed =
      replace_two_site_from_svd(mps, first_site, decomposition, selection, direction,
                                {.bond_label = mps.site(first_site).codomain().template space<0>().label()});

  if (direction == MpsSweepDirection::left_to_right)
    static_cast<void>(cache.left_environment(first_site + 1));
  else
    static_cast<void>(cache.right_environment(first_site + 1));

  return TwoSiteDmrgStepResult<real_type, decltype(installed)>{.first_site = first_site,
                                                               .direction = direction,
                                                               .local_energy = local_energy,
                                                               .residual_bound = residual_bound,
                                                               .iteration_count = iteration_count,
                                                               .matvec_count = matvec_count,
                                                               .bond = std::move(installed)};
}

/// \brief Traverse every adjacent bond once in one two-site DMRG direction.
/// \details A left-to-right sweep visits `0 ... L-2`; a right-to-left sweep
///          visits `L-2 ... 0`. Each step leaves the environment on its
///          completed side ready for the next pair.
/// \param mps Mutable finite MPS.
/// \param mpo Immutable finite MPO.
/// \param cache Environment cache attached to both chain objects.
/// \param direction Traversal and singular-value absorption direction.
/// \param options Shared local-solve and truncation policies for the sweep.
/// \return One result per bond in visitation order.
/// \throws std::invalid_argument If the chain has fewer than two sites or the direction is invalid.
template <class MpsChain, class MpoChain, SparseBlockStorage EnvironmentStorage>
[[nodiscard]] auto
sweep_two_site_dmrg(MpsChain& mps, MpoChain const& mpo,
                    MpoEnvironmentCache<MpsChain, MpoChain, EnvironmentStorage>& cache, MpsSweepDirection direction,
                    TwoSiteDmrgOptions<make_real_t<typename MpsChain::value_type>> const& options = {})
{
  detail::validate_two_site_dmrg_direction(direction);
  if (mps.size() < 2) throw std::invalid_argument("two-site DMRG sweep requires at least two sites");

  using result_type = decltype(optimize_two_site_dmrg_bond(mps, mpo, cache, std::size_t{}, direction, options));
  std::vector<result_type> result;
  result.reserve(mps.size() - 1);
  if (direction == MpsSweepDirection::left_to_right)
  {
    for (std::size_t first_site = 0; first_site + 1 < mps.size(); ++first_site)
      result.push_back(optimize_two_site_dmrg_bond(mps, mpo, cache, first_site, direction, options));
  }
  else
  {
    for (std::size_t second_site = mps.size() - 1; second_site > 0; --second_site)
      result.push_back(optimize_two_site_dmrg_bond(mps, mpo, cache, second_site - 1, direction, options));
  }
  return result;
}

} // namespace uni20::tensor_network
