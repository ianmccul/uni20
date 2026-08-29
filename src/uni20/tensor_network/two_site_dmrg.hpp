/**
 * \file two_site_dmrg.hpp
 * \ingroup tensor_network
 * \brief Performs directional finite-chain two-site DMRG updates.
 */

#pragma once

#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/performance_measurements.hpp>
#include <uni20/core/math.hpp>
#include <uni20/core/numeric_limits.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/krylov/block_tensor_vector.hpp>
#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/tensor_network/dmrg_lanczos.hpp>
#include <uni20/tensor_network/environment_cache.hpp>
#include <uni20/tensor_network/two_site_effective_hamiltonian.hpp>
#include <uni20/tensor_network/two_site_split.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::tensor_network
{

/// \brief Numerical policies for one finite-chain two-site DMRG update.
/// \details The local solve performs a fixed amount of work and installs the
///          resulting smallest Ritz vector without requiring local convergence.
///          The default truncation retains at least one singular state.
/// \tparam Real Real component type of the MPS scalar.
template <uni20::Real Real> struct TwoSiteDmrgOptions
{
    /// \brief Fixed-work Lanczos policy for each local solve.
    DmrgLanczosOptions<Real> local_solver = {};
    /// \brief Global block-SVD state-selection policy for each updated bond.
    linalg::SvdTruncationPolicy<Real> truncation = {
        .minimum_retained_extent = 1,
    };
};

/// \brief Policies for an alternating finite-chain two-site DMRG run.
/// \details One sweep is one complete directional traversal. A zero energy
///          tolerance selects `100 * numeric_limits<Real>::epsilon()`.
/// \tparam Real Real component type of the MPS scalar.
template <uni20::Real Real> struct TwoSiteDmrgRunOptions
{
    /// \brief Local solve and bond-truncation policy shared by every update.
    TwoSiteDmrgOptions<Real> bond_options = {};
    /// \brief Maximum number of alternating directional traversals.
    std::size_t maximum_sweeps = 10;
    /// \brief Relative/absolute hybrid tolerance for consecutive terminal energies.
    Real energy_tolerance = Real{};
    /// \brief Direction of the first traversal.
    MpsSweepDirection initial_direction = MpsSweepDirection::left_to_right;
};

/// \brief Stable phase identifiers for two-site DMRG performance measurements.
enum class TwoSiteDmrgPerformanceEvent
{
  run,
  sweep,
  bond_update,
  center_construction,
  local_eigensolver,
  effective_hamiltonian_application,
  krylov_vector_allocation,
  krylov_vector_update,
  krylov_reduction,
  block_svd,
  svd_sector_batch,
  state_selection,
  factor_materialization,
  environment_update,
  count,
};

/// \brief Coarse inclusive wall-duration measurements for two-site DMRG phases.
using TwoSiteDmrgPerformanceMeasurements =
    performance::DurationMeasurements<TwoSiteDmrgPerformanceEvent,
                                      static_cast<std::size_t>(TwoSiteDmrgPerformanceEvent::count)>;

/// \brief Detailed DMRG measurements including per-sector block-SVD batch timings.
using DetailedTwoSiteDmrgPerformanceMeasurements =
    performance::DetailedMeasurements<TwoSiteDmrgPerformanceEvent,
                                      static_cast<std::size_t>(TwoSiteDmrgPerformanceEvent::count)>;

/// \brief Return the stable display name of one DMRG performance event.
/// \pre `event` is not `TwoSiteDmrgPerformanceEvent::count`.
/// \param event Event identifier to name.
/// \return Static event name suitable for reports and table labels.
[[nodiscard]] constexpr auto
two_site_dmrg_performance_event_name(TwoSiteDmrgPerformanceEvent event) noexcept -> std::string_view
{
  using enum TwoSiteDmrgPerformanceEvent;
  switch (event)
  {
    case run:
      return "run";
    case sweep:
      return "sweep";
    case bond_update:
      return "bond update";
    case center_construction:
      return "center construction";
    case local_eigensolver:
      return "local eigensolver";
    case effective_hamiltonian_application:
      return "effective Hamiltonian application";
    case krylov_vector_allocation:
      return "Krylov vector allocation";
    case krylov_vector_update:
      return "Krylov vector update";
    case krylov_reduction:
      return "Krylov reduction";
    case block_svd:
      return "block SVD";
    case svd_sector_batch:
      return "SVD sector batch";
    case state_selection:
      return "state selection";
    case factor_materialization:
      return "factor materialization";
    case environment_update:
      return "environment update";
    case count:
      break;
  }
  std::unreachable();
}

/// \brief Aggregate diagnostics for one completed directional DMRG sweep.
/// \tparam Real Real component type of the MPS scalar.
template <uni20::Real Real> struct TwoSiteDmrgSweepSummary
{
    /// \brief Zero-based directional sweep index.
    std::size_t sweep_index;
    /// \brief Direction visited by this sweep.
    MpsSweepDirection direction;
    /// \brief Local Ritz value from the terminal bond before its SVD.
    Real terminal_local_energy;
    /// \brief Terminal-bond normalized discarded squared norm.
    Real terminal_discarded_weight;
    /// \brief Whether the terminal local energy is also the installed global-state energy.
    bool terminal_energy_is_global;
    /// \brief Absolute change from the preceding valid terminal energy.
    std::optional<Real> energy_change;
    /// \brief Largest normalized discarded weight over the sweep.
    Real maximum_discarded_weight;
    /// \brief Largest retained internal-bond dimension over the sweep.
    std::size_t maximum_bond_dimension;
    /// \brief Largest local Ritz residual bound over the sweep.
    Real maximum_residual_bound;
    /// \brief Sum of local Lanczos projection steps.
    int total_iteration_count;
    /// \brief Sum of effective-Hamiltonian applications.
    int total_matvec_count;
    /// \brief Whether this sweep satisfied the run convergence test.
    bool converged;
};

/// \brief Result of an alternating finite-chain two-site DMRG run.
/// \tparam Real Real component type of the MPS scalar.
template <uni20::Real Real> struct TwoSiteDmrgRunResult
{
    /// \brief Per-sweep summaries in execution order.
    std::vector<TwoSiteDmrgSweepSummary<Real>> sweeps;
    /// \brief Whether the final executed sweep satisfied the energy criterion.
    bool converged;
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
    /// \brief Number of Lanczos projection steps.
    int iteration_count;
    /// \brief Number of effective-Hamiltonian applications.
    int matvec_count;
    /// \brief Selected Schmidt spectrum and truncation diagnostics.
    InstalledBond bond;
};

namespace detail
{

/// \brief Local non-diagonal storage suitable for two-site center blocks.
/// \details This structural requirement does not imply that center construction,
///          environment placement, factorization, or another complete DMRG step
///          is implemented for every conforming leaf memory domain.
/// \tparam Storage Candidate BlockTensor storage policy.
/// \tparam Scalar Numerical center element type.
template <class Storage, class Scalar>
concept TwoSiteCenterStorageFor =
    LocalBlockStorageFor<Storage, std::remove_cv_t<Scalar>, 4, 2> && !DiagonalBlockStorage<Storage> &&
    !AsyncLocalBlockStorageFor<Storage, std::remove_cv_t<Scalar>, 4, 2>;

template <class Function>
void execute_materialization_batch(SerialBlockExecution, std::size_t size, Function&& function)
{
  for (std::size_t index = 0; index < size; ++index)
    function(index);
}

template <class Function>
void execute_materialization_batch(SchedulerBatchBlockExecution, std::size_t size, Function&& function)
{
  async::execute_batch(size, std::forward<Function>(function));
}

/// \brief Materialize a local BlockTensor view into a selected storage policy.
/// \details Each stored block is copied through tensor-level dispatch so a
///          differing leaf memory domain performs an explicit domain transfer.
///          Missing blocks are initialized to zero when the output storage
///          contains a wider legal block set than the input.
/// \tparam OutputStorage Destination BlockTensor storage policy.
/// \tparam StoreAllLegalBlocks Whether sparse output stores every legal key.
/// \param input Source tensor or mapped tensor view.
/// \return Owning BlockTensor in `OutputStorage`.
template <BlockTensorStorage OutputStorage, bool StoreAllLegalBlocks, BlockTensorView Input>
  requires LocalBlockStorageFor<OutputStorage, block_tensor_value_t<Input>,
                                block_tensor_type_t<Input>::key_coordinate_count(),
                                block_tensor_type_t<Input>::dense_block_order()>
[[nodiscard]] auto materialize_local_block_tensor(Input const& input)
{
  using output_type = BlockTensor<block_tensor_value_t<Input>, block_tensor_domain_t<Input>,
                                  block_tensor_codomain_t<Input>, OutputStorage>;
  auto output = [&] {
    if constexpr (CompleteBlockStorage<OutputStorage>)
    {
      return output_type(input.symmetry(), input.domain(), input.codomain());
    }
    else
    {
      using key_type = typename output_type::key_type;
      auto keys = [&] {
        if constexpr (StoreAllLegalBlocks)
          return input.legal_block_keys();
        else
          return std::vector<key_type>(input.stored_keys().begin(), input.stored_keys().end());
      }();
      return output_type(input.symmetry(), input.domain(), input.codomain(), std::move(keys));
    }
  }();
  using value_type = block_tensor_value_t<Input>;
  execute_materialization_batch(
      typename OutputStorage::block_execution_policy{}, output.stored_block_count(), [&](std::size_t output_ordinal) {
        auto const& key = output.stored_keys()[output_ordinal];
        auto const found = std::ranges::lower_bound(input.stored_keys(), key);
        auto output_block = output.block_by_ordinal(output_ordinal);
        if (found == input.stored_keys().end() || *found != key)
        {
          uni20::fill(output_block, value_type{});
          return;
        }
        auto const input_ordinal = static_cast<std::size_t>(found - input.stored_keys().begin());
        auto input_block = input.block_by_ordinal(input_ordinal);
        uni20::copy(output_block, input_block);
      });
  return output;
}

/// \brief Construct a two-site center in the requested local storage.
/// \details Contraction executes directly in the selected leaf memory domain.
///          A sparse selected policy is subsequently widened to the complete
///          legal center key set required by Krylov iteration.
template <BlockTensorStorage CenterStorage, BlockTensorView Left, BlockTensorView Right>
[[nodiscard]] auto make_two_site_center(Left const& left, Right const& right)
{
  if constexpr (CompleteBlockStorage<CenterStorage>)
  {
    return contract_adjacent<1, CenterStorage>(left, right);
  }
  else
  {
    auto contracted = contract_adjacent<1, CenterStorage>(left, right);
    return materialize_local_block_tensor<CenterStorage, true>(contracted);
  }
}

/// \brief Sparse environment storage matched to a two-site center memory domain.
/// \details Environment block presence remains sparse, while block execution
///          follows the center's serial or scheduler-batch policy.
template <class CenterStorage>
using DmrgEnvironmentStorage =
    std::conditional_t<std::same_as<typename CenterStorage::block_execution_policy, SchedulerBatchBlockExecution>,
                       ParallelPackedSparseBlockStorage<typename CenterStorage::leaf_storage_policy>,
                       PackedSparseBlockStorage<typename CenterStorage::leaf_storage_policy>>;

/// \brief Place an environment in the center's leaf memory domain.
/// \details Matching domains return a borrowed zero-copy view. Cross-domain
///          placement returns an owning BlockTensor; callers must retain that
///          object for every descriptor borrowed by asynchronous execution.
/// \param environment Cached environment to place.
/// \return Borrowed same-domain view or owning cross-domain materialization.
template <class CenterStorage, BlockTensorView Environment>
[[nodiscard]] auto place_environment_for_center(Environment const& environment)
{
  using environment_storage = typename block_tensor_type_t<Environment>::storage_policy;
  if constexpr (std::same_as<typename CenterStorage::leaf_storage_policy,
                             typename environment_storage::leaf_storage_policy>)
  {
    return as_block_tensor_view(environment);
  }
  else
  {
    return materialize_local_block_tensor<DmrgEnvironmentStorage<CenterStorage>, false>(environment);
  }
}

inline void validate_two_site_dmrg_direction(MpsSweepDirection direction)
{
  if (direction != MpsSweepDirection::left_to_right && direction != MpsSweepDirection::right_to_left)
    throw std::invalid_argument("two-site DMRG received an invalid sweep direction");
}

template <uni20::Real Real> void validate_two_site_dmrg_options(TwoSiteDmrgOptions<Real> const& options)
{
  if (options.local_solver.matvec_iterations == 0)
    throw std::invalid_argument("two-site DMRG requires at least one local matvec iteration");
}

template <uni20::Real Real> void validate_two_site_dmrg_run_options(TwoSiteDmrgRunOptions<Real> const& options)
{
  validate_two_site_dmrg_direction(options.initial_direction);
  validate_two_site_dmrg_options(options.bond_options);
  if (options.maximum_sweeps == 0) throw std::invalid_argument("two-site DMRG run requires at least one sweep");
  if (!uni20::isfinite(options.energy_tolerance) || options.energy_tolerance < Real{})
    throw std::invalid_argument("two-site DMRG energy tolerance must be finite and nonnegative");
}

inline auto opposite_dmrg_direction(MpsSweepDirection direction) -> MpsSweepDirection
{
  return direction == MpsSweepDirection::left_to_right ? MpsSweepDirection::right_to_left
                                                       : MpsSweepDirection::left_to_right;
}

template <uni20::Real Real> auto dmrg_abs(Real value) -> Real
{
  using std::abs;
  return abs(value);
}

template <uni20::Real Real> auto terminal_energies_converged(Real previous, Real current, Real tolerance) -> bool
{
  Real const effective_tolerance = tolerance > Real{} ? tolerance : Real{100} * uni20::numeric_limits<Real>::epsilon();
  Real const scale = std::max({Real{1}, dmrg_abs(previous), dmrg_abs(current)});
  return dmrg_abs(current - previous) <= effective_tolerance * scale;
}

template <class Ops, class Measurements> class MeasuredDmrgKrylovOps {
  public:
    using ops_type = std::remove_cvref_t<Ops>;
    using tensor_type = typename ops_type::tensor_type;
    using scalar_type = typename ops_type::scalar_type;
    using real_type = typename ops_type::real_type;

    MeasuredDmrgKrylovOps(ops_type ops, Measurements& measurements) : ops_(std::move(ops)), measurements_(measurements)
    {}

    [[nodiscard]] auto problem_dimension() const noexcept -> std::size_t { return ops_.problem_dimension(); }

    [[nodiscard]] auto vector_dimension(tensor_type const& vector) const -> std::size_t
    {
      return ops_.vector_dimension(vector);
    }

    [[nodiscard]] auto allocate_like(tensor_type const& vector) const -> tensor_type
    {
      return this->measure_detailed(TwoSiteDmrgPerformanceEvent::krylov_vector_allocation,
                                    [&] { return ops_.allocate_like(vector); });
    }

    void copy(tensor_type& output, tensor_type const& input) const
    {
      this->measure_detailed(TwoSiteDmrgPerformanceEvent::krylov_vector_update, [&] { ops_.copy(output, input); });
    }

    void axpy(tensor_type& output, scalar_type factor, tensor_type const& input) const
    {
      this->measure_detailed(TwoSiteDmrgPerformanceEvent::krylov_vector_update,
                             [&] { ops_.axpy(output, factor, input); });
    }

    void scal(tensor_type& vector, scalar_type factor) const
    {
      this->measure_detailed(TwoSiteDmrgPerformanceEvent::krylov_vector_update, [&] { ops_.scal(vector, factor); });
    }

    void set_zero(tensor_type& vector) const
    {
      this->measure_detailed(TwoSiteDmrgPerformanceEvent::krylov_vector_update, [&] { ops_.set_zero(vector); });
    }

    [[nodiscard]] auto inner_product(tensor_type const& lhs, tensor_type const& rhs) const -> scalar_type
    {
      return this->measure_detailed(TwoSiteDmrgPerformanceEvent::krylov_reduction,
                                    [&] { return ops_.inner_product(lhs, rhs); });
    }

    [[nodiscard]] auto norm(tensor_type const& vector) const -> real_type
    {
      return this->measure_detailed(TwoSiteDmrgPerformanceEvent::krylov_reduction, [&] { return ops_.norm(vector); });
    }

    void matvec(tensor_type& output, tensor_type const& input)
    {
      performance::measure_duration(measurements_, TwoSiteDmrgPerformanceEvent::effective_hamiltonian_application,
                                    [&] { ops_.matvec(output, input); });
    }

  private:
    template <class Function>
    decltype(auto) measure_detailed(TwoSiteDmrgPerformanceEvent event, Function&& function) const
    {
      if constexpr (performance::measurement_level_v<Measurements> == performance::MeasurementLevel::detailed)
        return performance::measure_duration(measurements_, event, std::forward<Function>(function));
      else
        return std::invoke(std::forward<Function>(function));
    }

    ops_type ops_;
    Measurements& measurements_;
};

template <class Center, class EffectiveHamiltonian, uni20::Real Real, class Measurements>
  requires performance::DurationMeasurementPolicy<Measurements, TwoSiteDmrgPerformanceEvent>
auto solve_two_site_ground_state(Center const& initial, EffectiveHamiltonian effective_hamiltonian,
                                 DmrgLanczosOptions<Real> const& options, Measurements& measurements)
{
  using scalar_type = block_tensor_value_t<Center>;
  auto solve = [&](auto& ops) { return dmrg_lanczos_ground_state<scalar_type>(ops, initial, options); };

  krylov::BlockTensorMatrixFreeOps ops(initial, std::move(effective_hamiltonian));
  if constexpr (performance::measurement_level_v<Measurements> == performance::MeasurementLevel::none)
  {
    return solve(ops);
  }
  else
  {
    MeasuredDmrgKrylovOps<decltype(ops), Measurements> measured_ops(std::move(ops), measurements);
    return solve(measured_ops);
  }
}

} // namespace detail

/// \brief Optimize and replace one adjacent MPS pair with a measured ground-state DMRG step.
/// \details The operation forms the current two-site center, obtains the two
///          reusable environments outside the pair, solves the fixed sparse
///          effective Hamiltonian, performs staged block-SVD truncation, and
///          installs the directional split. The existing internal-bond label
///          is preserved. After replacement, only the completed-side
///          environment needed by the next step is rebuilt.
/// \throws std::invalid_argument If the pair, cache, direction, or options are invalid.
/// \tparam MpsChain Concrete finite MPS owner type.
/// \tparam MpoChain Concrete finite MPO owner type.
/// \tparam EnvironmentStorage Sparse local storage used by the cache.
/// \tparam Measurements Compile-time-selected performance measurement policy.
/// \tparam CenterStorage Local ordinary-block storage used by the transient center and Krylov vectors.
/// \param mps Mutable MPS whose selected pair is replaced by the fixed-work local update.
/// \param mpo Immutable MPO defining the effective Hamiltonian.
/// \param cache Environment cache attached to exactly \p mps and \p mpo.
/// \param first_site Index of the left site in the active pair.
/// \param direction Direction in which the canonical center moves.
/// \param options Fixed-work local solve and global SVD-truncation policies.
/// \param measurements Explicit performance measurement policy or collector.
/// \param center_storage Stateless policy tag selecting center allocation and block execution.
/// \return Local energy, residual and work diagnostics, and selected bond data.
template <class MpsChain, class MpoChain, SparseBlockStorage EnvironmentStorage, class Measurements,
          BlockTensorStorage CenterStorage = typename MpsChain::storage_policy>
  requires detail::TwoSiteCenterStorageFor<CenterStorage, typename MpsChain::value_type> &&
           performance::DurationMeasurementPolicy<Measurements, TwoSiteDmrgPerformanceEvent>
[[nodiscard]] auto
optimize_two_site_dmrg_bond(MpsChain& mps, MpoChain const& mpo,
                            MpoEnvironmentCache<MpsChain, MpoChain, EnvironmentStorage>& cache, std::size_t first_site,
                            MpsSweepDirection direction,
                            TwoSiteDmrgOptions<make_real_t<typename MpsChain::value_type>> const& options,
                            Measurements& measurements, CenterStorage center_storage = {})
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

  return performance::measure_duration(measurements, TwoSiteDmrgPerformanceEvent::bond_update, [&] {
    static_cast<void>(center_storage);
    auto initial = performance::measure_duration(measurements, TwoSiteDmrgPerformanceEvent::center_construction, [&] {
      return detail::make_two_site_center<CenterStorage>(mps.site(first_site), mps.site(first_site + 1));
    });
    auto local_solution =
        performance::measure_duration(measurements, TwoSiteDmrgPerformanceEvent::local_eigensolver, [&] {
          auto const& left_environment = cache.left_environment(first_site);
          auto const& right_environment = cache.right_environment(first_site + 2);
          auto placed_left = detail::place_environment_for_center<CenterStorage>(left_environment);
          auto placed_right = detail::place_environment_for_center<CenterStorage>(right_environment);
          auto effective_hamiltonian =
              TwoSiteEffectiveHamiltonian(initial, std::move(placed_left), as_block_tensor_view(mpo.site(first_site)),
                                          as_block_tensor_view(mpo.site(first_site + 1)), std::move(placed_right));
          return detail::solve_two_site_ground_state(initial, std::move(effective_hamiltonian), options.local_solver,
                                                     measurements);
        });

    real_type const local_energy = local_solution.energy;
    real_type const residual_bound = local_solution.residual_bound;
    int const iteration_count = local_solution.iteration_count;
    int const matvec_count = local_solution.matvec_count;
    auto decomposition = performance::measure_duration(measurements, TwoSiteDmrgPerformanceEvent::block_svd, [&] {
      return decompose_two_site_center(local_solution.vector, {}, measurements,
                                       TwoSiteDmrgPerformanceEvent::svd_sector_batch);
    });
    auto selection = performance::measure_duration(measurements, TwoSiteDmrgPerformanceEvent::state_selection, [&] {
      return select_svd_states(decomposition.spectrum(), options.truncation);
    });
    auto installed =
        performance::measure_duration(measurements, TwoSiteDmrgPerformanceEvent::factor_materialization, [&] {
          return replace_two_site_from_svd(mps, first_site, decomposition, selection, direction,
                                           {.bond_label = mps.site(first_site).codomain().template space<0>().label()});
        });

    performance::measure_duration(measurements, TwoSiteDmrgPerformanceEvent::environment_update, [&] {
      if (direction == MpsSweepDirection::left_to_right)
        static_cast<void>(cache.left_environment(first_site + 1));
      else
        static_cast<void>(cache.right_environment(first_site + 1));
    });

    return TwoSiteDmrgStepResult<real_type, decltype(installed)>{.first_site = first_site,
                                                                 .direction = direction,
                                                                 .local_energy = local_energy,
                                                                 .residual_bound = residual_bound,
                                                                 .iteration_count = iteration_count,
                                                                 .matvec_count = matvec_count,
                                                                 .bond = std::move(installed)};
  });
}

/// \brief Optimize and replace one adjacent MPS pair without performance measurements.
template <class MpsChain, class MpoChain, SparseBlockStorage EnvironmentStorage,
          BlockTensorStorage CenterStorage = typename MpsChain::storage_policy>
  requires detail::TwoSiteCenterStorageFor<CenterStorage, typename MpsChain::value_type>
[[nodiscard]] auto
optimize_two_site_dmrg_bond(MpsChain& mps, MpoChain const& mpo,
                            MpoEnvironmentCache<MpsChain, MpoChain, EnvironmentStorage>& cache, std::size_t first_site,
                            MpsSweepDirection direction,
                            TwoSiteDmrgOptions<make_real_t<typename MpsChain::value_type>> const& options = {},
                            CenterStorage center_storage = {})
{
  performance::NoMeasurements measurements;
  return optimize_two_site_dmrg_bond(mps, mpo, cache, first_site, direction, options, measurements, center_storage);
}

/// \brief Traverse every adjacent bond once in one measured two-site DMRG direction.
/// \details A left-to-right sweep visits `0 ... L-2`; a right-to-left sweep
///          visits `L-2 ... 0`. Each step leaves the environment on its
///          completed side ready for the next pair.
/// \throws std::invalid_argument If the chain has fewer than two sites or the direction is invalid.
/// \tparam MpsChain Concrete finite MPS owner type.
/// \tparam MpoChain Concrete finite MPO owner type.
/// \tparam EnvironmentStorage Sparse local storage used by the cache.
/// \tparam Measurements Compile-time-selected performance measurement policy.
/// \tparam CenterStorage Local ordinary-block storage used by the transient center and Krylov vectors.
/// \param mps Mutable finite MPS.
/// \param mpo Immutable finite MPO.
/// \param cache Environment cache attached to both chain objects.
/// \param direction Traversal and singular-value absorption direction.
/// \param options Shared local-solve and truncation policies for the sweep.
/// \param measurements Explicit performance measurement policy or collector.
/// \param center_storage Stateless policy tag selecting center allocation and block execution.
/// \return One result per bond in visitation order.
template <class MpsChain, class MpoChain, SparseBlockStorage EnvironmentStorage, class Measurements,
          BlockTensorStorage CenterStorage = typename MpsChain::storage_policy>
  requires detail::TwoSiteCenterStorageFor<CenterStorage, typename MpsChain::value_type> &&
           performance::DurationMeasurementPolicy<Measurements, TwoSiteDmrgPerformanceEvent>
[[nodiscard]] auto sweep_two_site_dmrg(MpsChain& mps, MpoChain const& mpo,
                                       MpoEnvironmentCache<MpsChain, MpoChain, EnvironmentStorage>& cache,
                                       MpsSweepDirection direction,
                                       TwoSiteDmrgOptions<make_real_t<typename MpsChain::value_type>> const& options,
                                       Measurements& measurements, CenterStorage center_storage = {})
{
  detail::validate_two_site_dmrg_direction(direction);
  if (mps.size() < 2) throw std::invalid_argument("two-site DMRG sweep requires at least two sites");

  return performance::measure_duration(measurements, TwoSiteDmrgPerformanceEvent::sweep, [&] {
    using result_type = decltype(optimize_two_site_dmrg_bond(mps, mpo, cache, std::size_t{}, direction, options,
                                                             measurements, center_storage));
    std::vector<result_type> result;
    result.reserve(mps.size() - 1);
    if (direction == MpsSweepDirection::left_to_right)
    {
      for (std::size_t first_site = 0; first_site + 1 < mps.size(); ++first_site)
        result.push_back(
            optimize_two_site_dmrg_bond(mps, mpo, cache, first_site, direction, options, measurements, center_storage));
    }
    else
    {
      for (std::size_t second_site = mps.size() - 1; second_site > 0; --second_site)
        result.push_back(optimize_two_site_dmrg_bond(mps, mpo, cache, second_site - 1, direction, options, measurements,
                                                     center_storage));
    }
    return result;
  });
}

/// \brief Traverse every adjacent bond once without performance measurements.
template <class MpsChain, class MpoChain, SparseBlockStorage EnvironmentStorage,
          BlockTensorStorage CenterStorage = typename MpsChain::storage_policy>
  requires detail::TwoSiteCenterStorageFor<CenterStorage, typename MpsChain::value_type>
[[nodiscard]] auto
sweep_two_site_dmrg(MpsChain& mps, MpoChain const& mpo,
                    MpoEnvironmentCache<MpsChain, MpoChain, EnvironmentStorage>& cache, MpsSweepDirection direction,
                    TwoSiteDmrgOptions<make_real_t<typename MpsChain::value_type>> const& options = {},
                    CenterStorage center_storage = {})
{
  performance::NoMeasurements measurements;
  return sweep_two_site_dmrg(mps, mpo, cache, direction, options, measurements, center_storage);
}

/// \brief Run measured alternating directional sweeps until terminal-energy convergence.
/// \details The terminal local Ritz value is treated as the installed global
///          state energy only when the terminal split has zero discarded
///          weight. Convergence compares two consecutive such terminal
///          energies. An invalid terminal energy clears the comparison state,
///          so provisional local values never stop the run. Per-bond installed
///          spectra are summarized and released after each traversal.
/// \throws std::invalid_argument If the chain, cache, or options are invalid.
/// \tparam MpsChain Concrete finite MPS owner type.
/// \tparam MpoChain Concrete finite MPO owner type.
/// \tparam EnvironmentStorage Sparse local storage used by the cache.
/// \tparam Measurements Compile-time-selected performance measurement policy.
/// \tparam CenterStorage Local ordinary-block storage used by the transient center and Krylov vectors.
/// \param mps Mutable finite MPS, initially canonical for the first direction.
/// \param mpo Immutable finite MPO.
/// \param cache Environment cache attached to both chain objects.
/// \param options Run, local-solve, and truncation policies.
/// \param measurements Explicit performance measurement policy or collector.
/// \param center_storage Stateless policy tag selecting center allocation and block execution.
/// \return Ordered sweep summaries and final convergence state.
template <class MpsChain, class MpoChain, SparseBlockStorage EnvironmentStorage, class Measurements,
          BlockTensorStorage CenterStorage = typename MpsChain::storage_policy>
  requires detail::TwoSiteCenterStorageFor<CenterStorage, typename MpsChain::value_type> &&
           performance::DurationMeasurementPolicy<Measurements, TwoSiteDmrgPerformanceEvent>
[[nodiscard]] auto run_two_site_dmrg(MpsChain& mps, MpoChain const& mpo,
                                     MpoEnvironmentCache<MpsChain, MpoChain, EnvironmentStorage>& cache,
                                     TwoSiteDmrgRunOptions<make_real_t<typename MpsChain::value_type>> const& options,
                                     Measurements& measurements, CenterStorage center_storage = {})
{
  using real_type = make_real_t<typename MpsChain::value_type>;
  detail::validate_two_site_dmrg_run_options(options);
  if (!cache.is_attached_to(mps, mpo))
    throw std::invalid_argument("two-site DMRG cache is attached to different chain objects");
  if (mps.size() < 2) throw std::invalid_argument("two-site DMRG run requires at least two sites");

  return performance::measure_duration(measurements, TwoSiteDmrgPerformanceEvent::run, [&] {
    TwoSiteDmrgRunResult<real_type> result{.sweeps = {}, .converged = false};
    result.sweeps.reserve(options.maximum_sweeps);
    std::optional<real_type> previous_terminal_energy;
    MpsSweepDirection direction = options.initial_direction;
    for (std::size_t sweep_index = 0; sweep_index < options.maximum_sweeps; ++sweep_index)
    {
      auto steps = sweep_two_site_dmrg(mps, mpo, cache, direction, options.bond_options, measurements, center_storage);
      auto const& terminal = steps.back();
      real_type maximum_discarded_weight{};
      std::size_t maximum_bond_dimension = 0;
      real_type maximum_residual_bound{};
      int total_iteration_count = 0;
      int total_matvec_count = 0;
      for (auto const& step : steps)
      {
        maximum_discarded_weight = std::max(maximum_discarded_weight, step.bond.truncation.discarded_weight);
        maximum_bond_dimension = std::max(maximum_bond_dimension, step.bond.truncation.retained_rank);
        maximum_residual_bound = std::max(maximum_residual_bound, step.residual_bound);
        total_iteration_count += step.iteration_count;
        total_matvec_count += step.matvec_count;
      }

      real_type const terminal_discarded_weight = terminal.bond.truncation.discarded_weight;
      bool const terminal_energy_is_global = terminal_discarded_weight == real_type{};
      std::optional<real_type> energy_change;
      bool converged = false;
      if (terminal_energy_is_global)
      {
        if (previous_terminal_energy)
        {
          energy_change = detail::dmrg_abs(terminal.local_energy - *previous_terminal_energy);
          converged = detail::terminal_energies_converged(*previous_terminal_energy, terminal.local_energy,
                                                          options.energy_tolerance);
        }
        previous_terminal_energy = terminal.local_energy;
      }
      else
      {
        previous_terminal_energy.reset();
      }

      result.sweeps.push_back({.sweep_index = sweep_index,
                               .direction = direction,
                               .terminal_local_energy = terminal.local_energy,
                               .terminal_discarded_weight = terminal_discarded_weight,
                               .terminal_energy_is_global = terminal_energy_is_global,
                               .energy_change = energy_change,
                               .maximum_discarded_weight = maximum_discarded_weight,
                               .maximum_bond_dimension = maximum_bond_dimension,
                               .maximum_residual_bound = maximum_residual_bound,
                               .total_iteration_count = total_iteration_count,
                               .total_matvec_count = total_matvec_count,
                               .converged = converged});
      if (converged)
      {
        result.converged = true;
        break;
      }
      direction = detail::opposite_dmrg_direction(direction);
    }
    return result;
  });
}

/// \brief Run alternating directional sweeps without performance measurements.
template <class MpsChain, class MpoChain, SparseBlockStorage EnvironmentStorage,
          BlockTensorStorage CenterStorage = typename MpsChain::storage_policy>
  requires detail::TwoSiteCenterStorageFor<CenterStorage, typename MpsChain::value_type>
[[nodiscard]] auto
run_two_site_dmrg(MpsChain& mps, MpoChain const& mpo,
                  MpoEnvironmentCache<MpsChain, MpoChain, EnvironmentStorage>& cache,
                  TwoSiteDmrgRunOptions<make_real_t<typename MpsChain::value_type>> const& options = {},
                  CenterStorage center_storage = {})
{
  performance::NoMeasurements measurements;
  return run_two_site_dmrg(mps, mpo, cache, options, measurements, center_storage);
}

} // namespace uni20::tensor_network
