# Directional Two-Site DMRG Sweeps

**Status:** implemented immediate-host finite-chain ground-state run.

The first sweep API combines the existing finite-chain components without
introducing a new tensor representation:

```cpp
TwoSiteDmrgOptions<double> options;
options.local_solver.matvec_iterations = 4;
options.truncation.maximum_retained_extent = 256;
options.truncation.maximum_discarded_weight = 1.0e-10;

auto steps = sweep_two_site_dmrg(
    mps,
    mpo,
    environments,
    MpsSweepDirection::left_to_right,
    options);
```

Every Krylov vector, effective-Hamiltonian operand, SVD factor, and installed
site remains a symmetry-aware `BlockTensor`. No whole-chain or whole-center
dense projection is used.

For repeated alternating sweeps:

```cpp
TwoSiteDmrgRunOptions<double> run_options;
run_options.maximum_sweeps = 10;
run_options.energy_tolerance = 1.0e-10;
run_options.bond_options = options;

auto result = run_two_site_dmrg(mps, mpo, environments, run_options);
```

One run-level sweep is one complete directional traversal. Directions alternate
from `initial_direction`, which defaults to left-to-right.

The optional final storage-policy tag selects the representation and block
execution policy of the transient two-site center and all Krylov vectors:

```cpp
auto result = run_two_site_dmrg(
    mps,
    mpo,
    environments,
    run_options,
    ParallelPackedCompleteBlockStorage<>{});
```

This does not change the persistent MPS or MPO storage. With an active
`TbbScheduler`, independent center-output blocks execute through synchronous
lightweight task batches, while every contribution to one output block remains
serial within its batch item. The environment cache has its own storage-policy
template parameter and may select the same execution policy independently.

## Performance Measurements

Performance instrumentation is opt-in and separate from the mathematical run
result. The coarse collector records inclusive host wall durations for the run,
each sweep and bond update, center construction, the local eigensolver, block
SVD, state selection, selected-factor materialization, environment update, and
each effective-Hamiltonian application within the local eigensolver:

```cpp
TwoSiteDmrgPerformanceMeasurements measurements;

auto result = run_two_site_dmrg(
    mps,
    mpo,
    environments,
    run_options,
    measurements,
    ParallelPackedCompleteBlockStorage<>{});
```

`DetailedTwoSiteDmrgPerformanceMeasurements` additionally times Krylov vector
allocation, update, and reduction calls and retains item timing for every
per-charge block-SVD batch. Ordinary overloads instantiate
`performance::NoMeasurements`; optimized builds contain no profiling branch,
clock read, counter, or measurement state on that path.

Parent and child phase durations are inclusive and must not be added together.
Host wall timing around future asynchronous CUDA work would measure submission,
not device execution; provider-domain timings require their own event mechanism.
The fine Krylov events intentionally perform clock reads around individual
vector primitives and therefore belong only to detailed attribution runs. The
generic measurement levels, batch fields, and overhead contract are in
[Performance Measurements](../development/performance_measurements.md).

## One Bond Step

`optimize_two_site_dmrg_bond()` applies this sequence to adjacent sites `i`
and `i+1`:

1. Contract the two current MPS sites over their shared bond using the selected
   center storage. A complete policy allocates every symmetry-legal center block
   directly and zeros blocks without a contribution. For compatibility, a
   sparse center policy is widened once into an explicitly complete sparse key
   set. Blocks absent from the current MPS are exact zero but remain available
   to the local Hamiltonian.
2. Obtain `left[i]` and `right[i+2]` from the attached
   `MpoEnvironmentCache`.
3. Compile a fixed-center `TwoSiteEffectiveHamiltonian` from those
   environments and the two MPO sites. This also prepares the host R/A/B/C
   grouping, output order, and reusable intermediate workspace. The sweep
   retains zero-copy identity views for the local solve rather than copying the
   owning BlockTensor payloads.
4. Perform the configured fixed number of three-term Lanczos steps and use the
   smallest Ritz vector of that local projection.
5. Apply the staged block SVD and global charge-sector truncation policy.
6. Preserve the existing internal-bond label and install the directional MPS
   split with `FiniteMps::replace_pair()`.
7. Rebuild the environment immediately beyond the completed site.

The default is four effective-Hamiltonian applications per bond. The local
solver does not test Ritz convergence, perform a restart, or fully
reorthogonalize the basis. It exits early only for an exact numerical
invariant-subspace breakdown or when the local vector-space dimension is less
than four. The resulting approximate Ritz vector is installed and the sweep
continues so that the environments, rather than one stale local problem, are
converged. Invalid solver or truncation options and failed SVD materialization
leave the MPS unchanged.

The complete local key structure is essential for optimization. A sparse MPS
may omit a numerically zero site block even though the Hamiltonian couples its
current center into the corresponding legal two-site block. Restricting Krylov
to the contraction's initially stored keys would reject that valid coupling or
solve the wrong local problem.

The cache must be attached to the exact MPS and MPO objects passed to the
step, not merely owners of the same types. This prevents a compound operation
from solving with environments belonging to another chain instance.

## Direction And Cache State

For a left-to-right step, singular values are absorbed into site `i+1`. Site
`i` is left as the selected isometry, and the step rebuilds `left[i+1]` from
the still-valid `left[i]`:

```text
active pair:       i, i+1
reused inputs:     left[i], right[i+2]
completed site:    i
refreshed entry:   left[i+1]
next pair:         i+1, i+2
```

For a right-to-left step, singular values are absorbed into site `i`. Site
`i+1` is the selected isometry, and the step rebuilds `right[i+1]` from the
still-valid `right[i+2]`:

```text
active pair:       i, i+1
reused inputs:     left[i], right[i+2]
completed site:    i+1
refreshed entry:   right[i+1]
next pair:         i-1, i
```

The revision cache invalidates the other entries affected by pair replacement.
They remain absent until a later step requires them.

## Results

Each `TwoSiteDmrgStepResult` records:

- the left site index and sweep direction;
- the lowest local Ritz value before truncation;
- its three-term-recurrence Ritz residual estimate;
- Lanczos projection-step and matrix-vector counts;
- the selected diagonal Schmidt tensor and exact truncation statistics.

`local_energy` is deliberately not called the installed-state energy. SVD
truncation changes the optimized center and can raise its expectation value.
The run controller does not perform a second effective-Hamiltonian apply to
measure that post-truncation value.

## Full Traversal

`sweep_two_site_dmrg()` visits every bond exactly once:

```text
left_to_right:  0, 1, ..., L-2
right_to_left:  L-2, ..., 1, 0
```

It returns step results in visitation order. Environments are built lazily;
the caller does not need to call `build_all()` before a sweep. Repeated sweeps
reuse the two chain-boundary entries and the directional entries retained or
refreshed by preceding steps.

## Alternating Run And Convergence

`run_two_site_dmrg()` repeatedly calls the directional traversal and retains a
compact `TwoSiteDmrgSweepSummary` rather than all installed bond spectra. Each
summary records:

- the sweep index and direction;
- the terminal local Ritz energy and terminal discarded weight;
- whether that terminal energy is the installed global-state energy;
- its change from the preceding valid terminal energy;
- maximum discarded weight, retained bond dimension, and residual bound;
- total local iteration and effective-Hamiltonian application counts.

At the terminal pair, the active bond lies next to a one-dimensional chain
boundary. If its SVD has zero discarded weight, factorization does not change
the optimized center. Under the mixed-canonical sweep invariant, the terminal
local Ritz value is therefore also the installed whole-chain expectation
value. Only such energies participate in convergence. A nonzero terminal
discarded weight clears the preceding comparison state, so even a large energy
tolerance cannot report convergence from provisional local values.

Convergence compares consecutive valid terminal energies with

```text
abs(E_n - E_(n-1)) <= tolerance * max(1, abs(E_n), abs(E_(n-1)))
```

A zero configured tolerance selects `100 * numeric_limits<Real>::epsilon()`.
At least two valid terminal energies are therefore required. Reaching
`maximum_sweeps` without satisfying the criterion returns normally with
`converged == false`.

The input MPS must be canonical for `initial_direction`. A normalized rank-one
product MPS is canonical from both directions; in particular,
`models::make_neel_product_mps()` satisfies this precondition. Each directional
split preserves the mixed-canonical state needed by the next update.

The registered `spin_half_heisenberg_dmrg_example` exercises this path on the
length-four open chain. Without truncation it reaches the exact energy
`-(3 + 2 sqrt(3))/4` while preserving U(1) block structure.

The first larger CPU scaling measurements, the exact block-level parallel
boundary, and cross-library orientation points are recorded in
[DMRG Performance Baselines](dmrg_performance_baselines.md). The current
effective-Hamiltonian plan schedules estimated-expensive output groups first.
Per-charge SVDs now run as estimated-cost-ordered lightweight batch items.
Selected-factor construction remains serial and is the next parallel checkpoint.

## Current Limits

This checkpoint is synchronous and immediate-host. It requires `BlockSpace`
MPS bonds, `LocalSpace` MPO auxiliaries, bosonic Abelian symmetry, LAPACK scalar
support and the current sparse MPO effective-Hamiltonian planner.

It does not yet provide energy or variance measurement after truncation,
generic canonicalization of an arbitrary initial MPS, adaptive local-work
schedules, noise or subspace expansion, excited-state projection, per-bond
truncation policies, async block solves, CUDA, MPI, or checkpoint/restart.
Those are separate layers over the verified directional state transition.
Legal center keys are initially found by scanning the Cartesian product of
key-bearing factors; a charge-indexed planner can replace that structural setup
when large sector counts make it material.
