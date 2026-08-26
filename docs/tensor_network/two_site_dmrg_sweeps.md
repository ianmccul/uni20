# Directional Two-Site DMRG Sweeps

**Status:** implemented immediate-host finite-chain ground-state checkpoint.

The first sweep API combines the existing finite-chain components without
introducing a new tensor representation:

```cpp
TwoSiteDmrgOptions<double> options;
options.eigensolver.tolerance = 1.0e-12;
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

## One Bond Step

`optimize_two_site_dmrg_bond()` applies this sequence to adjacent sites `i`
and `i+1`:

1. Contract the two current MPS sites over their shared bond, then copy that
   value into a center storing every symmetry-legal key for the fixed external
   bonds and physical spaces. Blocks absent from the current MPS are exact
   zero but remain available to the local Hamiltonian.
2. Obtain `left[i]` and `right[i+2]` from the attached
   `MpoEnvironmentCache`.
3. Compile a fixed-center `TwoSiteEffectiveHamiltonian` from those
   environments and the two MPO sites.
4. Solve for one smallest-algebraic Ritz vector with the native Hermitian
   Lanczos implementation.
5. Apply the staged block SVD and global charge-sector truncation policy.
6. Preserve the existing internal-bond label and install the directional MPS
   split with `FiniteMps::replace_pair()`.
7. Rebuild the environment immediately beyond the completed site.

The default local Krylov dimension is the standard Uni20 default. A problem
whose selected Krylov dimension covers the complete local vector space uses
the full-projection solver. A larger problem uses implicit restart. The MPS is
not changed unless the solver reports one converged ground-state vector.
Invalid solver or truncation options and failed SVD materialization also leave
the MPS unchanged.

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
- its Ritz residual bound;
- Lanczos iteration/restart and matrix-vector counts;
- the selected diagonal Schmidt tensor and exact truncation statistics.

`local_energy` is deliberately not called the installed-state energy. SVD
truncation changes the optimized center and can raise its expectation value.
The first checkpoint does not perform a second effective-Hamiltonian apply to
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

## Current Limits

This checkpoint is synchronous and immediate-host. It requires `BlockSpace`
MPS bonds, `LocalSpace` MPO auxiliaries, bosonic Abelian symmetry, LAPACK scalar
support, the current sparse MPO effective-Hamiltonian planner, and a converged
single-vector ground-state solve.

It does not yet provide convergence across multiple sweeps, energy or variance
measurement after truncation, normalization policy, adaptive local tolerances,
noise or subspace expansion, excited-state projection, per-bond truncation
policies, async block solves, CUDA, MPI, or checkpoint/restart. Those are
separate layers over the verified directional state transition. Legal center
keys are initially found by scanning the Cartesian product of key-bearing
factors; a charge-indexed planner can replace that structural setup when large
sector counts make it material.
