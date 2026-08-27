# Directional Two-Site MPS Splitting

**Status:** implemented immediate-host block-SVD and finite-chain replacement
checkpoint.

The two-site split keeps Uni20's staged truncation model. Factorization and
selection remain separate:

```cpp
auto decomposition = decompose_two_site_center(center);
auto selection = select_svd_states(decomposition.spectrum(), policy);

auto installed = replace_two_site_from_svd(
    mps,
    first_site,
    decomposition,
    selection,
    MpsSweepDirection::left_to_right,
    {.bond_label = "bond"});
```

The reusable decomposition exposes the complete singular spectrum. The caller
may apply a standard truncation policy, construct an explicit selection, or
materialize other selected or null-space factors before changing the MPS.

## Matrix Boundary

A canonical two-site center has boundary:

```text
Domain<left bond, left physical, right physical>
    -> Codomain<right bond>
```

`decompose_two_site_center()` creates a zero-copy repartition view:

```text
Domain<left bond, left physical>
    -> Codomain<right bond, Dual<right physical>>
```

and passes that immediate view to `block_svd()`. No whole-tensor dense or
symmetry-erasing projection is introduced. As in `block_svd`, one assembled
dense matrix is factorized independently per conserved charge.

## Factor Orientation

For the morphism convention used by the block-SVD API, the selected matrix is
reconstructed as:

```text
right_singular_vectors_adjoint
    * singular_values
    * left_singular_vectors
```

The names describe the provider SVD factors. Their tensor-network roles after
the center repartition are:

| Factor | MPS role |
| --- | --- |
| `right_singular_vectors_adjoint` | Left site, before absorption |
| `left_singular_vectors` | Right site, after bending the right physical leg back to the domain |

The sweep direction determines where the diagonal tensor is absorbed:

```text
left_to_right:
    first  = right_adjoint
    second = repartition(singular_values * left)

right_to_left:
    first  = right_adjoint * singular_values
    second = repartition(left)
```

Thus the site left behind by the sweep is the selected isometry and the site in
the direction of travel carries the canonical center.

## Materialization And Installation

`materialize_two_site_mps_split()` constructs two owning `MpsSite` values in a
selected immediate sparse storage policy with ordinary dense blocks. Async,
complete, and generalized-diagonal site policies are not materialization targets
for this synchronous path. The operation also returns the real diagonal
`singular_values` tensor, exact truncation statistics, and the absorption
direction. The selected state set must be nonempty; an empty internal MPS bond
is rejected even though empty selections remain useful for other block-SVD
materializations.

`replace_two_site_from_svd()` selects the finite MPS storage policy, constructs
both sites, and then calls `FiniteMps::replace_pair()`. The chain validates both
external bonds, both physical spaces, and the new shared internal bond before
installing the pair. The returned `InstalledMpsBond` retains the diagonal
Schmidt tensor and truncation diagnostics after its values have been absorbed
into one site.

Replacing the pair increments both site revisions. An attached
`MpoEnvironmentCache` consequently preserves `left[i]` and `right[i+2]` while
invalidating all entries that depend on either replacement site. Any local
effective-Hamiltonian object borrowing the invalidated environments must be
destroyed before replacement.

## Current Limits

The first path is synchronous and immediate-host. It uses the existing LAPACK
block-SVD, sparse BlockTensor contractions, and explicit owning materialization
of the two replacement sites. It supports real and complex LAPACK scalars, with
the real singular values multiplied into complex factors through ordinary
BlockTensor contraction.

Move-aware materialization, per-block async factorization, cuSOLVER, distributed
sector selection, normalization policy, and canonical-form diagnostics remain
later work. The immediate-host ground-state DMRG path now composes this split
with cached environments, the effective Hamiltonian, native Lanczos, pair
replacement, and incremental cache refresh. See
[Directional Two-Site DMRG Sweeps](two_site_dmrg_sweeps.md).
