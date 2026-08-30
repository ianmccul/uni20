# Finite Chains And Environment Caches

**Status:** implemented immediate-host ownership and cache checkpoint.

The finite tensor-network layer now owns homogeneous MPS and MPO site chains:

```cpp
FiniteMps<Scalar, Bond, Physical, Storage>
FiniteMpo<Scalar, Auxiliary, Physical, Storage>
```

Every site remains an ordinary symmetry-preserving `BlockTensor`. The chain
owner adds ordering, exact connectivity, and a controlled mutation boundary; it
does not introduce another numerical tensor container.

## Chain Invariants

Both owners require a nonempty chain and one shared `Symmetry`. Adjacent MPS
right and left bond spaces must compare exactly equal. Adjacent MPO right and
left auxiliary spaces have the same requirement. Exact comparison includes
the leg label as well as charges and multiplicities.

Sites are exposed read-only. The supported mutation paths are:

```cpp
mps.replace_site(i, replacement);
mpo.replace_site(i, replacement);
mps.replace_pair(i, first, second);
```

Single-site replacement preserves the complete domain and codomain. It is the
path for changing numerical values without changing tensor-network structure.

Adjacent MPS replacement preserves the pair's two external bonds and both
physical spaces, but may replace the shared internal bond. Both replacement
sites are validated before either chain position changes. This is the
structural operation needed after a two-site SVD chooses a new bond space.

Each successful replacement increments the affected site revision. Mutable
site references are deliberately absent: dependent caches use these revisions
as their coherence protocol.

## Environment Cache

`MpoEnvironmentCache` borrows a compatible `FiniteMps` and `FiniteMpo`. For a
chain of length `L`, it has `L+1` entries in each direction:

```text
left[0]       = left identity boundary
left[i + 1]   = extend_left_environment(left[i], mps[i], mpo[i])

right[L]      = right identity boundary
right[i]      = extend_right_environment(right[i + 1], mps[i], mpo[i])
```

The left and right MPO boundary-state indices are explicit constructor
arguments. Construction validates equal chain lengths, symmetry, and exact
physical spaces at every site. It creates only the two identity boundaries;
`left_environment(bond)` and `right_environment(bond)` build missing entries
on demand, while `build_all()` constructs both complete directional caches.

## Invalidation

Before observing or building an entry, the cache compares current MPS and MPO
site revisions with its snapshot. A change at site `i` invalidates exactly:

```text
left[i + 1 ... L]
right[0 ... i]
```

Replacing adjacent sites `i` and `i+1` takes the union. Consequently,
`left[i]` and `right[i+2]` remain reusable around a two-site optimization
center. Rebuilding starts from those nearest valid entries.

References returned by `left_environment()` or `right_environment()` remain
valid only until a later cache operation observes a replacement on which that
entry depends. A local effective-Hamiltonian object that borrows those
environments must therefore be destroyed before installing its optimized site
or site pair.

The cache is a borrowed object. Its MPS and MPO owners must outlive it, remain
at their original addresses, and be mutated only through the replacement
members while borrowed. Moving or assigning an owner invalidates the cache as
an object-lifetime operation rather than a site update.

## Current Limits

This first cache uses immediate host `BlockSpace` MPS bonds, `LocalSpace` MPO
auxiliaries, and sparse host environment storage. It stores expectation-value
environments with one MPS as both bra and ket. Distinct bra/ket chain owners,
per-block async epochs, backend placement, reusable contraction plans, CUDA,
and MPI distribution remain later extensions.

The two-site DMRG path now combines the cache, local solve, selected block-SVD
split, pair replacement, and completed-side environment refresh. See
[Directional Two-Site DMRG Sweeps](two_site_dmrg_sweeps.md). Sweep-level
convergence, measurement, and adaptive policies remain separate layers.
