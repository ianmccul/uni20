# Spin-Half Model Builders

**Status:** implemented U(1) spin-half construction checkpoint. The historical
`tensorcontraction-integration` branch remains the reference for later
U(1)xU(1) Fermi-Hubbard builders.

The model layer constructs symmetry-aware finite-chain values over the ordinary
`BlockTensor`, `FiniteMps`, and `FiniteMpo` types. It does not introduce a
second tensor representation.

## Local Space

`models::make_spin_half_u1_site()` returns `SpinHalfU1Site`, containing:

- the chosen one-component U(1) `Symmetry`;
- a `LocalSpace` ordered as `|up>`, `|down>`;
- the corresponding `+1/2` and `-1/2` `QNum` values.

The component name defaults to `"Sz"`. The explicit ordering is used by the
MPS and MPO builders, while all legality still follows from the stored quantum
numbers rather than positional assumptions in contraction code.

```cpp
auto const local = models::make_spin_half_u1_site();
```

## Néel Product MPS

`models::make_neel_product_mps()` constructs an alternating normalized product
state. Its default pattern is

```text
|up down up down ...>
```

and `SpinHalfState::down` selects the opposite first state. Every site contains
one scalar block of value one. Bond `i` contains one dimension-one sector whose
charge is the cumulative sum of physical charges to its left:

```text
q_bond(0) = 0
q_bond(i+1) = q_bond(i) + q_physical(i)
```

Consequently each stored key obeys the MPS selection rule
`q_right = q_left + q_physical`, and the final boundary records the total-charge
sector. The rank-one state is normalized and both left- and right-canonical.
This is the intended initial state for the first finite DMRG driver, so that
driver does not require a separate canonicalization pass.

```cpp
auto mps = models::make_neel_product_mps(20, local);
```

## Open Heisenberg MPO

`models::make_spin_half_heisenberg_mpo()` constructs

```text
H = J sum_i [S_i^z S_(i+1)^z
             + 1/2 (S_i^+ S_(i+1)^- + S_i^- S_(i+1)^+)]
    + h_z sum_i S_i^z
```

as a sparse `FiniteMpo`. The five-state interior auxiliary is ordered as:

| Coordinate | Charge | Meaning |
|---:|---:|---|
| 0 | 0 | start |
| 1 | -1 | pending `S-` channel, opened by `S+` |
| 2 | +1 | pending `S+` channel, opened by `S-` |
| 3 | 0 | pending `Sz` channel |
| 4 | 0 | finish |

In left-auxiliary row and right-auxiliary column order, the bulk operator is:

```text
[ I   S+   S-   Sz   h_z Sz ]
[ 0    0    0    0   J/2 S- ]
[ 0    0    0    0   J/2 S+ ]
[ 0    0    0    0   J Sz   ]
[ 0    0    0    0   I      ]
```

The left and right boundary auxiliary spaces each contain one scalar state.
The first site retains the start row, the last site retains the finish column,
and boundary coordinate zero therefore selects exactly the open-chain
Hamiltonian. This is the boundary convention consumed directly by
`MpoEnvironmentCache(mps, mpo, 0, 0)`.

```cpp
auto const mpo = models::make_spin_half_heisenberg_mpo(20, local, 1.0);
tensor_network::MpoEnvironmentCache environments(mps, mpo, 0, 0);

tensor_network::TwoSiteDmrgRunOptions<double> options;
options.maximum_sweeps = 10;
options.energy_tolerance = 1.0e-10;
auto result = tensor_network::run_two_site_dmrg(
    mps, mpo, environments, options);
```

The run alternates complete directional traversals. The product MPS is
canonical from both directions, so it satisfies the first-run precondition
without a separate canonicalization pass. See
[Directional Two-Site DMRG Sweeps](two_site_dmrg_sweeps.md) for the terminal
energy convergence contract and current truncation limits. The registered
[model example](../../examples/models/) includes four-site analytic validation
and a reproducible 20-site comparison against Matrix Product Toolkit. Larger
development timings and the rules for interpreting cross-library measurements
are recorded in
[DMRG Performance Baselines](dmrg_performance_baselines.md).

For a one-site chain only the longitudinal-field term remains. A zero-length
MPS or MPO is rejected. Couplings are real even when the requested storage
scalar is complex, preserving the Hermitian model contract required by the
current symmetric Lanczos DMRG path.

## Storage And Labels

Both builders currently return packed sparse host storage. This makes the
small rank-one MPS and scalar MPO blocks immediately usable by the current host
environment and sweep implementations while retaining the ordinary storage
policy boundary for later builders.

Every constructed leg receives a stable string label. Shared MPS and MPO bonds
use exactly equal copied space values, including that label, so finite-chain
connectivity validation remains strict. Directional SVD replacement preserves
the internal MPS bond label.

## Later Models

The next model checkpoint can rebuild the integration branch's U(1)xU(1)
Fermi-Hubbard convention:

- local states `|0>`, `|up down>`, `|down>`, `|up>`;
- charges `(0,0)`, `(2,0)`, `(1,-1/2)`, `(1,+1/2)`;
- explicit fermion parity and signed down-spin creation;
- pending hopping channels in a triangular MPO.

That work should reuse the current `FiniteMps`/`FiniteMpo` construction surface
and preserve its two symmetry factors. It must not lower the U(1)xU(1) state to
a dense no-symmetry model.
