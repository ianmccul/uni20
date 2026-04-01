# Models

Uni20 now has a small model layer above the operator infrastructure.

## Scope

This layer is intentionally narrow.

- `SpinHalfSite` is the first concrete local model bundle.
- `make_spin_half_u1_site()` constructs a U(1)-symmetric spin-1/2 local space and
  its standard symmetry-pure local operators.
- `make_spin_half_heisenberg_bulk_component()` and
  `make_spin_half_heisenberg_mpo()` provide the first hand-built triangular MPO
  path for DMRG.

This is not the final long-term model hierarchy.

## `SpinHalfSite`

`uni20::SpinHalfSite` bundles:

- `symmetry`
- `space`
- `up`, `down`
- `identity`
- `sz`
- `sp`
- `sm`
- `sigma_z`

The local states are ordered as:

- `|up>` with `Sz = +1/2`
- `|down>` with `Sz = -1/2`

under the chosen U(1) component name, which defaults to `"Sz"`.

## Symmetry-Pure Operators Only

The current `LocalOperator` type always carries one definite `transforms_as()`
label, so only symmetry-pure local operators are representable directly.

That means:

- `I`
- `Sz`
- `S+`
- `S-`
- `sigma_z`

fit naturally in the current U(1) layer.

But operators such as `Sx`, `Sy`, `sigma_x`, and `sigma_y` do not transform as a
single U(1) charge, so they are not represented as one `LocalOperator` here.

## Spin-1/2 Heisenberg MPO

`make_spin_half_heisenberg_bulk_component(site, j, hz)` builds one repeated
upper-triangular site component for the Hamiltonian

`H = J sum_i [ 1/2 (S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) + S^z_i S^z_{i+1} ] + h_z sum_i S^z_i`

with virtual channel order:

- `0`
- `-1`
- `+1`
- `0`
- `0`

interpreted as:

- start
- pending `S^-`
- pending `S^+`
- pending `S^z`
- finish

`make_spin_half_heisenberg_mpo(length, site, j, hz)` then constructs a finite
triangular MPO by repeating this same bulk component at every site.

## Boundary Convention

The current first-pass builder keeps the bulk virtual space unchanged at the
boundaries rather than reducing the bond dimension there.

This is deliberate:

- it keeps the implementation simple
- it matches the immediate DMRG prototype needs
- boundary-specific optimizations can be added later without changing the bulk
  operator layout
