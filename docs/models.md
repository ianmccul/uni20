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
- `FermiHubbardSite` is the first U(1)xU(1) local model bundle.
- `make_fermi_hubbard_u1u1_site()` constructs the spinful fermion local space,
  parity operator, creation/annihilation operators, and parity-modified hopping
  operators.
- `make_fermi_hubbard_bulk_component()` and `make_fermi_hubbard_mpo()` provide a
  nearest-neighbor Fermi-Hubbard triangular MPO path for block-sparse DMRG.

This is not the final long-term model hierarchy.

## SpinHalfSite

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

The same rule applies to the Hubbard helpers. Each creation or annihilation
operator carries one definite `(N,Sz)` transform charge. Fermion parity is a
separate scalar local operator used to encode the nearest-neighbor
Jordan-Wigner sign convention in the MPO.

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

## U(1)xU(1) Fermi-Hubbard MPO

`make_fermi_hubbard_u1u1_site()` follows the Matrix Product Toolkit
`FermionU1U1` convention:

- symmetry factors are `N:U(1),Sz:U(1)` by default
- local states are ordered as `|0>`, `|up down>`, `|down>`, `|up>`
- charges are `(0,0)`, `(2,0)`, `(1,-1/2)`, `(1,+1/2)`
- `P=(-1)^N` is stored as an explicit scalar local operator

The down-spin creation convention includes the local fermion sign:

```text
CHdown |0>  = |down>
CHdown |up> = -|up down>
```

`make_fermi_hubbard_bulk_component(site, t, U)` builds one repeated
upper-triangular component for

```text
H = -t sum_i,sigma (c^dagger_i,sigma c_{i+1,sigma}
                    + c^dagger_{i+1,sigma} c_{i,sigma})
    + U sum_i n_i,up n_i,down
```

The virtual channel order is:

- `0`
- `Cup`
- `CHup`
- `Cdown`
- `CHdown`
- `0`

The hopping channels store `O_i P_i` on the left site and the complementary
fermion operator on the right site. This explicitly encodes the adjacent-site
Jordan-Wigner sign and matches the Matrix Product Toolkit convention
`-dot(CH(0), C(1)) + dot(C(0), CH(1))`.

`make_fermi_hubbard_mpo(length, site, t, U)` constructs a finite triangular MPO
by repeating this same bulk component at every site.

The `fermi_hubbard_u1u1_dmrg` example currently targets the half-filled
spin-zero sector. For an even chain length `L`, it initializes the strict
U(1)xU(1) path from the alternating product state

```text
|up>, |down>, |up>, |down>, ...
```

The cumulative MPS bond convention is `q_right = q_left + q_physical`, so this
sets the final right boundary sector to `(N=L, Sz=0)`. Two-site SVD truncation
then rebuilds intermediate bond spaces only from symmetry-allowed SVD sectors,
so the sweep remains in that total sector. Odd lengths are rejected by the
example because half filling with `Sz=0` is not possible for the alternating
single-occupancy seed.

## Boundary Convention

The current first-pass builder keeps the bulk virtual space unchanged at the
boundaries rather than reducing the bond dimension there.

This is deliberate:

- it keeps the implementation simple
- it matches the immediate DMRG prototype needs
- boundary-specific optimizations can be added later without changing the bulk
  operator layout
