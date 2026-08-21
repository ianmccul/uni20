# TensorContraction Integration Local-Operator Layer

**Status:** functional integration-branch reference. The `LocalOperator`,
`OperatorComponent`, and `FiniteTriangularMPO` classes described here are not
present on `main`. Their operator, charge, and MPO semantics are requirements
for the pure-Uni20 symmetry-aware tensor-network layer.

## Scope

This layer is intentionally narrow.

- `LocalSpace` is a semantic wrapper around `QNumList`.
- `LocalOperator` is the first concrete operator object.
- `OperatorComponent` is the per-site MPO object.
- `FiniteTriangularMPO` is the first lattice-level MPO type used by the DMRG
  path.
- Concrete model helpers from the same branch are described in
  `docs/tensor_network/models.md`.

## LocalSpace

`uni20::LocalSpace`
represents an explicit sparse state space used by the operator layer.

It is backed by `QNumList`, so it:

- preserves ordering
- allows repeated quantum numbers
- remains explicitly sparse
- carries a `Symmetry` even when empty

In the prototype, `LocalSpace` was used for both:

- physical on-site spaces
- MPO auxiliary / virtual bond spaces

## LocalOperator

`uni20::LocalOperator`
is conceptually the first realization of a symmetry-aware morphism whose
domain contains the ket local space and whose codomain contains the bra local
space, with the operator irrep as a second domain factor:

```text
ket_space tensor operator_irrep -> bra_space
```

but the prototype used a dedicated class rather than a generic tensor
instantiation. This follows the independent concrete-space/`Dual<S>` and
`Domain`/`Codomain` model in
[Spaces, Duals, and Tensor Morphisms](../symmetry/spaces_duals_and_morphisms.md);
it does not require a `co<LocalSpace>` direction wrapper.

It stores:

- `bra_space`
- `ket_space`
- `transforms_as()`
- a sparse coefficient matrix over explicit local states

The prototype coefficient storage was `SparseMatrix<double>`.

## OperatorComponent

`uni20::OperatorComponent` is the next level up.

It represents one site component of an MPO-like object:

- local bra space
- local ket space
- left virtual space
- right virtual space
- a sparse matrix of `LocalOperator`

This matches the practical point of view that a site of an MPO is a matrix of
local operators, not primarily a generic four-leg tensor.

`OperatorComponent` also exposes a separate `is_upper_triangular(...)`
predicate on its virtual indices. For rectangular components, the branch
convention forbade entries strictly below the main diagonal.

## FiniteTriangularMPO

`uni20::FiniteTriangularMPO` is the first lattice-level MPO container.

It stored a finite sequence of `OperatorComponent` sites and enforced:

- one shared symmetry across the chain
- exact matching of adjacent virtual spaces
- upper-triangular virtual structure at every site

This is intentionally much narrower than a full MPO class hierarchy.

## Integration-Branch API

The prototype `LocalSpace` supported:

- construction from `Symmetry`, `QNum`, or `QNumList`
- `symmetry()`, `size()`, `empty()`
- `push_back()`, `clear()`
- `contains()`
- indexed access and iteration
- `qnums()`

The prototype `LocalOperator` supported:

- construction from `bra_space`, `ket_space`, and `transforms_as`
- optional construction from an existing sparse coefficient matrix
- `bra_space()`, `ket_space()`, `symmetry()`
- `transforms_as()`
- `rows()`, `cols()`, `nnz()`
- sparse coefficient mutation and lookup via `insert_or_assign()`, `erase()`,
  `contains()`, `at()`, and `clear()`

The prototype `OperatorComponent` supported:

- construction from local spaces plus left/right virtual `LocalSpace`s
- optional construction from an existing sparse matrix of `LocalOperator`
- `local_bra_space()`, `local_ket_space()`
- `left_virtual_space()`, `right_virtual_space()`, `symmetry()`
- `rows()`, `cols()`, `nnz()`
- sparse entry mutation and lookup via `insert_or_assign()`, `erase()`,
  `contains()`, `at()`, and `clear()`
- `is_upper_triangular(component)`

The prototype `FiniteTriangularMPO` supported:

- construction from a site sequence
- `size()`, `empty()`, indexed access, iteration
- `symmetry()`
- `left_boundary_virtual_space()`, `right_boundary_virtual_space()`
- `check_structure()`
- `is_upper_triangular(mpo)`

## Planned Use

The intended next steps are:

- extend the model layer beyond the first spin-1/2 helpers
- add richer MPO transformations and manipulations on top of `OperatorComponent`
- compile that host-side MPO representation into TensorContraction block lists
  and coefficient terms
