# Local Operators

Uni20 currently provides a minimal operator layer for local physical operators
and per-site MPO components.

## Scope

This layer is intentionally narrow.

- `LocalSpace` is a semantic wrapper around `QNumList`.
- `LocalOperator` is the first concrete operator object.
- `OperatorComponent` is the per-site MPO object.
- `FiniteTriangularMPO` is the first lattice-level MPO type used by the DMRG
  path.
- Concrete model helpers now live in `docs/models.md`.

## `LocalSpace`

`uni20::LocalSpace`
represents an explicit sparse state space used by the operator layer.

It is backed by `QNumList`, so it:

- preserves ordering
- allows repeated quantum numbers
- remains explicitly sparse
- carries a `Symmetry` even when empty

In the current first implementation, `LocalSpace` is used for both:

- physical on-site spaces
- MPO auxiliary / virtual bond spaces

## `LocalOperator`

`uni20::LocalOperator`
is conceptually the first realization of

`Tensor<co<LocalSpace>, QNum, LocalSpace>`

but the current implementation is a dedicated class rather than a generic tensor
instantiation.

It stores:

- `bra_space`
- `ket_space`
- `transforms_as()`
- a sparse coefficient matrix over explicit local states

The current coefficient storage is `SparseMatrix<double>`.

## `OperatorComponent`

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
predicate on its virtual indices. For rectangular components, the current
convention is that entries strictly below the main diagonal are forbidden.

## `FiniteTriangularMPO`

`uni20::FiniteTriangularMPO` is the first lattice-level MPO container.

It stores a finite sequence of `OperatorComponent` sites and currently enforces:

- one shared symmetry across the chain
- exact matching of adjacent virtual spaces
- upper-triangular virtual structure at every site

This is intentionally much narrower than a full MPO class hierarchy.

## Current API

`LocalSpace` currently supports:

- construction from `Symmetry`, `QNum`, or `QNumList`
- `symmetry()`, `size()`, `empty()`
- `push_back()`, `clear()`
- `contains()`
- indexed access and iteration
- `qnums()`

`LocalOperator` currently supports:

- construction from `bra_space`, `ket_space`, and `transforms_as`
- optional construction from an existing sparse coefficient matrix
- `bra_space()`, `ket_space()`, `symmetry()`
- `transforms_as()`
- `rows()`, `cols()`, `nnz()`
- sparse coefficient mutation and lookup via `insert_or_assign()`, `erase()`,
  `contains()`, `at()`, and `clear()`

`OperatorComponent` currently supports:

- construction from local spaces plus left/right virtual `LocalSpace`s
- optional construction from an existing sparse matrix of `LocalOperator`
- `local_bra_space()`, `local_ket_space()`
- `left_virtual_space()`, `right_virtual_space()`, `symmetry()`
- `rows()`, `cols()`, `nnz()`
- sparse entry mutation and lookup via `insert_or_assign()`, `erase()`,
  `contains()`, `at()`, and `clear()`
- `is_upper_triangular(component)`

`FiniteTriangularMPO` currently supports:

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
