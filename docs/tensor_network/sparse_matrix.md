# TensorContraction Integration Sparse-Matrix Layer

**Status:** functional integration-branch reference. `uni20::SparseMatrix<T>` and the
matrix/operator source directories described below are not present on `main`.
This note records the representation used by the working operator and MPO
implementation and informs the pure-Uni20 replacement.

## Scope

This is intentionally a narrow data-structure layer:

- It is not a generic sparse tensor library.
- It is not the dense linear-algebra layer under `src/uni20/linalg/`.
- Its first intended uses are local operators and MPOs.

The container stores each row as a sorted list of `(column, value)` entries.
That keeps mutation simple while preserving the access pattern needed for
iterating over sparse operator rows.

## Integration-Branch API

The prototype `SparseMatrix<T>` supported:

- construction from `(rows, cols)`
- shape queries: `rows()`, `cols()`, `shape()`, `nnz()`
- row inspection: `row_size(i)`, `row(i)`
- mutation: `insert_or_assign(i, j, value)`, `erase(i, j)`, `clear_row(i)`, `clear()`
- lookup: `contains(i, j)`, `find(i, j)`, `at(i, j)`
- `transpose()`

Basic sparse algebra was provided in
`src/uni20/matrix/sparse_matrix_ops.hpp` on that branch:

- `add(lhs, rhs)`
- `scale(matrix, scalar)`
- `multiply(lhs, rhs)`
- `kronecker(lhs, rhs)` / `kron(lhs, rhs)`

Each of these also has a variant that accepts custom functors for the nested
value operation. This is intended for future MPO algebra where the matrix entry
type is itself an operator-like object.

Rows are kept sorted by column index. There is no implicit zero-culling rule,
because future values such as `LocalOperator` are not naturally compared to a
distinguished scalar zero.

## Reusable Direction

The intended use was:

- `LocalOperator` will use a sparse matrix of scalar coefficients.
- `MPO` will use a sparse matrix of `LocalOperator`.

If later algorithms need a more compact compressed format or a transpose cache,
those can be added on top of this API without changing the higher-level operator
abstractions.
