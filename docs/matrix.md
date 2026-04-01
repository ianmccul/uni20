# Sparse Matrices

Uni20 currently provides a small row-oriented sparse matrix container in
`uni20::SparseMatrix<T>`.

## Scope

This is intentionally a narrow data-structure layer:

- It is not a generic sparse tensor library.
- It is not the dense linear-algebra layer under `src/uni20/linalg/`.
- Its first intended uses are local operators and MPOs.

The container stores each row as a sorted list of `(column, value)` entries.
That keeps mutation simple while preserving the access pattern needed for
iterating over sparse operator rows.

## Current API

`SparseMatrix<T>` currently supports:

- construction from `(rows, cols)`
- shape queries: `rows()`, `cols()`, `shape()`, `nnz()`
- row inspection: `row_size(i)`, `row(i)`
- mutation: `insert_or_assign(i, j, value)`, `erase(i, j)`, `clear_row(i)`, `clear()`
- lookup: `contains(i, j)`, `find(i, j)`, `at(i, j)`
- `transpose()`

Basic sparse algebra is provided in `src/uni20/matrix/sparse_matrix_ops.hpp`:

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

## Planned Use

The current plan is:

- `LocalOperator` will use a sparse matrix of scalar coefficients.
- `MPO` will use a sparse matrix of `LocalOperator`.

If later algorithms need a more compact compressed format or a transpose cache,
those can be added on top of this API without changing the higher-level operator
abstractions.
