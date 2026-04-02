# TensorContraction Integration Note

This is a temporary working note for integrating
`../TensorContraction` into Uni20 for a minimal U(1) DMRG implementation.

The current goal is not to make TensorContraction a generic Uni20 backend.
Instead:

- Uni20 owns the DMRG driver, basis bookkeeping, symmetry-aware metadata, and
  host-side tensor manipulations.
- TensorContraction remains an opaque specialized engine for the effective
  Hamiltonian matrix-vector multiply and a small amount of related linear
  algebra.

## Scope

The near-term target is:

- real-valued U(1) DMRG
- block-sparse bookkeeping on the Uni20 side
- TensorContraction for the effective-Hamiltonian MVP
- CPU-side decompositions such as SVD

This note is about the driver boundary, not the final long-term tensor design.

## Current TensorContraction Surface

The current branch in `../TensorContraction` now appears sufficient for a first
iterative eigensolver loop.

Useful entry points are in:

- `../TensorContraction/cpp/include/Arranger.hpp`
- `../TensorContraction/cpp/include/Matrix.hpp`
- `../TensorContraction/cpp/include/MatrixAllocator.hpp`
- `../TensorContraction/cpp/include/Swapper.hpp`

Important `Arranger` operations:

- `doContraction(...)`
- `compileInnerProductForLinearAlgebra(...)`
- `compileAddAccuForLinearAlgebra(...)`
- `compileScalarMulForLinearAlgebra(...)`
- `doLinearAlgebra()`
- `collectiveExchangeMatrix(...)`
- `broadcastRtoB(...)`

This is enough to support:

- `y = Hx`
- `dot(x, y)`
- `y += alpha x`
- `x *= alpha`
- `norm(x)` computed as `sqrt(dot(x, x))` on the Uni20 side
- explicit readback to CPU memory for SVD or other host-side operations

Recent TensorContraction changes that matter:

- `Matrix` is now a `shared_ptr`-backed pImpl descriptor
- `R` and intermediate matrices are explicitly allocated on CPU memory
- `collectiveExchangeMatrix(...)` now handles collective readback from local CPU,
  local GPU, or a remote rank
- `broadcastRtoB(...)` was added for iterative DMRG execution
- `example.cu` now demonstrates the full iterative contraction / linear-algebra /
  broadcast / readback workflow that Uni20 should wrap

## Conceptual Split

### Uni20 side

Uni20 should own:

- `Symmetry`, `QNum`, `QNumList`, and future `BlockSpace`
- sparse-space / block-space metadata
- host-side sparse MPO representation
- compilation of MPO plus basis labels into TensorContraction term lists
- eigensolver iteration logic
- sweep logic, truncation policy, and environment updates
- CPU-side SVD and reshape logic

### TensorContraction side

TensorContraction should own:

- distributed block descriptors
- GPU/CPU placement and MPI/NCCL movement
- specialized contraction execution
- the small LA worklists it already implements

Uni20 should not try to schedule inside TensorContraction for the first pass.

## Core Data Model

The intended Uni20-side picture is:

- `QNum`
  - one-dimensional irreducible tensor space
- `QNumList`
  - explicit sparse tensor space
- `BlockSpace`
  - future coalesced `(QNum, dim)` tensor space
- `conjugate<T>` / `co<T>`
  - tensor-leg direction

For TensorContraction integration, the important representation is still the
block family expected by TensorContraction:

- logical tensor or tensor slice
  - represented as a `std::vector<tensor::Matrix>`

But Uni20 should not expose raw `std::vector<tensor::Matrix>` directly.

## Proposed Wrapper Boundary

The first useful adapter type is an opaque wrapper around TensorContraction
matrix blocks. The old notes referred to this loosely as `MatrixList`; that is
still the right idea.

Responsibilities of this wrapper:

- own the storage that `tensor::Matrix::ptr` refers to
- preserve block ordering and tensor-space metadata
- expose the underlying `std::vector<tensor::Matrix>` only to the adapter layer
- support explicit conversion to and from host-side Uni20 tensor objects

The wrapper should separate:

- logical block identity and metadata
- transient host/device materialization managed by TensorContraction

For the eigensolver phase, TensorContraction storage should be the canonical
representation. Conversion back to host-side Uni20 objects should happen only at
explicit boundaries such as SVD.

## Effective Hamiltonian Compilation

TensorContraction does not want a symbolic MPO. It wants:

- `rMats`
- `aMats`
- `bMats`
- `cMats`
- `FTerms`

where each `FTerm` is a sparse coefficient entry:

- `(r_idx, a_idx, b_idx, c_idx, coeff)`

So Uni20 needs a host-side compiler stage:

1. start from a sparse MPO representation
2. combine it with basis labels and block metadata
3. enumerate all allowed block couplings
4. produce block families plus `FTerms`
5. cache that compiled effective-Hamiltonian plan for repeated eigensolver use

For U(1), the coefficient is just the MPO coefficient after basis selection.
Later non-Abelian support will multiply in recoupling data.

## Minimal Driver API

The first DMRG-facing Uni20 API should be very small.

Suggested operations on the execution-side vector object:

- `apply_hamiltonian(plan, x, y)`
- `dot(x, y)`
- `axpy(alpha, x, y)`
- `scal(alpha, x)`
- `norm(x)`
- `copy_to_host(x)`

This is enough for a first Davidson or Lanczos-style solver.

The outer DMRG code should work with these opaque objects rather than with the
raw TensorContraction types.

## First-Pass Execution Flow

The current TensorContraction flow, as shown in `../TensorContraction/cpp/src/example.cu`,
is roughly:

1. build matrix descriptors on rank 0
2. distribute matrix ownership across MPI ranks
3. distribute `FTerms` by `R`
4. analyze computation
5. allocate matrices
6. pre-copy A/B/C inputs
7. compile contraction worklists
8. execute contraction
9. run linear-algebra worklists as needed
10. broadcast `R` back into `B` when iterative execution needs it
11. collectively exchange selected matrices back to rank 0 when host-side code
    needs them

Uni20 should treat this as one opaque execution pipeline.

## Immediate Implementation Order

### 1. Adapter layer

Implement a small Uni20 wrapper around TensorContraction matrices:

- create / destroy block families
- initialize from host data
- collectively exchange back to host data
- map to TensorContraction calls

### 2. Host-side MPO compiler

Implement a minimal sparse MPO representation and compile it to:

- block families
- `FTerms`

### 3. Minimal eigensolver interface

Wrap the available TensorContraction operations as:

- `Hx`
- `dot`
- `axpy`
- `scal`
- `norm`

### 4. CPU-side decomposition path

After eigensolver convergence:

- fetch the resulting blocks to host
- convert to Uni20 host tensor representation
- run SVD on CPU
- rebuild the next TensorContraction-facing vector/state as needed

## Things To Avoid In First Pass

- treating TensorContraction as a general Uni20 dense-tensor backend
- automatic coalescing of sparse physical or MPO spaces
- trying to schedule inside TensorContraction with Uni20 async machinery
- designing the final non-Abelian fusion/coupling layer before U(1) DMRG works

## Known Caveats

### Coefficient lifetime in LA helpers

`compileAddAccuForLinearAlgebra(...)` snapshots `*coff` at compile time.

`compileScalarMulForLinearAlgebra(...)` stores the pointer and dereferences it
later during execution.

So Uni20 should treat these coefficient arguments carefully and avoid assuming a
uniform lifetime model across the LA wrapper functions.

### Host fetch ownership

`collectiveExchangeMatrix(...)` may return:

- the matrix's existing host pointer
- or a freshly allocated buffer

On ranks that are not requesting a concrete matrix, the call may simply return
`nullptr`.

So the adapter must:

- call the exchange collectively and in lockstep across ranks
- track ownership on rank-0 readback
- free only when the returned pointer is not the matrix's own host pointer

### Placement policy is still TensorContraction-owned

The current TensorContraction distribution policy is still the existing
size-balanced / `R`-assigned policy. Uni20 should accept that for now and avoid
redesigning placement at the same time as the driver integration.

## Longer-Term Direction

### TensorContraction Convergence

In its current form, TensorContraction should not become a separate Uni20
backend. The `A/B/C/R + FTerms` representation is
not just an implementation detail around a generic backend; it is the new
algorithmic object that Uni20 wants to exploit, and implement as a key algorithm that back-ends will need to implement.

First, Uni20 should retain a specialized DMRG contraction path built around the
same kind of compiled block-family / coefficient-term representation, even if
the current API surface evolves:

- the `A/B/C/R + FTerms` view of the effective Hamiltonian
- the compilation of sparse operator and basis data into that form
- the DMRG-specific contraction strategy itself

Second, the useful execution-side ideas in TensorContraction should converge
into a more general Uni20 MPI+GPU backend:

- distributed block descriptors
- host / device residency management
- MPI / NCCL transport
- worklist compilation and execution scheduling
- collective broadcast and readback patterns

So the long-term goal is not to preserve the current TensorContraction API as a
public Uni20 backend surface. The goal is to absorb and generalize its runtime
and placement machinery until Uni20 has a native distributed GPU backend of its
own, while also preserving the DMRG-specific contraction ideas that motivated
the integration in the first place.

### Broader Uni20 Tensor / Symmetry Work

Separately, Uni20 will also grow the more general tensor and symmetry features
that are needed regardless of the TensorContraction adapter:

- tensor-backed MPS / MPO / effective-vector data structures rather than the
  current prototype U(1) `LocalSpace` / `OperatorComponent` layer
- first-class block-space objects
- more explicit sparse-space semantics
- non-Abelian coupling data
- braiding-aware tensor operations

But none of that is required to get the first U(1) DMRG path working.
