# Mdspan BLAS/LAPACK Wrapper Layer Plan

**Status:** implemented BLAS and initial LAPACK checkpoints plus forward plan.
This layer is below tensor front-end dispatch and above the existing raw
BLAS/LAPACK provider facades. Direct mdspan GEMM/GEMV, accessor-respecting CPU
fallbacks, explicit Tensor copy/materialization, and strict LAPACK projected
eigensystem/Schur plus dense self-adjoint adapters now use the operation-tag
dispatcher.

The implemented mdspan-facing wrappers accept resolved views, build
provider-compatible descriptors, and call the existing Uni20 BLAS/LAPACK
backend wrappers. Future work extends this adapter layer operation by operation;
it does not replace the backend facades.

Related notes:

- [`mdspan_linalg_dispatch_plan.md`](mdspan_linalg_dispatch_plan.md) - umbrella
  plan for mdspan linalg and later operation-tag dispatch.
- [`backend_dispatch.md`](backend_dispatch.md) - compile-time capability,
  runtime attempt, and fallback contract.
- [`kernel_dispatch.md`](kernel_dispatch.md) - operation tags and backend-list
  walk.
- [`scalar_policy.md`](scalar_policy.md) - BLAS/LAPACK scalar concept policy.

## Layer Decision

Keep the existing wrapper layers and add one new layer above them:

```text
Tensor / future linalg front end
  -> resolved mdspan-like views
  -> mdspan BLAS/LAPACK wrapper layer        (new)
  -> src/uni20/backend/blas and backend/lapack wrappers
  -> provider library: BLAS, LAPACK, MPLAPACK, MKL, OpenBLAS, ...
```

The existing backend wrappers remain provider and ABI facades:

- `src/uni20/backend/blas`: BLAS discovery, vendor selection, LP64/ILP64
  configuration, raw BLAS declarations, and vendor extension targets.
- `src/uni20/backend/lapack`: checked and unchecked LAPACK overloads over the
  configured provider.

The new mdspan layer owns view interpretation:

- rank, scalar, mutability, and host-addressability concepts;
- conversion from mdspan extents/strides to BLAS matrix descriptors;
- transform composition and BLAS transpose character selection;
- direct no-copy runtime acceptance or decline;
- optional prepared input temporaries in wrappers whose contract allows them;
- operation-specific rewrites such as row-major GEMM output handling.

Do not move shape/materialization policy into `src/uni20/backend`. That layer
should stay close to provider ABI calls. The mdspan layer may call
`uni20::blas::*` and `uni20::lapack::*`, but it should not redeclare Fortran
symbols or include provider-specific headers directly.

## Initial Files

The direct BLAS descriptor implementation lives under `src/uni20/linalg/blas/`:

- `blas_matrix.hpp`
  - provider-ready `BlasReadableMatrix` and `BlasWritableMatrix` operands,
    backend-independent transform algebra, and BLAS transpose lowering.
- `blas_vector.hpp`
  - provider-ready `BlasReadableVector` and `BlasWritableVector` operands.
- `mdspan_access.hpp`
  - direct BLAS accessor eligibility shared across operand ranks.
- `mdspan_matrix.hpp`
  - `try_mdspan_matrix_stage(...)`, role-specific readable/writable lowering
    helpers, and strict
    column-major-compatible helpers for LAPACK update operands.
- `mdspan_vector.hpp`
  - `try_mdspan_vector_stage(...)`, BLAS increment lowering, and readable or
    writable rank-one descriptors, including strict contiguous writable LAPACK
    vectors.
- `gemm.hpp`
  - direct mdspan GEMM wrappers.
- `gemv.hpp`
  - direct mdspan GEMV wrappers.
- `blas.hpp`
  - include point for the adapter layer.

The first operation-tag dispatch slice adds:

- `operation_tags.hpp`
  - the central catalogue of operation values and their diagnostic names.
- `dispatch.hpp`
  - `KernelTypeAcceptance`, `backend_list`, selector normalization, type
    acceptance detection, and the runtime backend walk.
- `ops/gemm.hpp`
  - fixed-output Tensor `gemm(...)`; bare mdspans use the generic dispatch API
    directly.
- `ops/gemv.hpp`
  - fixed-output Tensor `gemv(...)` with rank-1 output/input and a rank-2 matrix.
- `backends/blas/gemm.hpp`
  - `BlasBackend` and `try_kernel(BlasBackend, gemm_op, ...)`.
- `backends/blas/gemv.hpp`
  - `BlasBackend` and `try_kernel(BlasBackend, gemv_op, ...)`.
- `backends/cpu/gemm.hpp`
  - `CpuReferenceBackend` and the accessor-respecting fallback GEMM oracle.
- `backends/cpu/gemv.hpp`
  - `CpuReferenceBackend` and the accessor-respecting fallback GEMV oracle.

LAPACK operation adapters live under `src/uni20/linalg/backends/lapack/`;
currently this includes tridiagonal and nonsymmetric eigensystems, Schur and
Hessenberg Schur decomposition, Schur reordering, and real symmetric/complex
Hermitian `syev`/`heev`. Shared provider-descriptor construction remains under
`src/uni20/linalg/blas/`. Keep that division so descriptor helpers remain
testable without running a LAPACK operation.

## Two Descriptor Levels

Keep two descriptor levels separate:

1. Minimal provider-ready BLAS matrix/vector operands. Matrix operands describe
   the column-major matrix that the BLAS/LAPACK ABI sees, plus any
   readable-input transform. Vector operands carry a logical size and BLAS
   increment.
2. Mdspan staging descriptors. These describe logical mdspan extents, strides,
   and accessor conjugation before deciding whether direct lowering, a
   transpose rewrite, or a temporary is needed.

Do not merge these roles. The first one is an ABI object; the second one is a
planning object.

### Provider-Ready BLAS Operands

`BlasWritableMatrix` is the small object that can be passed to wrappers around
`uni20::blas::*` and `uni20::lapack::*` for an output/update matrix. Its `rows`
and `cols` are the matrix dimensions before applying any BLAS transpose
character. They are not necessarily the logical mdspan dimensions.

`BlasReadableMatrix` has the same fields plus a `MatrixTransform`. This is the
provider-ready input operand for BLAS calls.

```cpp
enum class MatrixTransform : unsigned {
  normal = 0,
  transpose = 1,
  conjugate = 2,
  conjugate_transpose = 3
};

template <class Scalar, class Handle = Scalar*>
struct BlasWritableMatrix {
  Handle data;
  blas_int rows;
  blas_int cols;
  blas_int leading_dimension;
};

template <class Scalar, class Handle = Scalar const*>
struct BlasReadableMatrix {
  Handle data;
  blas_int rows;
  blas_int cols;
  blas_int leading_dimension;
  MatrixTransform transform = MatrixTransform::normal;
};
```

Writable BLAS outputs do not carry a transform. If an output mdspan is
row-major-like, the operation must be rewritten so the provider writes the
untransformed `BlasWritableMatrix`, or the wrapper must use an explicit output
temporary with copy-back policy.

Do not use inheritance or a common contained base for these two objects. The
duplicated fields are intentional: the objects are small, and implicit
conversion from a readable object to a common base would lose the transform.

`BlasReadableVector` and `BlasWritableVector` similarly duplicate `data`,
`size`, and `increment`. The readable vector has no transform field because
portable GEMV cannot conjugate `x`; `MdspanVectorStage::needs_conjugation`
therefore causes a direct GEMV decline instead of being discarded.

### Mdspan Staging Descriptor

The staging descriptor keeps the logical extents in mdspan axis order and
records which mdspan axis has stride `1`. The non-unit stride is canonicalized
for extent-1 axes so it also satisfies the BLAS leading-dimension rule. It does
not store BLAS transposition state.

This chooses the mdspan-storage-first policy rather than a BLAS-first policy.
The alternative would store BLAS `rows`, `cols`, `leading_dimension`, and a
transpose flag. That is too easy to make ambiguous: once a view is marked as
transposed, it is unclear at the staging boundary whether `rows` and `cols`
describe the logical matrix or the matrix seen by the BLAS ABI. In this plan,
the staging descriptor always describes logical mdspan axes; helper functions
derive the BLAS ABI reference.

```cpp
template <class Scalar, class Handle>
struct MdspanMatrixStage {
  Handle data;
  blas_int extent0;
  blas_int extent1;
  blas_int nonunit_stride;
  int unit_stride_axis; // 0 if stride(0) == 1, 1 if stride(1) == 1
  bool needs_conjugation = false;
};
```

There is only one staging descriptor. `needs_conjugation` defaults to `false`
and records that the logical matrix is conjugated relative to the raw storage
interpretation. For readable lowering, it composes into
`BlasReadableMatrix::transform`. For writable lowering, it is not represented in
`BlasWritableMatrix` itself. The current direct GEMM wrapper rejects such a
stage as a user-provided output view. This is separate from a prepared
operation internally conjugating the output storage before and after a provider
call as an implementation trick.

The staging helper derives `needs_conjugation` from the mdspan accessor policy.
It is not an independent option supplied by the BLAS adapter. The generic
mdspan `conj(...)` helper in `src/uni20/mdspan/conjugate_accessor.hpp` returns a
read-only conjugating accessor view for complex mdspans, cancels to the const
original view when applied twice, and returns a const identity view for
non-complex mdspans. The same header defines
`accessor_applies_conjugation_v<Accessor>` and
`mdspan_needs_conjugation_v<View>`. The implemented `conj(Tensor)` operation
returns a tensor view whose mdspan uses that conjugation accessor rather than
eagerly materializing conjugated storage. When that view reaches the BLAS
staging helper, `try_mdspan_matrix_stage(...)` inspects the accessor and sets
`needs_conjugation = true`.

The conjugating accessor follows the C++26 `std::linalg::conjugated_accessor`
direction in WG21 P3050R3. Uni20 deliberately keeps `uni20::conj(span)` as the
user-facing operation and does not adopt `conj-if-needed` terminology: for
non-complex values, `uni20::conj` is the identity on values and the mdspan
helper returns a read-only identity view.

The direct BLAS path should accept only accessors that can still expose the raw
storage handle plus declarative metadata such as `needs_conjugation`. An
accessor that computes arbitrary transformed values by value is a generic
transform view, not directly BLAS-addressable; it needs materialization or a
non-BLAS fallback. This follows the mdspan accessor model: mapping and accessor
describe how logical elements are read from the same handle, and the BLAS
adapter may lower only the subset it understands. The broader view policy lives
in `docs/tensor_dispatch_and_view_semantics_draft.md`: structural views such as
`real(x)` and `imag(x)` are slice-like address transformations, while semantic
transform views such as `conj(x)` are read-only by default.

A pointer `data_handle_type` is not enough to prove direct BLAS readability or
writeability. Direct BLAS/LAPACK wrappers may bypass `access(...)`; that is only
correct for `stdex::default_accessor<T>` or for accessors whose semantics the
wrapper explicitly lowers. For the current BLAS adapter, readable inputs may use
`stdex::default_accessor<T>` or Uni20's `conjugated_accessor` over a default
accessor. Writable outputs and LAPACK update operands require default-accessor
views.

### Aliasing Contract

Fixed-output GEMM requires the output element addresses to be disjoint from
both readable inputs. Fixed-output GEMV similarly requires the output vector to
be disjoint from the matrix and input vector. Readable operands may overlap each
other. These are leaf-kernel preconditions: supporting an aliased output
requires an explicit temporary and copy-back policy above the direct wrapper.

Do not reject operands merely because their enclosing allocation or bounding
address ranges overlap. Two slices of one matrix may still address disjoint
elements. A directly BLAS-compatible matrix is an ordered union of contiguous
unit-stride intervals, one per provider column, so a diagnostic or binding layer
can test exact overlap in linear time in the number of provider columns. Generic
accessor-respecting CPU views do not necessarily admit an equally cheap exact
test.

The C++ leaf API may rely on this documented precondition. Python and other
runtime-erased front ends should validate overlap whenever it is exactly
representable and should reject or materialize uncertain cases according to
their user-facing policy.

`unit_stride_axis` is deliberately an integer mdspan axis, not a row/column
enum. Names such as `row` and `column` are easy to misread because
`stride(0) == 1` is the column-major-compatible case, while `stride(1) == 1`
is the row-major-like case. The descriptor stores the facts about the view and
leaves interpretation to helper functions.

For a logical matrix `A(i, j)`:

- `extent0` is the logical row extent.
- `extent1` is the logical column extent.
- `unit_stride_axis == 0` means `stride(0) == 1` and
  `nonunit_stride == stride(1)`. The BLAS ABI sees the logical matrix directly.
- `unit_stride_axis == 1` means `stride(1) == 1` and
  `nonunit_stride == stride(0)`. The BLAS ABI sees the logical transpose.

When the corresponding extent is `1`, the non-unit stride is not observable in
the logical mdspan mapping. The staging helper may therefore choose a canonical
representative that satisfies BLAS, rather than preserving the arbitrary value
reported by the mapping.

Runtime descriptor construction should reject views where neither matrix stride
is `1`. The first pass should also reject negative strides for direct
BLAS/LAPACK calls. Any extent or stride that cannot be represented by the
configured signed `blas_int` ABI type must be rejected by the direct wrapper;
larger logical problems require an ILP64 provider or a higher-level blocked
operation.

For direct BLAS lowering, `MdspanMatrixStage::nonunit_stride` must satisfy the
provider ABI lower bound:

- `unit_stride_axis == 0`: `nonunit_stride >= max(1, extent0)`;
- `unit_stride_axis == 1`: `nonunit_stride >= max(1, extent1)`.

Degenerate dimensions need explicit tests because a non-unit stride may be
arbitrary when it corresponds to an extent-1 provider column axis. If the
provider column count is at most one, normalize `nonunit_stride` upward to
`max(1, provider_rows)` instead of declining the view. If the provider would
actually step through the non-unit stride, do not repair it by changing the
address arithmetic; reject the view unless the stride already satisfies the
provider contract.

Provider-ready BLAS operands are still derived from the mdspan staging
descriptor:

```cpp
template <class Scalar, class Handle>
BlasWritableMatrix<Scalar, Handle>
blas_writable_matrix(MdspanMatrixStage<Scalar, Handle> const& stage);

template <class Scalar, class Handle>
BlasReadableMatrix<Scalar, Handle>
blas_readable_matrix(MdspanMatrixStage<Scalar, Handle> const& stage);
```

These helpers return provider-ready objects with `rows = extent0`,
`cols = extent1`, and `leading_dimension = nonunit_stride` when
`unit_stride_axis == 0`; they swap the extents when `unit_stride_axis == 1`.
For readable operands, the returned `transform` is the effective transform after
composing the storage transform and `stage.needs_conjugation`. Callers should
express logical conjugation by passing `uni20::conj(span)` and should express
logical transpose by passing a transposed mdspan view. For writable operands,
`blas_writable_matrix(...)` returns the provider-writeable storage shape only. A
stage with `needs_conjugation == true` is not accepted by the current direct
GEMM wrappers: `try_gemm(...)` declines it and the checked `gemm(...)` wrapper
aborts. In normal use this is unreachable because `uni20::conj(mdspan)` returns
a const readable view, but the runtime guard keeps custom accessors honest. This
keeps the staging descriptor unambiguously mdspan-like while making the BLAS ABI
conversion local and testable.

## Transform Helpers

Represent transforms independently of the backend-specific character, using the
`MatrixTransform` enum above.

The transform helper layer should provide:

- `storage_transform(MdspanMatrixStage const&)`;
- `transpose_result_transform(MatrixTransform)`;
- `compose(MatrixTransform, MatrixTransform)`;
- `logical_rows(...)`, `logical_cols(...)`, `abi_rows(...)`, `abi_cols(...)`;
- `blas_trans_char_is_supported(MatrixTransform) -> bool`;
- `blas_trans_char(MatrixTransform) -> char`.

For readable operands, compose the storage transform and
`stage.needs_conjugation`. Operation-specific rewrites, such as row-major GEMM
output normalization, may compose additional internal transforms before the
provider call. User-facing mdspan wrappers should not take `MatrixTransform`
parameters; transform intent comes from the mdspan view/accessor itself. The
transform spelling helper maps the internal transform alphabet to `'N'`, `'T'`,
`'C'`, and `'R'`, where `'R'` means conjugate without transpose. The current
generic direct GEMM wrapper still gates provider calls to the portable `'N'`,
`'T'`, and `'C'` subset, so complex conjugate-only states are declined before
side effects. OpenBLAS documents the extension in its CBLAS enum as
`CblasConjNoTrans`, and its `interface/gemm.c` dispatcher also recognizes the
Fortran-style `'R'` spelling on the develop branch:
<https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gemm.c>. Intel
MKL's Fortran GEMM rejected `'R'` in local testing, so any direct `'R'` fast
path should be explicitly provider-gated rather than assumed as baseline BLAS.
Otherwise a prepared wrapper must materialize the conjugated readable operand
and dispatch through the ordinary `N/T/C` path.

`storage_transform(view)` returns `normal` when `unit_stride_axis == 0` and
`transpose` when `unit_stride_axis == 1`. The helper name should describe the
derived BLAS interpretation; the stored descriptor remains mdspan-axis based.
There should be no stored `transposed` field.

If `L` is the logical mdspan matrix and `S` is the untransformed
`BlasWritableMatrix` shape derived from the same stage, then
`L = storage_transform(stage)(S)`. The readable provider operand is lowered as:

```text
effective_blas_transform = conjugation_from_accessor o storage_transform(stage)
```

where `o` means function composition. For example, a row-major-like input stage
has `storage_transform == transpose`, so it lowers to BLAS `'T'`. If the same
view is wrapped by `uni20::conj(...)`, the accessor contributes conjugation and
the effective transform becomes conjugate-transpose. A caller that wants a
logical transpose should pass a structural transposed mdspan view rather than a
separate transform flag.

## Row-Major Writable GEMM

Writable outputs cannot carry a BLAS transpose flag. A row-major-like writable
matrix therefore needs an operation rewrite, not a flag on the output.

For GEMM:

```text
C = alpha * op(A) * op(B) + beta * C
```

if `C` has `unit_stride_axis == 1`, rewrite to:

```text
C^T = alpha * op(B)^T * op(A)^T + beta * C^T
```

The helper should swap readable operands and apply an internal transpose
transform to both staged readable operands. This keeps the writable output
untransformed while still describing the logical result.

This rewrite is appropriate for BLAS update-style operations. It is not a
general rule for LAPACK drivers that overwrite inputs with structured outputs.

## Writable Output Conjugation

`BlasWritableMatrix` intentionally has no transform field. It is only the
provider-writeable storage target. Operation-specific wrappers normalize any
output-side transform before dispatch.

There are two different notions that should stay separate:

- A conjugating output view supplied by the caller. The current direct mdspan
  GEMM layer does not accept this. Logical conjugation is represented by
  `uni20::conj(mdspan)`, and that view is read-only.
- Manual conjugation of the output storage by an implementation-owned prepared
  fallback. This can be a legitimate way to rewrite an otherwise unsupported
  combination of readable BLAS transforms into the ordinary `N`/`T`/`C`
  provider character set. It is not user-visible and does not require a
  `conjugate_output` flag on the direct wrapper.

For example, after composing row-major storage, input view conjugation, and
operation-specific normalization, the best lowering may require a
conjugate-only readable transform that the baseline provider cannot encode. A
future prepared GEMM fallback may:

1. conjugate the writable output storage in place before the BLAS call when
   `beta != 0`, so the accumulated old-output term represents `conj(C_old)`;
2. conjugate `alpha`, `beta`, and all readable operands logically, which flips
   each readable transform's conjugation bit and may turn the unsupported
   conjugate-only case into an ordinary `N`/`T`/`C` provider call;
3. call BLAS into the same writable storage using the representable transforms;
4. conjugate the output storage in place after the BLAS call to restore the
   caller-visible result.

When `beta == 0`, the pre-call output conjugation can be skipped because the old
output is not read. This fallback requires an explicit matrix-conjugation
primitive and must only be used when the transformed readable operands are all
representable by the selected provider. If conjugating everything merely moves
an unsupported conjugate-only transform from one input to another, the prepared
wrapper still needs input materialization or must decline.

Direct no-copy wrappers should not silently materialize output temporaries. If a
future backend cannot implement the required provider transform directly, that
backend-specific wrapper should either expose a clear fallback contract or
decline before side effects.

## Worked GEMM Example

Consider the logical operation:

```text
C(4, 5) = alpha * A(4, 3) * B(3, 5) + beta * C(4, 5)
```

Assume `A` and `B` are column-major-compatible and `C` is row-major-like:

```text
A: extent0 = 4, extent1 = 3, unit_stride_axis = 0, nonunit_stride = 4
B: extent0 = 3, extent1 = 5, unit_stride_axis = 0, nonunit_stride = 3
C: extent0 = 4, extent1 = 5, unit_stride_axis = 1, nonunit_stride = 5
```

The mdspan staging descriptors describe the logical operands. Because `C` is
row-major-like, the wrapper first chooses the GEMM rewrite:

```text
C^T = alpha * B^T * A^T + beta * C^T
```

It then derives provider-ready operands for the rewritten call:

```text
B_read: rows = 3, cols = 5, leading_dimension = 3, transform = transpose
A_read: rows = 4, cols = 3, leading_dimension = 4, transform = transpose
C_write: rows = 5, cols = 4, leading_dimension = 5
```

`C_write` is the untransformed writable matrix seen by BLAS, so it represents
logical `C^T`.

The provider call is therefore:

```text
gemm(transa = 'T', transb = 'T',
     m = 5, n = 4, k = 3,
     A = B_read.data, lda = 3,
     B = A_read.data, ldb = 4,
     C = C_write.data, ldc = 5)
```

The intermediate staging layer is what made this rewrite possible: it knew that
the output was row-major-like before constructing the final BLAS call. The
final `BlasReadableMatrix` and `BlasWritableMatrix` operands alone are not
enough to decide whether to swap operands, apply output-normalization
transforms, or allocate temporaries.

If `A` were row-major-like instead, its stage would derive a
`BlasReadableMatrix` whose `rows` and `cols` describe the provider's
untransformed `A^T` storage, and whose `transform` is `'T'` for a logical
input. If composing storage layout, accessor conjugation, and any internal
operation rewrite produces `conjugate`, the generic direct wrapper declines.
Provider-specific wrappers may add a fast path for OpenBLAS-style
conjugate-no-transpose support later.

## Direct And Prepared Wrappers

There are two wrapper contracts:

1. **Direct operand wrappers** do not materialize operands and do not copy as a
   fallback. They build descriptors, call the provider wrapper if the instance
   is representable, or decline before side effects.
2. **Prepared wrappers** may materialize input-only operands into explicit
   scratch storage when their public contract says so.

Output/update temporaries are stricter. They require explicit copy-back policy,
aliasing checks before side effects, and diagnostics that report the
materialization.

LAPACK workspace is not operand materialization. Many LAPACK operations need
work arrays whose size is chosen by a workspace query or by the wrapper's
algorithm policy. A direct LAPACK operation wrapper may allocate and free those
work arrays internally while still being a direct no-copy wrapper for its
matrix/vector operands. The distinction is:

- **Workspace allocation**: temporary storage that LAPACK uses internally and
  whose contents are not part of any user-visible operand. This is allowed in a
  direct LAPACK operation wrapper unless the API explicitly exposes
  caller-managed workspace.
- **Operand materialization**: copying, conjugating, transposing, or packing a
  user-visible input/output/update operand into scratch storage. This is a
  prepared-wrapper behavior and must be named or documented explicitly.

For example, a direct `self_adjoint_eigh` wrapper may query and allocate
`work`/`rwork` arrays before calling `syev` or `heev`. It may not silently copy a
row-major matrix into a column-major temporary and copy eigenvectors back unless
the wrapper contract says that operand materialization is allowed.

Example prepared input shape:

```cpp
template <class Scalar>
struct PreparedBlasReadableMatrix {
  Matrix<Scalar> scratch;
  BlasReadableMatrix<Scalar> readable;
};
```

The prepared wrapper points `readable.data` at either the original storage or
scratch storage.

## Relation To Dispatch

The direct GEMM and GEMV wrappers are wired into the generic backend-list
dispatcher as leaf kernels. Bare mdspans call `dispatch_kernel` or
`try_dispatch_kernel` directly with the operation tag; fixed-output Tensor
overloads derive the default selector from tensor storage. The remaining axes
of progress are:

1. Add more direct BLAS/LAPACK operation wrappers over the same mdspan
   descriptors.
2. Add higher-level allocating or shape-changing tensor operations above the
   fixed-output GEMM interface.

The implemented GEMM/GEMV slices prove the dispatcher, backend selector, type
acceptance CPO, runtime decline path, matrix/vector descriptors, and CPU
fallback shape without mixing in LAPACK workspace or overwrite semantics.
Bare mdspans use explicit selectors such as
`backend_list{BlasBackend{}, CpuReferenceBackend{}}`; storage-derived default
backend lists are implemented in the tensor front end.

The direct wrapper layer still exposes functions that are easy to test:

```cpp
KernelAttempt try_gemm(... mdspan-like operands ...);
void gemm(... mdspan-like operands ...);

KernelAttempt try_gemv(... mdspan-like operands ...);
void gemv(... mdspan-like operands ...);

KernelAttempt try_lapack_self_adjoint_eigh(... mdspan-like operands ...);
void lapack_self_adjoint_eigh_or_throw(... mdspan-like operands ...);
```

`try_gemm(...)` and `try_gemv(...)` return `unsupported_layout` when an operand
cannot be staged as a direct no-copy BLAS view and `unsupported_transform` when
the provider cannot express the required transform. GEMV therefore declines a
conjugating input vector, while a conjugating row-major matrix can lower to
conjugate-transpose. Once staging succeeds, dimension consistency is a checked
precondition: invalid operation parameters are logic errors and should abort
through `CHECK`/`CHECK_EQUAL`, not report ordinary kernel non-acceptance.

Operation-tag dispatch wraps the direct GEMM leaf like this:

```cpp
KernelAttempt try_kernel(BlasBackend, gemm_op, C&& c, Scalar alpha, A&& a,
                          B&& b, Scalar beta);
KernelAttempt try_kernel(BlasBackend, gemv_op, Y&& y, Scalar alpha, A&& a,
                          X&& x, Scalar beta);
KernelAttempt try_kernel(LapackBackend, self_adjoint_eigh_op, W&& w,
                          A&& matrix_work);
```

`kernel_accepts_types(...)` remains type-level: scalar family, rank, mutability,
and host raw-addressability. Descriptor construction remains runtime:
dynamic extents, strides, pointer values, and aliasing.

For backends that provide this static CPO, it is the authoritative type filter.
The corresponding `try_kernel(...)` should not repeat the same long `requires`
clause; it may use a local `static_assert` against the shared support predicate
to catch accidental direct calls with rejected types.

For GEMM and GEMV, `try_kernel(BlasBackend, operation, ...)` is a thin adapter
over the corresponding direct wrapper. The operation-tag layer does not
duplicate mdspan stride analysis. If the direct wrapper declines because an
instance cannot be represented as a no-copy BLAS call, the dispatch walk
continues to the next backend, normally the CPU reference oracle.

## Implementation Order

Completed checkpoint:

1. Add concept and descriptor helper tests.
2. Implement `try_mdspan_matrix_stage(...)` and role-specific readable/writable
   lowering helpers.
3. Implement transform and shape helpers.
4. Implement direct mdspan GEMM over existing `uni20::blas::gemm`.
5. Add row-major writable GEMM output rewrite tests.
6. Add strict column-major writable descriptor helper for LAPACK update
   operands.
7. Add the minimal operation-tag dispatch infrastructure needed by GEMM.
8. Implement `gemm_op`, `BlasBackend`, and `CpuReferenceBackend` dispatch tests.
9. Implement `try_kernel(BlasBackend, gemm_op, ...)` by calling
   `uni20::linalg::blas::try_gemm(...)`.
10. Add a CPU reference GEMM fallback that is independently testable and respects
   BLAS `beta == 0` no-read semantics.
11. Add selector-list tests: BLAS accepts representable mdspans, BLAS
   declines unsupported strides before side effects, and the fallback runs.
12. Add provider-ready vector operands and rank-one mdspan staging.
13. Implement direct mdspan GEMV, including strided vectors and explicit
    zero/empty-product scaling semantics.
14. Implement `gemv_op`, BLAS and CPU backend adapters, Tensor forwarding, and
    clean fallback for conjugating vector accessors.

Completed LAPACK checkpoint:

1. Add checked scalar-generic conversion of LAPACK workspace-query results to
   `blas_int`.
2. Define strict direct column-major update-matrix contracts with clean
   pre-mutation layout decline.
3. Implement `sterf`/`steqr`, `geev`, `gees`, `hseqr`, and `trexc` adapters.
4. Route the active Krylov projected operations through those adapters.
5. Add strict contiguous eigenvalue-vector lowering and the in-place
   `self_adjoint_eigh` adapter over `syev`/`heev`; the preserving `eigh` value
   API explicitly copies to a column-major work matrix first.

The next wrapper should be driven by an active algorithm or example. It must
define overwrite, aliasing, workspace, and any operand-materialization policy
before its provider call is formed.

This order keeps the generic dispatch path tested by the GEMM and GEMV leaf
kernels while separating tensor default-backend policy from LAPACK workspace
and update-operand semantics.

## Tests

Descriptor tests:

- `layout_left`, `layout_right`, and `layout_stride` rank-2 views.
- padded views where one stride remains `1`.
- rejection when neither stride is `1`.
- rejection when extents or leading dimensions do not fit `blas_int`.
- const input and mutable output handle types.
- `needs_conjugation` is derived from the accessor policy.
- default/raw accessors derive `needs_conjugation == false`.
- `uni20::conj(mdspan)` derives `needs_conjugation == true` and composes into
  readable transforms.
- writable `needs_conjugation` is rejected by the direct GEMM wrapper.
- no implicit conversion erases readable metadata.

GEMM tests:

- column-major output direct path.
- row-major writable output rewrite.
- readable row-major inputs handled by transform helpers.
- complex transpose and conjugate-transpose combinations.
- conjugate-only complex states are declined by the generic direct wrapper
  before any provider call.
- CPU/reference fallback comparison for representative shapes and strides.

GEMV tests:

- direct column-major and row-major matrix lowering;
- positive-stride input and output vectors;
- singleton increment normalization;
- negative multi-element strides decline until lowering adjusts the provider
  starting pointer explicitly;
- conjugate-transpose matrix lowering and conjugating-vector decline;
- clean CPU fallback after a direct BLAS decline;
- `alpha == 0`, `beta == 0`, and empty-inner-dimension no-read semantics;
- Tensor storage-default and explicit-selector forwarding.

LAPACK tests:

- strict column-major writable matrix accepted.
- row-major-like writable matrix declined for in-place eigensolver first pass.
- eigenvalue vector shape and stride checks.
- scalar coverage for `float`, `double`, `uni20::complex<float>`, and
  `uni20::complex<double>`.
- binary128 real/complex when MPLAPACK is enabled.

## Open Questions

- Should prepared wrappers live beside direct wrappers or in a separate
  `prepared_*.hpp` namespace/header?
- Should direct GEMM retain only the Fortran-style BLAS facade, or also add a
  CBLAS-backed path when available?
