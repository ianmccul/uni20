# Mdspan BLAS/LAPACK Wrapper Layer Plan

**Status:** implementation plan for the first concrete mdspan dense-linalg
wrapper layer. This is below tensor front-end dispatch and above the existing
raw BLAS/LAPACK provider facades.

The immediate target is to add mdspan-facing BLAS/LAPACK wrappers that accept
resolved strided views, build BLAS-compatible matrix descriptors, and then call
the existing Uni20 BLAS/LAPACK backend wrappers. This is a new linalg adapter
layer, not a replacement for the current backend facades.

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

The first implementation lives under `src/uni20/linalg/blas/`:

- `matrix_transform.hpp`
  - backend-independent transform algebra and standard BLAS transpose lowering.
- `matrix_operand.hpp`
  - provider-ready `BlasReadableMatrix` and `BlasWritableMatrix` operands.
- `mdspan_matrix_stage.hpp`
  - `try_mdspan_matrix_stage(...)` and mdspan accessor conjugation traits.
- `mdspan_matrix_operand.hpp`
  - role-specific readable/writable lowering helpers and strict
    column-major-compatible helpers for LAPACK update operands.
- `gemm.hpp`
  - direct mdspan GEMM wrappers.
- `blas.hpp`
  - include point for the adapter layer.

Future LAPACK operation wrappers should either live under this directory when
they are shared BLAS/LAPACK adapter utilities, or under
`src/uni20/linalg/backends/lapack/` when they are LAPACK-backend operation
entry points. Keep the division between descriptor construction and operation
wrappers. Tests should be able to exercise descriptor helpers without linking
or running a LAPACK operation.

## Two Descriptor Levels

Keep two structures separate:

1. Minimal provider-ready BLAS matrix operands. These describe the
   column-major matrix that the BLAS/LAPACK ABI sees, plus any readable-input
   transform.
2. An mdspan staging descriptor. This describes the logical mdspan matrix and
   is used to decide whether direct lowering, a transpose rewrite, or a
   temporary is needed.

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
enum class MatrixTransform {
  normal,
  transpose,
  conjugate_transpose,
  conjugate
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

### Mdspan Staging Descriptor

The staging descriptor keeps the logical extents in mdspan axis order and
records which mdspan axis has stride `1`. It does not store BLAS
transposition state.

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
`BlasWritableMatrix` itself; it must be handled by the operation plan through an
explicit output postprocess, a backend extension, or a temporary/copy-back path.

The staging helper derives `needs_conjugation` from the mdspan accessor policy.
It is not an independent option supplied by the BLAS adapter. Uni20 does not yet
have a conjugating mdspan accessor, so the first implementation should derive
`false` for `stdex::default_accessor` and the current raw tensor accessors.
A future `conj(Tensor)` operation should return a tensor view whose mdspan uses
a conjugation adaptor accessor rather than eagerly materializing conjugated
storage. When that view reaches the BLAS staging helper,
`try_mdspan_matrix_stage(...)` should inspect the accessor and set
`needs_conjugation = true`. Future conjugating or Hermitian-view accessors
should opt into a local trait or customization point, for example:

```cpp
template <class Accessor>
inline constexpr bool accessor_applies_conjugation_v = false;

template <class View>
constexpr bool mdspan_needs_conjugation_v =
  accessor_applies_conjugation_v<typename std::remove_cvref_t<View>::accessor_type>;
```

The direct BLAS path should accept only accessors that can still expose the raw
storage handle plus declarative metadata such as `needs_conjugation`. An
accessor that computes arbitrary transformed values by value is a generic
transform view, not directly BLAS-addressable; it needs materialization or a
non-BLAS fallback. This follows the mdspan accessor model: mapping and accessor
describe how logical elements are read from the same handle, and the BLAS
adapter may lower only the subset it understands.

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

Runtime descriptor construction should reject views where neither matrix stride
is `1`. The first pass should also reject negative strides for direct
BLAS/LAPACK calls.

For non-degenerate matrices, the derived `BlasWritableMatrix::leading_dimension`
must satisfy the provider ABI lower bound:

- `unit_stride_axis == 0`: `nonunit_stride >= max(1, extent0)`;
- `unit_stride_axis == 1`: `nonunit_stride >= max(1, extent1)`.

Degenerate dimensions need explicit tests because the stride of an extent-1
axis may not describe a real next column. If both strides are `1`, choose an
axis whose derived BLAS leading dimension satisfies the provider
contract; prefer `unit_stride_axis == 0` only when both choices are valid. The
first implementation may reject ambiguous degenerate layouts, but it must not
silently construct a BLAS reference whose leading dimension violates the
provider contract.

Provider-ready BLAS information is derived, not stored in the mdspan staging
descriptor:

```cpp
template <class Scalar, class Handle>
BlasWritableMatrix<Scalar, Handle>
blas_writable_matrix(MdspanMatrixStage<Scalar, Handle> const& stage);

template <class Scalar, class Handle>
BlasReadableMatrix<Scalar, Handle>
blas_readable_matrix(MdspanMatrixStage<Scalar, Handle> const& stage,
                     MatrixTransform requested = MatrixTransform::normal);
```

These helpers return provider-ready objects with `rows = extent0`,
`cols = extent1`, and `leading_dimension = nonunit_stride` when
`unit_stride_axis == 0`; they swap the extents when `unit_stride_axis == 1`.
For readable operands, the returned `transform` is the effective transform after
composing the requested logical transform, the storage transform, and
`stage.needs_conjugation`. For writable operands, `blas_writable_matrix(...)`
returns the provider-writeable storage shape only. A stage with
`needs_conjugation == true` requires the caller's operation plan to account for
that conjugation explicitly. This keeps the staging descriptor unambiguously
mdspan-like while making the BLAS ABI conversion local and testable.

## Transform Helpers

Represent transforms independently of the backend-specific character, using the
`MatrixTransform` enum above.

The transform helper layer should provide:

- `storage_transform(MdspanMatrixStage const&)`;
- `transpose_result_transform(MatrixTransform)`;
- `compose(MatrixTransform, MatrixTransform)`;
- `logical_rows(...)`, `logical_cols(...)`, `abi_rows(...)`, `abi_cols(...)`;
- `standard_blas_trans_char(MatrixTransform) -> std::optional<char>`.

For readable operands, compose the requested logical transform, the storage
transform, and `stage.needs_conjugation`. Standard BLAS can lower `normal`,
`transpose`, and `conjugate_transpose` to `'N'`, `'T'`, and `'C'`. It should
decline `conjugate` unless a higher layer materializes, or a backend-specific
path uses an extension such as CBLAS `CblasConjNoTrans` or an MKL-style `'R'`
where available.

`storage_transform(view)` returns `normal` when `unit_stride_axis == 0` and
`transpose` when `unit_stride_axis == 1`. The helper name should describe the
derived BLAS interpretation; the stored descriptor remains mdspan-axis based.
There should be no stored `transposed` field.

If `L` is the logical mdspan matrix and `S` is the untransformed
`BlasWritableMatrix` shape derived from the same stage, then
`L = storage_transform(stage)(S)`. A requested logical input transform is
lowered into `BlasReadableMatrix::transform` as:

```text
effective_blas_transform = requested_logical_transform o storage_transform(stage)
```

where `o` means function composition. For example, a row-major-like input
stage has `storage_transform == transpose`. A requested logical `normal`
therefore lowers to BLAS `'T'`, while a requested logical `transpose` lowers to
BLAS `'N'`.

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

The helper should swap readable operands and apply
`transpose_result_transform(...)` to each requested operand transform. If the
resulting plan cannot be represented by the selected backend's ordinary BLAS
transpose characters, the operation planner can try a backend extension, an
explicit output postprocess, or a temporary path. A strict direct wrapper that
implements none of those should decline before side effects.

This rewrite is appropriate for BLAS update-style operations. It is not a
general rule for LAPACK drivers that overwrite inputs with structured outputs.

## Output Postprocessing

`BlasWritableMatrix` intentionally has no transform field. It is only the
provider-writeable storage target. Some operation lowerings still need an
output-side transform that cannot be expressed by standard BLAS transpose
characters.

For example, after composing row-major storage, requested operand transforms,
and input conjugation, the best standard-BLAS lowering may require a
conjugate-only state. If the selected backend supports a conjugate-no-transpose
extension for that operation, for example CBLAS `CblasConjNoTrans`, the
backend-specific path may lower it directly.
For a standard BLAS backend, a better generic plan may be:

1. call BLAS into the writable storage using the closest representable
   transpose/conjugate-transpose combination;
2. explicitly conjugate the output matrix in place after the BLAS call.

That postprocess is not part of `BlasWritableMatrix`; it belongs to the
operation plan. A sketch is:

```cpp
enum class MatrixOutputPostprocess {
  none,
  conjugate
};

template <class Scalar, class Handle = Scalar*>
struct BlasWritableMatrixPlan {
  BlasWritableMatrix<Scalar, Handle> output;
  MatrixOutputPostprocess postprocess = MatrixOutputPostprocess::none;
};
```

Direct no-copy wrappers may still decline if they do not implement the required
postprocess. They should report that this particular lowering path does not
support writable conjugation.

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
enough to decide whether to swap operands, transpose requested transforms, or
allocate temporaries.

If `A` were row-major-like instead, its stage would derive a
`BlasReadableMatrix` whose `rows` and `cols` describe the provider's
untransformed `A^T` storage, and whose `transform` is `'T'` for a logical
`normal` input. If composing the requested transform with storage layout
produces `conjugate` for a standard BLAS backend, the direct wrapper must
decline or use a prepared input temporary whose contract allows materialization.

## Direct And Prepared Wrappers

There are two wrapper contracts:

1. **Direct wrappers** do not allocate and do not copy as a fallback. They build
   descriptors, call the provider wrapper if the instance is representable, or
   decline before side effects.
2. **Prepared wrappers** may materialize input-only operands into explicit
   scratch storage when their public contract says so.

Output/update temporaries are stricter. They require explicit copy-back policy,
aliasing checks before side effects, and diagnostics that report the
materialization.

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

The first implementation does not need the full generic backend-list dispatcher.
The wrapper layer can start with direct functions that are easy to test:

```cpp
bool try_gemm(... mdspan-like operands ...);
void gemm_or_throw(... mdspan-like operands ...);

bool try_lapack_self_adjoint_eigh(... mdspan-like operands ...);
void lapack_self_adjoint_eigh_or_throw(... mdspan-like operands ...);
```

Once these are correct, operation-tag dispatch can wrap them:

```cpp
bool try_kernel(BlasBackend, gemm_op, C&& c, Scalar alpha, A&& a, B&& b, Scalar beta);
bool try_kernel(LapackBackend, self_adjoint_eigh_op, W&& w, A&& matrix_work);
```

`kernel_accepts_types(...)` remains type-level: scalar family, rank, mutability,
and host raw-addressability. Descriptor construction remains runtime:
dynamic extents, strides, pointer values, and aliasing.

## First Implementation Slice

1. Add concept and descriptor helper tests.
2. Implement `try_mdspan_matrix_stage(...)` and role-specific readable/writable
   lowering helpers.
3. Implement transform and shape helpers.
4. Implement direct mdspan GEMM over existing `uni20::blas::gemm`.
5. Add row-major writable GEMM output rewrite tests.
6. Add strict column-major writable descriptor helper for LAPACK update
   operands.
7. Implement one LAPACK wrapper, likely self-adjoint eigensystem, over existing
   `uni20::lapack::syev/heev`.
8. Wrap the direct functions in operation-tag `try_kernel(...)` overloads after
   the helper behavior is stable.

This order deliberately tests descriptor mechanics before binding them to the
generic dispatch machinery.

## Tests

Descriptor tests:

- `layout_left`, `layout_right`, and `layout_stride` rank-2 views.
- padded views where one stride remains `1`.
- rejection when neither stride is `1`.
- rejection when extents or leading dimensions do not fit `blas_int`.
- const input and mutable output handle types.
- `needs_conjugation` is derived from the accessor policy.
- default/raw accessors derive `needs_conjugation == false`.
- a fake conjugating accessor derives `needs_conjugation == true` and composes
  into readable transforms.
- writable `needs_conjugation` is handled by an explicit output postprocess,
  backend extension, temporary/copy-back path, or a clean runtime decline.
- no implicit conversion erases readable metadata.

GEMM tests:

- column-major output direct path.
- row-major writable output rewrite.
- readable row-major inputs handled by transform helpers.
- complex transpose and conjugate-transpose combinations.
- standard BLAS handles conjugate-only states through materialization, output
  postprocessing, a backend extension, or a clean runtime decline.
- CPU/reference fallback comparison for representative shapes and strides.

LAPACK tests:

- strict column-major writable matrix accepted.
- row-major-like writable matrix declined for in-place eigensolver first pass.
- eigenvalue vector shape and stride checks.
- scalar coverage for `float`, `double`, `uni20::complex<float>`, and
  `uni20::complex<double>`.
- binary128 real/complex when MPLAPACK is enabled.

## Open Questions

- Should direct BLAS wrappers return `bool` plus leave diagnostics to the
  caller, or return a small decline reason for better test messages?
- Should prepared wrappers live beside direct wrappers or in a separate
  `prepared_*.hpp` namespace/header?
- What should the accessor-derived conjugation trait/customization point be
  called, and can it align with Kokkos/stdBLAS-style accessor conventions?
- Which existing Krylov dense helpers should migrate first after the mdspan
  LAPACK wrapper exists?
- Should the first GEMM wrapper use only the Fortran-style BLAS facade, or also
  add a CBLAS-backed convenience path when available?
