# Mdspan Linear Algebra Dispatch Plan

**Status:** design plan for the next linear algebra layer. The existing LAPACK
wrappers and Krylov dense helpers are implemented, but the mdspan-based linalg
dispatch layer described here is not yet the public API.

This note makes the next implementation step concrete: add mdspan-based wrappers
over the existing Uni20 BLAS/LAPACK backend layer, then use those wrappers as the
first real consumer of the kernel-dispatch design in
[`kernel_dispatch.md`](kernel_dispatch.md).

The static CPO name in this note is `kernel_accepts_types(...)`. It returns a
`KernelTypeAcceptance` value (`no`, `maybe`, or `yes`) for the C++ argument
types. It does not inspect extents, strides, pointer values, library
availability, or device state. Those checks belong to the runtime attempt,
`try_kernel(...)`.

Related notes:

- [`backend_dispatch.md`](backend_dispatch.md) - compile-time capability,
  runtime attempt, and fallback contract.
- [`kernel_dispatch.md`](kernel_dispatch.md) - operation tags and ordered
  backend lists.
- [`tensor_dispatch_and_view_semantics_draft.md`](tensor_dispatch_and_view_semantics_draft.md)
  - tensor/front-end operation roles versus resolved mdspan leaf kernels.
- [`matrix.md`](matrix.md) - sparse Matrix Market helpers and current matrix
  vocabulary.
- [`mplapack_binary128_setup.md`](mplapack_binary128_setup.md) - current
  binary128 LAPACK provider setup.

## Goal

The linalg leaf-kernel entry point should be a strided mdspan-like view, not a
Krylov-only matrix class and not a vendor-specific pointer API.

The intended contract is:

1. Compile-time checks decide whether a backend is even a candidate for the
   operand types.
2. Runtime checks decide whether a particular view instance can be represented
   by the backend call.
3. The backend performs the operation or declines cleanly.

For LAPACK, this means the type-level check verifies host-accessible scalar
storage with a supported scalar type and the required rank/mutability. The
runtime check verifies details such as extents fitting `blas_int` and matrix
strides being representable by LAPACK.

Matrix operands for this layer are rank-2 views. Vector operands such as
eigenvalue arrays, Householder coefficients, pivots, and work/result vectors are
rank-1 views. Rank is therefore an immediate compile-time filter in the linalg
concepts, not a runtime LAPACK-layout check.

## Existing Inputs

Uni20 already has the pieces to build on:

- `uni20::StridedMdspan` and `uni20::MutableStridedMdspan` in
  `src/uni20/mdspan/concepts.hpp`.
- Checked LAPACK wrappers in `src/uni20/backend/lapack/lapack.hpp`.
- Provider-specific unchecked wrappers under `src/uni20/backend/lapack/reference`
  and `src/uni20/backend/lapack/mplapack`.
- Prototype dense local matrices in
  `src/uni20/linalg/backends/cpu/dense_matrix.hpp`.
- Krylov dense subspace helpers in `src/uni20/krylov/dense_subspace.hpp`.

The abandoned `../cytnx-mdspan` prototype is useful for operation-tag examples
and diagnostic formatting, but Uni20 should not copy its `std::variant` dispatch
funnel for the first pass. Uni20's current draft design uses an ordered backend
list with a type-level capability CPO and a runtime attempt CPO.

## Non-Goals

- Do not redeclare raw Fortran symbols in the mdspan layer. It should call the
  existing `uni20::lapack::*` wrappers.
- Do not make MPLAPACK a separate front-end backend. MPLAPACK is the configured
  provider beneath `uni20::lapack` for supported binary128 types.
- Do not add hidden copies in strict low-level `try_kernel(...)` wrappers as a
  surprise fallback. Prepared-view helpers may materialize input-only operands
  into explicit scratch matrices, and higher-level tensor operations may own
  output copy-back policy deliberately.
- Do not introduce runtime `std::variant` backend selection yet. It can be
  layered on later as another backend-list entry if Python or plugin boundaries
  need it.

## Layer Shape

The final layering should look like this:

1. **Tensor/linalg front end**
   - Accepts Uni20 tensors, tensor refs, matrix adaptors, or explicit mdspans.
   - Prepares outputs with `ensure_shape(...)` when the output is resizable.
   - Derives or receives a backend selector.
2. **Kernel dispatch**
   - Walks an ordered backend list for an operation tag.
   - Uses `kernel_accepts_types(...)` / detected `try_kernel(...)`.
   - Calls the first backend whose runtime `try_kernel(...)` succeeds.
3. **Linalg leaf kernel**
   - Receives resolved mdspan-like views.
   - Converts extents, strides, transpose flags, and triangle flags to the
     vendor wrapper call.

Bare mdspan calls are allowed as leaf-kernel calls, but they cannot derive a
default Uni20 backend stack by themselves. They need an explicit backend
selector, or a concrete convenience wrapper such as "use LAPACK for this view".

## Mdspan Concepts

The first implementation should refine the existing Uni20 mdspan concepts
instead of introducing unrelated concept names. The generic rank filter is not
specific to linalg and should probably live with the mdspan concepts. The linalg
layer should then add scalar-family and host-addressability refinements.

Candidate generic mdspan refinement:

```cpp
template <class View, std::size_t Rank>
concept RankedStridedMdspan =
    uni20::StridedMdspan<std::remove_cvref_t<View>> &&
    std::remove_cvref_t<View>::rank() == Rank;

template <class View, std::size_t Rank>
concept MutableRankedStridedMdspan =
    uni20::MutableStridedMdspan<std::remove_cvref_t<View>> &&
    std::remove_cvref_t<View>::rank() == Rank;
```

Candidate linalg refinements:

```cpp
template <class View>
concept LinalgMatrixView =
    RankedStridedMdspan<View, 2> &&
    uni20::RealOrComplex<
        std::remove_cv_t<typename std::remove_cvref_t<View>::element_type>>;

template <class View>
concept MutableLinalgMatrixView =
    MutableRankedStridedMdspan<View, 2> &&
    LinalgMatrixView<View>;

template <class View>
concept LinalgVectorView =
    RankedStridedMdspan<View, 1> &&
    uni20::RealOrComplex<
        std::remove_cv_t<typename std::remove_cvref_t<View>::element_type>>;

template <class View>
concept MutableLinalgVectorView =
    MutableRankedStridedMdspan<View, 1> &&
    LinalgVectorView<View>;

template <class View>
concept RealLinalgMatrixView =
    RankedStridedMdspan<View, 2> &&
    uni20::Real<
        std::remove_cv_t<typename std::remove_cvref_t<View>::element_type>>;

template <class View>
concept ComplexLinalgMatrixView =
    RankedStridedMdspan<View, 2> &&
    uni20::Complex<
        std::remove_cv_t<typename std::remove_cvref_t<View>::element_type>>;

template <class View>
concept RealLinalgVectorView =
    RankedStridedMdspan<View, 1> &&
    uni20::Real<
        std::remove_cv_t<typename std::remove_cvref_t<View>::element_type>>;

template <class View>
concept ComplexLinalgVectorView =
    RankedStridedMdspan<View, 1> &&
    uni20::Complex<
        std::remove_cv_t<typename std::remove_cvref_t<View>::element_type>>;

template <class View, std::size_t Rank>
concept HostRawAddressableRankedView =
    RankedStridedMdspan<View, Rank> &&
    raw_host_accessor_v<typename std::remove_cvref_t<View>::accessor_type> &&
    raw_data_handle_compatible_v<std::remove_cvref_t<View>>;

template <class View, std::size_t Rank>
concept LapackScalarRankedView =
    HostRawAddressableRankedView<View, Rank> &&
    uni20::LapackRealOrComplex<
        std::remove_cv_t<typename std::remove_cvref_t<View>::element_type>>;
```

The exact names can change, but the split matters:

- `StridedMdspan`: structural mdspan-like API with runtime strides.
- `RankedStridedMdspan<View, Rank>`: structural mdspan-like API plus a static
  rank requirement; this belongs in the generic mdspan layer.
- `LinalgMatrixView` and `LinalgVectorView`: rank-specific real-or-complex
  linalg operands.
- `RealLinalgMatrixView`, `ComplexLinalgMatrixView`,
  `RealLinalgVectorView`, and `ComplexLinalgVectorView`: longer names for
  operations that need a specific scalar family.
- `HostRawAddressableRankedView`: memory can be passed to a host pointer ABI.
- `LapackScalarRankedView`: element type is exactly supported by the configured
  LAPACK layer; this is a backend refinement, not the base linalg concept.
- mutability constraints belong to each operation.

The implementation can add named mutable real/complex aliases such as
`MutableRealLinalgVectorView` when call sites become clearer, but the important
policy is that the short names are real-or-complex and the longer names state a
specific scalar family.

The compile-time check should reject device accessors, transform accessors,
unsupported scalar types, rank mismatches, and const outputs before the function
body is instantiated.

## Parameter Order

New Uni20 linalg and kernel APIs should put mutable outputs first. This applies
to mdspan leaf wrappers, tensor-facing linalg wrappers, and backend
runtime-attempt overloads.

Examples:

```cpp
matvec(y, A, x);
gemm(C, alpha, A, B, beta);
self_adjoint_eigh(eigenvalues, matrix_work);
schur(schur_form, schur_vectors, eigenvalues, matrix_work);
```

For update operations, the updated object is both input and output, but it still
appears first. Backend selectors and optional policy objects remain trailing
parameters. Older draft documents may still contain BLAS/LAPACK-style
output-last examples; new code should use the output-first convention unless an
external ABI boundary forces another order.

## Runtime LAPACK View Checks

LAPACK can operate directly only when a view instance is representable by the
Fortran ABI call. This is a runtime property because dynamic extents and strides
matter.

For a rank-2 matrix view, the direct adapter should compute a descriptor:

```cpp
enum class LapackUnitStrideAxis { row, column };

template <class Scalar>
struct LapackMatrixViewInfo {
  Scalar* data;
  blas_int rows;
  blas_int cols;
  blas_int leading_dimension;
  LapackUnitStrideAxis unit_stride_axis;
  static constexpr bool needs_conjugation = false;
};

template <class Real>
struct LapackMatrixViewInfo<uni20::complex<Real>> {
  uni20::complex<Real>* data;
  blas_int rows;
  blas_int cols;
  blas_int leading_dimension;
  LapackUnitStrideAxis unit_stride_axis;
  bool needs_conjugation = false;
};
```

For real matrix element types, conjugation is a compile-time no-op. For complex
matrix element types, `needs_conjugation` records whether the storage
interpretation requires elementwise conjugation in addition to any transpose.
Conjugating views are input-only. Uni20 should not expose a conjugate view as
the left-hand side of an update or output expression; if an output/update would
need conjugation, the backend must rewrite the operation, use an explicit
temporary/copy-back path owned by the higher-level wrapper, or decline.

Runtime checks:

- rank is already a compile-time operation constraint.
- `extent(0)` and `extent(1)` fit `blas_int`.
- at least one matrix stride is `1`.
- the non-unit stride fits `blas_int`.
- the leading dimension is large enough for the interpretation used by the
  operation.
- vector lengths and increments fit the backend integer type.
- writable outputs do not have const element type.
- operation-specific aliasing restrictions are satisfied.

The simple column-major case is:

```text
stride(0) == 1
lda = stride(1)
```

The row-major-like case is:

```text
stride(1) == 1
lda = stride(0)
```

Having `stride(1) == 1` means the same storage can often be interpreted as a
column-major view of the transpose. Whether that is valid without copying is
operation-specific:

- Value-only operations may be able to flip transpose or triangle flags.
- BLAS update operations may be representable through transpose flags.
- In-place factorization and eigenvector-producing LAPACK drivers may need the
  native column-major interpretation, or an explicit higher-level temporary.

So "one stride is 1" is the baseline runtime representability test, but each
operation still decides whether row-major-as-transposed semantics are valid for
its outputs.

### Direct Views And Prepared Views

The low-level direct LAPACK view helper is non-owning. If the input mdspan is
already representable by a BLAS/LAPACK ABI call, `LapackMatrixViewInfo::data`
points at the original storage and no allocation occurs.

For input-only operands, a higher-level prepared-view helper may own an optional
scratch `Matrix`:

```cpp
template <class Scalar>
struct PreparedLapackMatrixView {
  Matrix<Scalar> scratch;              // empty when no copy is needed
  LapackMatrixViewInfo<Scalar> info;   // points at original storage or scratch
};
```

If the input view is not directly representable, the helper copies it into
`scratch` in a supported layout and points `info.data` at the scratch storage.
Constructing an empty `Matrix` for the direct case is acceptable; dense
projected matrices are small in the Krylov use cases, and the LAPACK call will
dominate.

Mutable outputs and update operands are stricter. A helper may rewrite an
operation algebraically, for example by interpreting a row-major output as the
transpose of a column-major output and swapping/transposing the input operands,
but this should be implemented through reusable preprocessing helpers rather
than ad-hoc fixes inside each kernel. If a genuine temporary output is needed,
the wrapper must own the copy-back semantics explicitly and reject unsupported
aliasing or partial-overlap cases.

Complex transpose state should be represented explicitly, for example:

```cpp
enum class MatrixTransform {
  normal,
  transpose,
  conjugate_transpose,
  conjugate
};
```

Standard BLAS/LAPACK transpose flags cover `normal`, `transpose`, and
`conjugate_transpose` (`'N'`, `'T'`, and `'C'`). MKL additionally supports
`'R'` for conjugate-without-transpose in some BLAS interfaces. Uni20 should keep
`conjugate` as a first-class internal transform state; the standard LAPACK
backend can materialize or decline when it cannot express that state directly,
while an MKL-specific backend can lower it to `'R'` where that extension is
available.

### Materialization Policy

"No hidden copies" should mean "no copies that are invisible in the contract",
not "copies are forbidden everywhere." There are three useful levels:

1. **Direct leaf kernels**: `try_kernel(...)` over already-resolved views does
   not allocate or copy as a fallback. It either accepts the instance and calls
   the backend, or declines before side effects. This keeps backend dispatch
   predictable and makes performance debugging straightforward.
2. **Prepared linalg wrappers**: named helpers or public linalg functions may
   document that they accept general strided input views and may materialize
   input-only operands into scratch storage. This is often the ergonomic default
   for dense projected Krylov matrices, where the copy cost is small compared
   with the LAPACK call.
3. **Output/update materialization**: copy-back is much more delicate because it
   changes aliasing, lifetime, and failure semantics. It should be allowed only
   when the wrapper explicitly owns that policy, checks aliasing before the
   first side effect, and records the materialization in diagnostics.

A public wrapper can expose this choice explicitly:

```cpp
enum class MaterializationPolicy {
  direct_only,        // no allocation; decline or throw if not backend-ready
  input_temporaries,  // may copy read-only operands into backend-ready storage
  explicit_copy_back  // may use output temporaries with documented copy-back
};
```

The likely default is `input_temporaries` for value-style high-level linalg
wrappers and `direct_only` for low-level `try_kernel(...)` calls and in-place
mdspan wrappers. `explicit_copy_back` should be opt-in until the aliasing and
diagnostic behavior is well exercised.

## Kernel Dispatch Interface

The mdspan linalg layer should be the first concrete user of the operation-tag
model. Backend CPOs put the backend value first, then the operation tag, then
ordinary reference parameters matching the public call order. A separate
`state_type<State>` / `backend_type<Backend>` tag layer is not needed for the
LAPACK first pass.

```cpp
struct self_adjoint_eigh_op {
  char jobz = 'N';
  char uplo = 'U';
};

struct gees_op {
  char jobvs = 'N';
  char sort = 'N';
};

struct LapackBackend {};

template <class W, class A>
consteval KernelTypeAcceptance
kernel_accepts_types(LapackBackend const&, self_adjoint_eigh_op const&,
                     W&, A&)
{
  if constexpr (mutable_lapack_candidate_real_vector<W> &&
                mutable_lapack_candidate_matrix<A> &&
                same_real_value_type<A, W>) {
    return KernelTypeAcceptance::maybe; // extents, strides, and aliasing are runtime
  } else {
    return KernelTypeAcceptance::no;
  }
}

template <class W, class A>
bool try_kernel(LapackBackend, self_adjoint_eigh_op op, W&& w, A&& a)
{
  auto values = try_lapack_vector_view(w);
  auto matrix = try_lapack_matrix_view(a);
  if (!matrix || !values) {
    return false;
  }

  // Query workspace, then call uni20::lapack::syev/heev through checked wrappers.
  return true;
}
```

`kernel_accepts_types(...)` should stay type-level. It should not inspect
runtime strides, sizes, pointer values, or backend state. The dispatcher may
call it with private type-probe lvalues, not real operands. The runtime attempt
does the value checks and returns `false` if the selected backend cannot
represent this particular view.

The dispatch helper can call this consteval function using probe lvalues, so
the CPO does not need a tuple or explicit type-pack token:

```cpp
enum class KernelTypeAcceptance { no, maybe, yes };

namespace detail {
template <class T>
extern std::remove_reference_t<T> kernel_type_probe_object;

template <class T>
constexpr std::remove_reference_t<T>& kernel_type_probe_arg() noexcept
{
  return kernel_type_probe_object<T>;
}
} // namespace detail

template <class Backend, class Op, class... Args>
consteval KernelTypeAcceptance kernel_type_acceptance()
{
  if constexpr (requires {
                  kernel_accepts_types(detail::kernel_type_probe_arg<Backend const&>(),
                                       detail::kernel_type_probe_arg<Op const&>(),
                                       detail::kernel_type_probe_arg<Args>()...);
                }) {
    constexpr auto acceptance =
        kernel_accepts_types(detail::kernel_type_probe_arg<Backend const&>(),
                             detail::kernel_type_probe_arg<Op const&>(),
                             detail::kernel_type_probe_arg<Args>()...);
    if constexpr (acceptance == KernelTypeAcceptance::no) {
      return KernelTypeAcceptance::no;
    } else {
      static_assert(backend_has_try_kernel<Backend, Op, Args...>());
      return acceptance;
    }
  }

  if constexpr (backend_has_try_kernel<Backend, Op, Args...>())
    return KernelTypeAcceptance::maybe;
  else
    return KernelTypeAcceptance::no;
}
```

The fallback to a constrained `try_kernel(...)` overload keeps simple backends
from having to define a separate static CPO when the runtime-attempt signature is
already specific enough.

## Worked Example: Self-Adjoint Eigensolver

The first LAPACK mdspan wrapper should probably be the standard dense
symmetric/Hermitian eigensolver. It is used by projected Lanczos problems, it
exercises real and complex scalar paths, and it shows the distinction between
type eligibility and runtime view representability.

### Public Shape

Use an output-first, in-place work-matrix signature:

```cpp
struct SelfAdjointEighOptions {
  bool compute_vectors = true;
  MatrixTriangle triangle = MatrixTriangle::Upper;
};

template <class W, class A, class BackendSelector = backend_list<LapackBackend>>
void self_adjoint_eigh(W&& eigenvalues, A&& matrix_work,
                       SelfAdjointEighOptions options = {},
                       BackendSelector selector = {});
```

`matrix_work` is mutable because LAPACK overwrites its input. If
`compute_vectors` is true, `matrix_work` contains the eigenvectors on return.
If `compute_vectors` is false, `matrix_work` should be treated as destroyed
workspace. A higher-level value API can preserve an input matrix by copying it
into an explicit temporary and then calling this in-place wrapper.

Move-aware higher-level wrappers can also avoid that copy:

```cpp
template <class Matrix>
auto self_adjoint_eigh(Matrix matrix) -> SelfAdjointEighResult<Matrix>
{
  using Real = uni20::make_real_t<typename std::remove_cvref_t<Matrix>::value_type>;
  std::vector<Real> eigenvalues(matrix.extent(0));
  self_adjoint_eigh(mdspan(eigenvalues), mdspan(matrix));
  return {.eigenvalues = std::move(eigenvalues),
          .eigenvectors = std::move(matrix)};
}
```

That is valid because the matrix is a single update operand: LAPACK's `A`
argument is both the input matrix and, for `jobz = 'V'`, the eigenvector output.
The higher-level wrapper can transfer ownership with `std::move(M)`, call the
in-place leaf wrapper, and return the same storage as the eigenvector matrix.

This should not be represented as arbitrary aliasing between independent
parameters. If a higher-level API has separate `input` and `eigenvectors`
arguments, then it should either:

- detect exact full aliasing and lower to the in-place wrapper, or
- reject aliasing and copy `input` into a work matrix before calling LAPACK.

Partial overlap must be rejected. The eigenvalue vector must not alias the
matrix work storage or LAPACK workspace.

The scalar relation is:

```text
matrix element type:      T or uni20::complex<T>
eigenvalue element type:  T
```

So `ssyev`/`dsyev`/`Rsyev` cover real matrices, and
`cheev`/`zheev`/`Cheev` cover complex Hermitian matrices.

### Compile-Time Eligibility

The operation-specific concept can be written in terms of the generic rank and
scalar-family concepts:

```cpp
template <class View>
using view_value_t =
    std::remove_cv_t<typename std::remove_cvref_t<View>::element_type>;

template <class A, class W>
concept SelfAdjointEighOperands =
    MutableLinalgMatrixView<A> &&
    MutableLinalgVectorView<W> &&
    uni20::LapackScalar<view_value_t<A>> &&
    uni20::LapackReal<view_value_t<W>> &&
    std::same_as<view_value_t<W>, uni20::make_real_t<view_value_t<A>>>;
```

For the LAPACK backend this is not quite enough, because LAPACK also needs raw
host-addressable storage:

```cpp
template <class A, class W>
concept LapackSelfAdjointEighOperands =
    SelfAdjointEighOperands<A, W> &&
    HostRawAddressableRankedView<A, 2> &&
    HostRawAddressableRankedView<W, 1>;
```

Then the backend type check is concise:

```cpp
template <class W, class A>
consteval KernelTypeAcceptance
kernel_accepts_types(LapackBackend const&, self_adjoint_eigh_op const&,
                     W&, A&)
{
  if constexpr (LapackSelfAdjointEighOperands<A, W>) {
    return KernelTypeAcceptance::maybe;
  } else {
    return KernelTypeAcceptance::no;
  }
}
```

This rejects, at compile time:

- rank-3 tensors accidentally passed as matrices.
- rank-2 matrices accidentally passed as eigenvalue vectors.
- const output views.
- integer or unsupported scalar types.
- complex eigenvalue output for a Hermitian problem.
- device or transform accessors that cannot be passed to a host pointer ABI.

### Runtime Representability

The strict first LAPACK implementation should require column-major direct
representation:

```text
matrix_work.rank() == 2          // compile-time checked
matrix_work.extent(0) == n
matrix_work.extent(1) == n
matrix_work.stride(0) == 1
lda = matrix_work.stride(1)
eigenvalues.rank() == 1          // compile-time checked
eigenvalues.extent(0) == n
eigenvalues.stride(0) == 1       // first pass; non-unit vector increments can be added later
```

`stride(1) == 1` row-major-like matrices should initially decline. Real
symmetric value-only solves can probably support row-major-as-transposed by
flipping `uplo`, but complex Hermitian eigenvectors are easy to mishandle
because row-major storage interpreted as column-major storage represents a
transpose. Keeping the first wrapper strict avoids silently returning
conjugated/transposed eigenvector data. A higher-level tensor wrapper can make
an explicit column-major temporary when it wants to support arbitrary strided
inputs.

### Runtime Attempt

The runtime attempt is schematic but should have this shape:

```cpp
template <class W, class A>
bool try_kernel(LapackBackend, self_adjoint_eigh_op op, W&& eigenvalues,
                A&& matrix_work)
{
  auto matrix = try_lapack_column_major_matrix(matrix_work);
  auto values = try_lapack_contiguous_vector(eigenvalues);
  if (!matrix || !values) {
    return false;
  }

  if (matrix->rows != matrix->cols || values->size != matrix->rows) {
    return false;
  }

  char const jobz = op.compute_vectors ? 'V' : 'N';
  char const uplo = lapack_triangle(op.triangle);
  blas_int const n = matrix->rows;
  blas_int const lda = matrix->leading_dimension;

  using MatrixScalar = view_value_t<A>;
  using Real = view_value_t<W>;

  if constexpr (uni20::LapackReal<MatrixScalar>) {
    MatrixScalar work_query{};
    uni20::lapack::syev(jobz, uplo, n, matrix->data, lda, values->data,
                        &work_query, -1);

    blas_int const lwork = checked_lwork(work_query);
    std::vector<MatrixScalar> work(static_cast<std::size_t>(lwork));
    uni20::lapack::syev(jobz, uplo, n, matrix->data, lda, values->data,
                        work.data(), lwork);
  } else {
    MatrixScalar work_query{};
    std::vector<Real> rwork(heev_rwork_size(n));
    uni20::lapack::heev(jobz, uplo, n, matrix->data, lda, values->data,
                        &work_query, -1, rwork.data());

    blas_int const lwork = checked_lwork(real_part(work_query));
    std::vector<MatrixScalar> work(static_cast<std::size_t>(lwork));
    uni20::lapack::heev(jobz, uplo, n, matrix->data, lda, values->data,
                        work.data(), lwork, rwork.data());
  }

  return true;
}
```

The actual helper names can differ, but the responsibilities should stay split:

- `kernel_accepts_types(...)` says this backend can compile for these view
  types, returning `maybe` for direct LAPACK views because extents, strides,
  aliasing, and provider availability are runtime facts.
- `try_lapack_column_major_matrix(...)` checks the particular extents and
  strides.
- prepared-view helpers may materialize input-only matrices into an owned
  `Matrix` scratch before calling a direct LAPACK wrapper.
- `uni20::lapack::syev/heev` perform the provider-dispatched LAPACK call.
- LAPACK convergence errors throw through the checked wrapper after the backend
  has accepted the operation.

### Tests For This Wrapper

Minimal tests:

- `float`, `double`, `uni20::complex<float>`, and `uni20::complex<double>`.
- binary128 real and complex when MPLAPACK is enabled.
- rank and scalar rejection through `static_assert`.
- runtime decline for a rank-2 view with neither stride equal to `1`.
- runtime decline for row-major `stride(1) == 1` until that case is explicitly
  implemented.
- eigenvalues agree with the current Krylov dense helper.
- eigenvectors, when requested, satisfy
  `||A v_i - lambda_i v_i|| / scale` within the scalar-specific tolerance.

## Error Semantics

Low-level `try_kernel(...)` functions should return `false` for clean runtime
decline before side effects. They should throw only for errors after the backend
operation has been selected and called, such as LAPACK convergence failures or
invalid vendor status codes.

Public convenience wrappers should turn "all backends declined" into a clear
exception:

```text
no available backend for self_adjoint_eigh: LAPACK requires a host raw-addressable
rank-2 matrix with at least one unit stride
```

The diagnostic machinery from the cytnx prototype is worth adapting later:
operation name, argument rank, extents, strides, scalar type, layout type, and
accessor kind.

## Initial Operation Set

The first mdspan wrappers should cover operations already used by the native
Krylov solvers:

| Operation | LAPACK routines | Reason |
| --- | --- | --- |
| symmetric/Hermitian dense eigensystem | `syev`, `heev`, then `syevd`/`heevd` | Lanczos projected problems and complex Hermitian coverage |
| symmetric tridiagonal eigensystem | `stevd`, `stedc`, `sterf` | Lanczos tridiagonal diagonalization |
| real and complex Schur decomposition | `gees`, `hseqr`, `trexc`, `trsen` | Arnoldi restarts and reordering |
| nonsymmetric eigensystem | `geev`, `geevx` | validation and examples |
| QR/LQ factorization | `geqrf`, `orgqr`/`ungqr`, `gelqf`, `orglq`/`unglq` | Arnoldi and future dense utilities |
| SVD and least squares | `gesvd`, `gesvdx`, `gelsd` | useful linalg API coverage and tests |

Do not expose every backend wrapper through mdspan immediately. Add wrappers
when a Uni20 algorithm or example needs them, and keep unused wrappers in a
separate tested header if we want to preserve prototype coverage.

## Implementation Phases

### Phase 1: Concepts and View Descriptors

Add linalg-specific mdspan concept refinements and runtime descriptor helpers.

Candidate files:

- `src/uni20/linalg/mdspan_concepts.hpp`
- `src/uni20/linalg/mdspan_lapack_view.hpp`

Tests:

- layout-left, layout-right, and layout-stride rank-1/rank-2 views.
- const input versus mutable output.
- unsupported scalar rejection through `static_assert`.
- device or transform accessor rejection where test accessors exist.
- runtime rejection when neither matrix stride is `1`.
- runtime rejection when extents or strides do not fit `blas_int`.
- complex input transform descriptors, including conjugate-only states.
- prepared input matrix copies when no matrix stride is `1`.

### Phase 2: LAPACK Operation Wrappers

Implement a small direct mdspan LAPACK layer over existing checked wrappers.

This layer should:

- perform workspace queries internally.
- derive `lda`, `ldu`, `ldv`, and vector increments from view descriptors.
- call `uni20::lapack::*`, never raw Fortran symbols.
- keep strict direct wrappers non-owning and no-copy.
- provide reusable prepared-view helpers for input-only materialization and
  algebraic transpose/conjugation rewrites.
- document when an operation supports row-major-as-transposed or conjugated
  views without copying.

Start with Hermitian/symmetric eigensolvers and Schur routines because those are
the highest-value Krylov dependencies.

### Phase 3: Kernel Dispatch Skeleton

Implement the generic pieces from `kernel_dispatch.md` only as much as the
mdspan LAPACK layer needs:

- `backend_list<...>`.
- backend values, usually empty stateless tags in the LAPACK first pass.
- `kernel_accepts_types(backend const&, op const&, args&...)` detection using
  private type-probe lvalues.
- ordered `dispatch_kernel(...)`.
- whole-list entry-point `static_assert`, not recursive-tail assertions.

Tests should use fake backends to verify:

- compile-time ineligible backend is skipped.
- runtime decline falls through to the next backend.
- forced single-backend dispatch reports a clean failure.
- a later eligible backend prevents a whole-list static assertion.

### Phase 4: Public Linalg Convenience API

Add public convenience wrappers for explicit mdspan calls:

```cpp
self_adjoint_eigh(w, a, backend_list<LapackBackend>{});
schur(t, q, w, a, backend_list<LapackBackend>{});
```

The exact function names should be chosen at implementation time, but the call
surface should make output ownership explicit. Mutable outputs should appear
first. For in-place LAPACK-style wrappers, the overwritten work matrix is a
mutable output/update operand and should be grouped with the other output
operands before read-only inputs.

### Phase 5: DenseMatrix and Tensor Integration

Add mdspan views to the temporary dense matrix type and later to the final
rank-2 tensor alias:

```cpp
auto view(DenseMatrix<T>& matrix) -> stdex::mdspan<T, ..., stdex::layout_left>;
auto view(DenseMatrix<T> const& matrix) -> stdex::mdspan<T const, ..., stdex::layout_left>;
```

Then migrate Krylov dense subspace helpers from direct pointer-level LAPACK
calls to mdspan linalg wrappers. This should remove most of the temporary
`RightMatrix`/copy-left-copy-right glue used while the LAPACK wrappers were
being brought up.

### Phase 6: Tensor Front-End Dispatch

Once the leaf mdspan path is stable, connect tensor-facing operations:

- derive default backend selectors from tensor storage policy.
- call `ensure_shape(...)` before materializing output views.
- lower tensors, tensor refs, and adaptors to resolved mdspan-like views.
- use the same operation tags and backend-list walk.

This is where CUDA, MPI/block tensor placement, async scheduling, and temporary
allocation policy enter. They should not be forced into the first LAPACK mdspan
wrapper step.

## Testing Strategy

Use three classes of tests:

1. **Compile-time concept tests**
   - Verify accepted and rejected scalar/accessor/rank/mutability combinations.
2. **Runtime view-representation tests**
   - Verify `layout_left`, `layout_right`, and `layout_stride` descriptors.
   - Check submatrix and padded views where one stride remains `1`.
   - Reject views where neither stride is `1`.
3. **Numerical backend tests**
   - Compare mdspan LAPACK wrappers against current Krylov dense helpers.
   - Exercise `float`, `double`, `uni20::complex<float>`,
     `uni20::complex<double>`, and binary128 types when MPLAPACK is enabled.
   - Include row-major-as-transposed tests only for operations where that
     semantic is explicitly supported.

For kernel dispatch itself, use fake backends first. That keeps dispatch
semantics independent of LAPACK availability and makes compile-time and runtime
decline behavior easy to test.

## Design Decisions To Preserve

- Leaf linalg kernels accept strided mdspan-like views.
- Backend type compatibility is compile-time.
- Backend instance representability is runtime.
- LAPACK requires raw host-addressable storage and supported scalar types.
- At least one matrix stride must be `1` before a direct LAPACK call is even a
  candidate.
- Provider selection remains inside `uni20::lapack`.
- `std::variant` dispatch is deferred.
- Strict direct `try_kernel(...)` wrappers do not hide layout copies. Higher
  linalg wrappers may use prepared-view helpers that own input scratch matrices
  or explicit output copy-back semantics.
- Materialization policy is part of the wrapper contract: direct-only for leaf
  dispatch, input temporaries for ergonomic value wrappers, and output copy-back
  only when explicitly requested or documented.

## Open Questions

- Which operations should support row-major-as-transposed views without copying?
- Which operations should support conjugate-only input transforms directly, and
  should an MKL backend lower those to the `'R'` extension when available?
- Which public wrappers should default to `input_temporaries`, and which should
  require `direct_only` unless the caller opts into materialization?
- What diagnostics should report materialization decisions: copied input,
  transpose rewrite, conjugation, output copy-back, and scratch size?
- Should mdspan convenience wrappers throw directly, or should all leaf wrappers
  expose both `try_*` and `*_or_throw` forms?
- What is the final namespace split between `uni20::linalg`, `uni20::kernel`,
  and `uni20::backend`?
- How much diagnostic formatting should be included in the first pass versus
  deferred until the tensor front end starts using backend lists?
- Should simple BLAS-like operations be added in this same layer, or should the
  first pass stay focused on LAPACK routines needed by Krylov?
