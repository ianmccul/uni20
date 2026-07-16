# Mdspan Linear Algebra Dispatch Plan

**Status:** implemented first vertical slices plus forward plan. The mdspan
linalg dispatch layer now covers GEMM/GEMV, accessor-respecting CPU fallbacks,
full inner-product and stable-norm reductions, Tensor output-shape preparation,
matrix products, explicit copy/materialization,
matrix initialization and exponential front ends, and the LAPACK projected
eigensystem/Schur operations used by native Krylov. Broader BLAS/LAPACK
coverage and domain-aware prepared operands remain design work.

This note records the implemented mdspan wrappers over the existing Uni20
BLAS/LAPACK backend layer and the next steps for extending them. These wrappers
are the first real consumer of the kernel-dispatch design in
[`../architecture/kernel_dispatch.md`](../architecture/kernel_dispatch.md).

The static CPO name in this note is `kernel_accepts_types(...)`. It returns a
`KernelTypeAcceptance` value (`no`, `maybe`, or `yes`) for the C++ argument
types. It does not inspect extents, strides, pointer values, library
availability, or device state. Those checks belong to the runtime attempt,
`try_kernel(...)`.

Related notes:

- [`../architecture/backend_dispatch.md`](../architecture/backend_dispatch.md) - compile-time capability,
  runtime attempt, and fallback contract.
- [`../architecture/kernel_dispatch.md`](../architecture/kernel_dispatch.md) - operation tags and ordered
  backend lists.
- [`blas_lapack_wrappers.md`](blas_lapack_wrappers.md) -
  first concrete mdspan BLAS/LAPACK wrapper layer above the existing provider
  facades.
- [`../tensor/dispatch_and_view_semantics_draft.md`](../tensor/dispatch_and_view_semantics_draft.md)
  - tensor/front-end operation roles versus resolved mdspan leaf kernels.
- [`../tensor_network/sparse_matrix.md`](../tensor_network/sparse_matrix.md) - sparse Matrix Market helpers and current matrix
  vocabulary.
- [`mplapack_binary128.md`](mplapack_binary128.md) - current
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
   - Converts extents, strides, storage orientation, view-derived readable
     transforms, and triangle flags to the vendor wrapper call.

Bare mdspan calls are allowed as leaf-kernel calls, but they cannot derive a
default Uni20 backend stack by themselves. They need an explicit backend
selector, or a concrete convenience wrapper such as "use LAPACK for this view".

## Mdspan Concepts

The first implementation refines the existing Uni20 mdspan concepts instead of
introducing unrelated concept names. Generic rank filters are not specific to
linalg and live with the mdspan concepts. The linalg layer should then add
scalar-family and host-addressability refinements.

Implemented generic mdspan refinement:

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

// From src/uni20/mdspan/conjugate_accessor.hpp:
// accessor_applies_conjugation_v<Accessor>
// mdspan_needs_conjugation_v<View>

template <class View, std::size_t Rank>
concept BlasScalarRankedView =
    HostRawAddressableRankedView<View, Rank> &&
    uni20::BlasScalar<
        std::remove_cv_t<typename std::remove_cvref_t<View>::element_type>>;

template <class View, std::size_t Rank>
concept LapackScalarRankedView =
    HostRawAddressableRankedView<View, Rank> &&
    uni20::LapackRealOrComplex<
        std::remove_cv_t<typename std::remove_cvref_t<View>::element_type>>;
```

The exact names can change, but the split matters:

- `StridedMdspan`: structural mdspan-like API with runtime strides.
- `RankedStridedMdspan<View, Rank>`: structural mdspan-like API plus a static
  rank requirement; this belongs in the generic mdspan layer and is implemented
  in `src/uni20/mdspan/concepts.hpp`.
- `LinalgMatrixView` and `LinalgVectorView`: rank-specific real-or-complex
  linalg operands.
- `RealLinalgMatrixView`, `ComplexLinalgMatrixView`,
  `RealLinalgVectorView`, and `ComplexLinalgVectorView`: longer names for
  operations that need a specific scalar family.
- `HostRawAddressableRankedView`: memory can be passed to a host pointer ABI.
  A pointer-like `data_handle_type` alone is not enough to prove this. Accessor
  semantics matter: a conjugating accessor can preserve the original pointer
  handle while changing the values returned by `access(...)`. Provider-facing
  direct concepts should therefore distinguish "storage handle can be passed to
  the ABI" from "the accessor is identity-like or otherwise explicitly lowered
  into provider transform metadata". The default direct-memory case is an
  mdspan with `stdex::default_accessor<T>`. Other accessors are direct only when
  the wrapper recognizes their semantics and lowers them deliberately. The BLAS
  adapter currently treats Uni20's C++26-style `conjugated_accessor` as
  lowerable for readable inputs because it maps to `MatrixTransform`; arbitrary
  transform accessors still require materialization or a generic path.
- `BlasScalarRankedView` and `LapackScalarRankedView`: element type is exactly
  supported by the configured BLAS or LAPACK layer. These are backend
  refinements, not base linalg concepts. Keep both names because BLAS and LAPACK
  coverage may diverge for extension scalar types.
- `accessor_applies_conjugation_v` and `mdspan_needs_conjugation_v` live in
  `src/uni20/mdspan/conjugate_accessor.hpp`. The generic mdspan `conj(...)`
  helper returns a read-only conjugating accessor view for complex mdspans,
  cancels to the const original view when applied twice, and is a no-op for
  non-complex values while still returning a const identity mdspan view. A
  `conj(Tensor)` returns a tensor view whose mdspan accessor advertises the
  same trait, so the mdspan-to-BLAS adapter discovers
  conjugation from the accessor instead of from tensor-specific side metadata.
  The accessor follows the C++26 `std::linalg::conjugated_accessor` direction
  in WG21 P3050R3, but Uni20 keeps `uni20::conj` as the value-level
  customization point instead of adopting `conj-if-needed` terminology.
- mutability constraints belong to each operation.
- Generic writable/LHS operands should be structural views with ordinary
  raw/default-style accessors. Component views such as `real(x)` and `imag(x)`
  are slice-like structural views with adjusted handles and strides, not proxy
  accessor adaptors. Writable proxy or semantic-transform accessors are special
  operation-specific cases that need an explicit assignment law and backend
  lowering. The general policy is in
  `docs/tensor/dispatch_and_view_semantics_draft.md`.

The implementation can add named mutable real/complex aliases such as
`MutableRealLinalgVectorView` when call sites become clearer, but the important
policy is that the short names are real-or-complex and the longer names state a
specific scalar family.

The compile-time check should reject device accessors, arbitrary transform
accessors, unsupported scalar types, rank mismatches, and const outputs before
the function body is instantiated. A transform accessor can still be eligible
when it exposes raw host storage and declares a BLAS-lowerable operation such as
accessor-level conjugation through a trait.

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
appears first. API tags and backend selectors appear before outputs; options
that are ordinary configuration values follow the required operands unless they
must prefix a parameter pack. Older draft documents may still contain
BLAS/LAPACK-style output-last examples; new code should use the prefix-tag,
output-first convention unless an external ABI boundary forces another order.

## Mdspan BLAS/LAPACK Wrapper Layer

The first concrete implementation target is a new mdspan wrapper layer above
the existing BLAS/LAPACK backend facades. The detailed plan lives in
[`blas_lapack_wrappers.md`](blas_lapack_wrappers.md).

The important split is:

- `src/uni20/backend/blas` and `src/uni20/backend/lapack` remain provider and
  ABI facades. They own vendor detection, LP64/ILP64 details, raw declarations,
  checked LAPACK status handling, and MPLAPACK provider overloads.
- the new `src/uni20/linalg` mdspan wrapper layer owns view interpretation:
  ranked mdspan concepts, mdspan-storage-first `MdspanMatrixStage` descriptors,
  derived provider-ready `BlasReadableMatrix` and `BlasWritableMatrix` objects,
  transform helpers, direct no-copy runtime acceptance or decline, prepared
  input temporaries, and operation-specific rewrites such as row-major GEMM
  output handling.
- future operation-tag dispatch wraps the mdspan wrappers; the descriptor
  helpers should be testable before the generic backend-list dispatcher exists.

This keeps dense linalg policy out of the raw provider wrappers while giving the
kernel-dispatch design a real, testable leaf-kernel boundary.

The staging object should store logical mdspan extents, the non-unit stride,
and the integer mdspan axis whose stride is `1`. It should not store a BLAS
transpose flag. Helpers derive the provider-ready BLAS rows, columns, leading
dimension, and readable-input transform from those storage facts plus accessor
metadata such as conjugation.

## Kernel Dispatch Interface

The mdspan linalg layer is the first concrete user of the operation-tag model.
The explicit-selector GEMM and GEMV vertical slices exist: direct mdspan BLAS
wrappers delegate through `try_kernel(BlasBackend, operation, ...)`, and the
dispatch walk falls back to `CpuReferenceBackend`. GEMV adds rank-one BLAS
increments and Tensor-to-mdspan forwarding before LAPACK workspace policy enters
the picture.

Backend CPOs put the backend value first, then the operation tag, then ordinary
reference parameters matching the public call order. A separate
`state_type<State>` / `backend_type<Backend>` tag layer is not needed for the
first pass.

```cpp
#include <uni20/linalg/operation_tags.hpp>

struct BlasBackend {};
struct CpuReferenceBackend {};

template <class C, class Alpha, class A, class B, class Beta>
consteval auto
kernel_accepts_types(BlasBackend const&, gemm_op const&, C&, Alpha const&,
                     A const&, B const&, Beta const&)
{
  if constexpr (/* host, same scalar, BLAS scalar, rank-2 writable/readable */) {
    return kernel_types_maybe; // strides and transforms are runtime
  } else {
    return kernel_types_no;
  }
}

template <class C, class Scalar, class A, class B>
KernelAttempt try_kernel(BlasBackend, gemm_op, C&& c, Scalar alpha, A&& a,
                          B&& b, Scalar beta)
{
  return uni20::linalg::blas::try_gemm(std::forward<C>(c), alpha,
                                      std::forward<A>(a), std::forward<B>(b),
                                      beta);
}
```

`try_kernel(...)` assumes dispatch has already obtained a non-`no` result from
`kernel_accepts_types(...)`; it does not repeat that type predicate.

The public mdspan dispatch entry point is the generic dispatcher:

```cpp
dispatch_kernel(backend_list{BlasBackend{}, CpuReferenceBackend{}},
                gemm_op{}, c, alpha, a, b, beta);
dispatch_kernel(BlasBackend{}, gemm_op{}, c, alpha, a, b, beta);
dispatch_kernel(CpuReferenceBackend{}, gemm_op{}, c, alpha, a, b, beta);
```

The first LAPACK wrapper can then use the same pattern with richer operand
rules:

```cpp
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/linalg/operation_tags.hpp>

template <class D, class E, class Z>
consteval auto
kernel_accepts_types(LapackBackend const&,
                     symmetric_tridiagonal_eigen_op const&,
                     D&, E&, Z&)
{
  if constexpr (mutable_lapack_candidate_real_vector<D> &&
                mutable_lapack_candidate_real_vector<E> &&
                mutable_lapack_candidate_matrix<Z>) {
    return kernel_types_maybe; // extents, strides, and aliasing are runtime
  } else {
    return kernel_types_no;
  }
}

template <class D, class E, class Z>
KernelAttempt try_kernel(LapackBackend,
                          symmetric_tridiagonal_eigen_op op,
                          D&& diagonal, E&& subdiagonal, Z&& eigenvectors)
{
  auto vectors = try_lapack_writable_matrix(eigenvectors);
  if (!vectors) {
    return KernelAttempt::unsupported_layout;
  }

  // Allocate LAPACK work and call sterf or steqr through the checked wrappers.
  return KernelAttempt::success;
}
```

The workspace query and allocation in this example are LAPACK work storage, not
operand materialization. The wrapper still counts as direct if it does not copy,
pack, transpose, or conjugate the user-visible matrix/vector operands.

`kernel_accepts_types(...)` should stay type-level. It should not inspect
runtime strides, sizes, pointer values, or backend state. The dispatcher
inspects its result type through unevaluated `decltype` and `std::declval`
expressions. The runtime attempt does the value checks and returns a specific
non-success `KernelAttempt` if the selected backend cannot represent this
particular view.

The acceptance constants encode the decision in their types, so the dispatcher
does not need synthetic objects, a tuple, or an explicit type-pack token:

```cpp
template <class Backend, class Op, class... Args>
consteval KernelTypeAcceptance backend_type_acceptance()
{
  using backend_type = std::remove_cvref_t<Backend>;
  using op_type = std::remove_cvref_t<Op>;
  if constexpr (requires {
                  kernel_accepts_types(std::declval<backend_type const&>(),
                                       std::declval<op_type const&>(),
                                       std::declval<std::remove_reference_t<Args>&>()...);
                }) {
    using result_type = std::remove_cvref_t<decltype(kernel_accepts_types(
        std::declval<backend_type const&>(), std::declval<op_type const&>(),
        std::declval<std::remove_reference_t<Args>&>()...))>;
    constexpr auto acceptance = result_type::value;
    if constexpr (acceptance == KernelTypeAcceptance::no) {
      return KernelTypeAcceptance::no;
    } else {
      static_assert(backend_has_try_kernel<backend_type, op_type, Args...>());
      return acceptance;
    }
  }

  return KernelTypeAcceptance::no;
}
```

`kernel_accepts_types(...)` owns type eligibility and may be narrowly
constrained. If it is not callable for the exact argument types, that backend is
a hard `no`; a callable `try_kernel(...)` does not change the result.
`try_kernel(...)` can therefore remain unconstrained, and the dispatcher does
not instantiate it for rejected types. This single-backend query remains a
detail helper. Public code uses
`probe_dispatch_kernel(backends, op, args...)`, which returns `yes` if any
candidate is `yes`, otherwise `maybe` if any is `maybe`, otherwise `no`.

## Implemented Example: Self-Adjoint Eigensolver

The standard dense symmetric/Hermitian eigensolver is now implemented. It
exercises real and complex scalar paths and shows the distinction between type
eligibility and runtime view representability.

### Public Shape

Use an output-first, in-place work-matrix signature:

```cpp
struct SelfAdjointEighOptions {
  bool compute_vectors = true;
  MatrixTriangle triangle = MatrixTriangle::Upper;
};

template <class W, class A>
void self_adjoint_eigh(W&& eigenvalues, A&& matrix_work,
                       SelfAdjointEighOptions options = {});

template <class BackendSelector, class W, class A>
  requires BackendSelectorLike<BackendSelector>
void self_adjoint_eigh(BackendSelector selector, W&& eigenvalues,
                       A&& matrix_work, SelfAdjointEighOptions options = {});
```

`matrix_work` is mutable because LAPACK overwrites its input. If
`compute_vectors` is true, `matrix_work` contains the eigenvectors on return.
If `compute_vectors` is false, `matrix_work` should be treated as destroyed
workspace. The higher-level `eigh(matrix)` value API has a preserving lvalue
form that copies into an explicit column-major temporary and a consuming owning
rvalue form that may reuse the input allocation:

```cpp
auto preserved = eigh(matrix);
auto consumed = eigh(std::move(matrix));
```

Reuse is valid because the matrix is a single update operand: LAPACK's `A`
argument is both the input matrix and, for `jobz = 'V'`, the eigenvector output.
For directly addressable column-major storage, the consuming wrapper transfers
the allocation unchanged. For square row-major storage, it adopts the same
allocation under a column-major `{1, LDA}` mapping, exchanges `Upper` and
`Lower`, and conjugates complex eigenvectors after LAPACK. This preserves a
padded leading dimension and any unused storage tail. A mapping without a
unit-stride axis is copied into column-major work storage instead.

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
consteval auto
kernel_accepts_types(LapackBackend const&, self_adjoint_eigh_op const&,
                     W&, A&)
{
  if constexpr (LapackSelfAdjointEighOperands<A, W>) {
    return kernel_types_maybe;
  } else {
    return kernel_types_no;
  }
}
```

This rejects, at compile time:

- rank-3 tensors accidentally passed as matrices.
- rank-2 matrices accidentally passed as eigenvalue vectors.
- const output views.
- integer or unsupported scalar types.
- complex eigenvalue output for a Hermitian problem.
- device accessors and arbitrary transform accessors that cannot be passed to a
  host pointer ABI. Accessor-declared conjugation is allowed for readable BLAS
  operands only when the accessor still exposes raw storage and the BLAS
  wrapper knows how to lower or materialize it.

### Runtime Representability

The strict first LAPACK implementation should require direct writable matrix
representation with unit stride along mdspan axis `0`:

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

In descriptor terms, this is `unit_stride_axis == 0` and
`nonunit_stride == stride(1)`. `stride(1) == 1` row-major-like matrices should
initially decline. Real
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
KernelAttempt try_kernel(LapackBackend, self_adjoint_eigh_op op,
                          W&& eigenvalues, A&& matrix_work)
{
  auto matrix_stage = try_blas_column_major_writable_stage(matrix_work);
  auto values = try_lapack_contiguous_vector(eigenvalues);
  if (!matrix_stage || !values) {
    return KernelAttempt::unsupported_layout;
  }

  auto const matrix_blas = blas_writable_matrix(*matrix_stage);
  blas_int const rows = matrix_blas.rows;
  blas_int const cols = matrix_blas.cols;
  CHECK_EQUAL(rows, cols);
  CHECK_EQUAL(values->size, rows);

  char const jobz = op.compute_vectors ? 'V' : 'N';
  char const uplo = lapack_triangle(op.triangle);
  blas_int const n = rows;
  blas_int const lda = matrix_blas.leading_dimension;

  using MatrixScalar = view_value_t<A>;
  using Real = view_value_t<W>;

  if constexpr (uni20::LapackReal<MatrixScalar>) {
    MatrixScalar work_query{};
    uni20::lapack::syev(jobz, uplo, n, matrix_blas.data, lda, values->data,
                        &work_query, -1);

    blas_int const lwork = checked_lwork(work_query);
    std::vector<MatrixScalar> work(static_cast<std::size_t>(lwork));
    uni20::lapack::syev(jobz, uplo, n, matrix_blas.data, lda, values->data,
                        work.data(), lwork);
  } else {
    MatrixScalar work_query{};
    std::vector<Real> rwork(heev_rwork_size(n));
    uni20::lapack::heev(jobz, uplo, n, matrix_blas.data, lda, values->data,
                        &work_query, -1, rwork.data());

    blas_int const lwork = checked_lwork(real_part(work_query));
    std::vector<MatrixScalar> work(static_cast<std::size_t>(lwork));
    uni20::lapack::heev(jobz, uplo, n, matrix_blas.data, lda, values->data,
                        work.data(), lwork, rwork.data());
  }

  return KernelAttempt::success;
}
```

The actual helper names can differ, but the responsibilities should stay split:

- `kernel_accepts_types(...)` says this backend can compile for these view
  types, returning `maybe` for direct LAPACK views because extents, strides,
  aliasing, and provider availability are runtime facts.
- `try_blas_column_major_writable_stage(...)` checks the particular extents,
  strides, and writable storage orientation.
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
- runtime decline for `unit_stride_axis == 1` row-major-like matrices until
  that case is explicitly implemented.
- eigenvalues agree with the current Krylov dense helper.
- eigenvectors, when requested, satisfy
  `||A v_i - lambda_i v_i|| / scale` within the scalar-specific tolerance.

## Error Semantics

Low-level `try_kernel(...)` functions return a specific non-success
`KernelAttempt` for clean runtime decline before side effects. Once a LAPACK
routine has been called, a terminal positive `INFO` raises a structured
`LapackError`; dispatch must not continue because update operands may already
have been overwritten. A negative `INFO` is an unconditional invariant failure:
it means the provider rejected arguments constructed by the checked Uni20
wrapper. Positive values explicitly documented as usable operation statuses
remain operation-specific return values. Decline categories are stable enum
values rather than backend-constructed strings.

Public convenience wrappers turn "all backends declined" into a structured
`KernelDispatchError`. Its presentation report lists the operation and each
backend's `no`/`maybe`/`yes` type acceptance, together with its structured
runtime result or `not eligible`. This records the dispatch decision without
requiring each backend to construct text.

Operand metadata such as extents, strides, scalar type, layout type, and
accessor kind may be added to optional structured diagnostics later.

## Initial Operation Set

Direct mdspan GEMM proves the BLAS matrix descriptor, transform helpers, and
row-major writable output rewrite. GEMV additionally proves rank-one descriptors
and mixed-rank dispatch. The first LAPACK mdspan wrappers cover the operations
needed by the active native Krylov projected problems. The remaining rows are
candidates to add only when an algorithm or example needs them:

| Operation | LAPACK routines | Status/reason |
| --- | --- | --- |
| symmetric tridiagonal eigensystem | `sterf`, `steqr` | Implemented for Lanczos projected diagonalization. |
| real and complex Schur decomposition | `gees`, `hseqr`, `trexc` | Implemented for Arnoldi restarts and reordering. |
| nonsymmetric eigensystem | `geev` | Implemented for projected Ritz extraction and validation. |
| symmetric/Hermitian dense eigensystem | `syev`, `heev`, then `syevd`/`heevd` | `syev`/`heev` implemented with in-place and preserving value APIs; divide-and-conquer variants remain future work. |
| QR/LQ factorization | `geqrf`, `orgqr`/`ungqr`, `gelqf`, `orglq`/`unglq` | Candidate for future dense utilities. |
| SVD and least squares | `gesvd`, `gesvdx`, `gelsd` | Candidate for future linalg API coverage. |

Do not expose every backend wrapper through mdspan immediately. Add wrappers
when a Uni20 algorithm or example needs them, and keep unused wrappers in a
separate tested header if we want to preserve prototype coverage.

## Implementation Phases

### Phase 1: Mdspan BLAS/LAPACK Wrapper Layer

Implement the wrapper layer described in
[`blas_lapack_wrappers.md`](blas_lapack_wrappers.md).

The completed BLAS slices are:

1. linalg-specific mdspan concept refinements;
2. mdspan-storage-first `MdspanMatrixStage` helpers, derived
   `BlasReadableMatrix`/`BlasWritableMatrix` helpers, and transform helpers;
3. direct mdspan GEMM over existing `uni20::blas::gemm`, including row-major
   writable output rewrite tests;
4. rank-one `MdspanVectorStage` and provider-ready vector operands;
5. direct mdspan GEMV, accessor-respecting CPU fallback, and fixed-output Tensor
   forwarding;
6. strict column-major writable descriptor helpers for LAPACK update operands.

The completed LAPACK slice adds checked workspace conversion and strict direct
column-major adapters for tridiagonal eigensystems, nonsymmetric eigensystems,
Schur decomposition, Hessenberg Schur reduction, and Schur reordering. Runtime
layout decline occurs before mutation; provider `INFO` failures are terminal.

This layer should call `uni20::blas::*` and `uni20::lapack::*`, never raw
Fortran symbols, and should keep strict direct wrappers non-owning and no-copy.
Prepared wrappers may later own input-only materialization by contract.

### Phase 2: Kernel Dispatch Skeleton

The generic pieces from `../architecture/kernel_dispatch.md` are implemented for the mdspan
dense-linalg layer:

- `backend_list<...>`.
- backend values, usually empty stateless tags in the first pass.
- `kernel_accepts_types(backend const&, op const&, args&...)` detection using
  private type-probe lvalues.
- ordered `try_dispatch_kernel(...)` returning whether a runtime candidate ran.
- checked `dispatch_kernel(...)`, which reports an exhausted list through
  `ERROR`.
- `dynamic_dispatch_kernel(...)` for Python and runtime-erased boundaries that
  must convert a type-level `no` into a runtime `ERROR`.
- a whole-selector entry-point constraint, not recursive-tail assertions.

Tests should use fake backends to verify:

- compile-time ineligible backend is skipped.
- runtime decline falls through to the next backend.
- forced single-backend dispatch reports a clean failure.
- a later eligible backend prevents a whole-list static assertion.

### Phase 3: Tensor Linalg Convenience API

Bare mdspans call generic dispatch directly:

```cpp
dispatch_kernel(LapackBackend{},
                symmetric_tridiagonal_eigen_op{.compute_vectors = true},
                diagonal, subdiagonal, eigenvectors);
dispatch_kernel(LapackBackend{}, schur_op{.compute_vectors = true},
                matrix_work, eigenvalues, schur_vectors);
```

Add operation-specific convenience functions only where the Tensor layer has
real policy work to perform, such as allocating results, selecting storage
backends, or resolving Tensor mdspans. Mutable outputs should appear first. For
in-place LAPACK-style operations, the overwritten work matrix is a mutable
output/update operand and should be grouped with the other output operands
before read-only inputs.

### Phase 4: DenseMatrix and Tensor Integration

This checkpoint is implemented. `uni20::DenseMatrix<T, LayoutPolicy>` is an
owning rank-2 alias of the concrete `Tensor` class, column-major by default,
and exposes its mdspan through the ordinary Tensor interface. The active Krylov
projected eigensystem, Schur, reorder, matrix-set, and matrix-exponential paths
now use matrix-level linalg front ends rather than raw pointer-level LAPACK
calls. Explicit row/column layout conversion remains where a Krylov algorithm
genuinely needs a differently ordered representation.

### Phase 5: Tensor Front-End Dispatch

The Tensor front-end checkpoints now:

- derive a default backend selector from tensor storage policy.
- enforce operation-specific ranked Tensor views.
- lower those operands through `mdspan()`; the generic CPU path does not
  require stridedness, while BLAS lowering does.
- use the same operation tag and backend-list walk as explicit mdspan calls.
- distinguish fixed `gemm`/`gemv` and `add_product` updates from resizable
  `assign_product` overwrites.
- provide lazy read-only `conj(tensor)` views and explicit eager `copy` and
  `make_tensor` operations. `make_tensor(conj(input))` allocates first and then
  dispatches `copy_op`.

Shape-changing operations call `ensure_shape(...)` before resolving the
writable mdspan. That policy does not belong in fixed-output GEMM or GEMV.

This is where CUDA, MPI/block tensor placement, async scheduling, and temporary
allocation policy enter. They should not be forced into the first LAPACK mdspan
wrapper step.

## Testing Strategy

Use three classes of tests:

1. **Compile-time concept tests**
   - Verify accepted and rejected scalar/accessor/rank/mutability combinations.
   - Verify default/raw accessors derive `mdspan_needs_conjugation_v == false`.
   - Use `uni20::conj(mdspan)` to verify accessor-derived conjugation metadata
     and double-conjugation cancellation.
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
- `conj(...)` remains a view. Explicit `copy_op` observes accessor and layout
  semantics; a future BLAS implementation may lower rank-two copy to provider
  `omatcopy`-style extensions without making conjugation eager.

## Open Questions

- Which operations should support row-major-as-transposed views without copying?
- Which BLAS providers and extension ABIs should implement rank-two `copy_op`
  for transpose/conjugate combinations (`omatcopy`, `omatcopy2`, or equivalent)?
- Which provider backends can support accessor-derived conjugate-only operands
  directly, for example through OpenBLAS `CblasConjNoTrans` or its
  Fortran-style `R` spelling, and where should fallback materialization live?
- Which prepared BLAS wrappers should use internal output-storage conjugation as
  a workaround for otherwise unavailable readable transform combinations?
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
