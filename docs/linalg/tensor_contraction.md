# Tensor Contraction

This document defines the current dense pairwise Tensor contraction contract
and separates it from future backend execution strategies. The November 2025
LaTeX paper in [`../latex/tensor_contraction.tex`](../latex/tensor_contraction.tex)
remains useful algorithmic background, but predates `TensorView`, `mdspec`, and
operation-tag dispatch.

## Implemented Contract

The first operation is a fixed-output update:

```cpp
linalg::contract(
    output,
    alpha,
    lhs,
    rhs,
    contracted_axes,
    beta);
```

It computes

```text
output = alpha * contract(lhs, rhs) + beta * output
```

For `N` contracted axis pairs, the output rank is
`rank(lhs) + rank(rhs) - 2*N`. Uncontracted left axes appear first in their
original order, followed by uncontracted right axes in their original order.
The front end rejects out-of-range axes, repeated axes, unequal paired extents,
an incorrectly shaped output, and an obvious same-object output/input alias.
Less obvious overlapping storage is a caller precondition.

Contraction is bilinear. It does not implicitly conjugate either input. A
caller requests conjugated values through an accessor-bearing view such as
`uni20::conj(lhs)`. Contracting all axes returns a rank-zero tensor but remains
distinct from the conjugate-linear-left `inner_product` operation. Contracting
no axes produces an outer product.

The initial scalar contract requires identical output, left, and right element
types. Scalar promotion is not inferred.

## Dispatch Boundary

`contract_op<LhsRank, RhsRank, ContractedRank>` carries normalized
`ContractionAxes`. Tensor storage policy remains visible while the front end
selects a backend. It then materializes each fixed operand exactly once through
`mdspec_of()` and dispatches:

```cpp
dispatch_kernel(
    selector,
    operation,
    output_mdspec,
    alpha,
    lhs_mdspec,
    rhs_mdspec,
    beta);
```

The dispatcher does not acquire data handles. The selected backend acquires
the execution-domain read and write leases and calls an ordinary resolved-view
implementation. No operation-tag redispatch occurs after acquisition.

`CpuReferenceBackend` accepts host-readable inputs and a host-writable output
when the acquired mdspan reference types support the required scalar
conversion, multiplication, accumulation, and assignment expressions. Every
element is evaluated through its accessor. Consequently conjugating and other
compatible semantic accessors are preserved.

`BlasBackend` implements the first direct no-copy provider lowering. It builds
joint M, N, and K stride groups from the normalized mdspec mappings before
acquiring host access. When each group collapses to at most one dimension, the
backend projects the acquired operands to rank-two `layout_stride` mdspans and
calls the ordinary BLAS GEMM leaf. The projection retains each resolved
accessor, so default and representable conjugating accessors use the same
transform validation as direct GEMM. Nonmergeable groups, unsupported matrix
strides or transforms, zero K extents, and unsupported scalar types cleanly
fall through to `CpuReferenceBackend`.

In `CpuReferenceBackend`, `alpha == 0` avoids input element reads and
`beta == 0` avoids reading output elements. Empty contracted extents therefore
produce the correctly scaled zero product without accessing the inputs. The
direct BLAS backend declines a zero K extent to this reference path.

## Execution Strategies

The mathematical operation and its axis descriptor do not select one physical
algorithm. Backends may implement the following hierarchy:

| Strategy | Intended role | Status |
|---|---|---|
| Generic indexed loop | Correctness path for arbitrary mappings and compatible accessors. | Implemented by `CpuReferenceBackend`. |
| Stride-grouped reference loop | Reduce mapping arithmetic for default-accessor strided views using merged M/N/K groups. | Existing historical code; not yet integrated into dispatch. |
| Direct GEMM | Collapse M, N, and K groups and call a provider when mappings and accessor transforms are representable. | Implemented by `BlasBackend`; cuBLAS remains planned. |
| Looped or batched GEMM | Loop over residual unmerged groups around provider GEMM calls. | Planned. |
| Pack-GEMM-unpack | Materialize favorable grouped layouts when packing cost is justified. | Planned; requires temporary-storage and cost policy. |

Direct GEMM and packing are backend lowerings of `contract_op`; the Tensor
front end must not replace the semantic operation with transpose/reshape/GEMM.
This leaves storage placement, provider capabilities, future automatic
differentiation, and block-sparse scheduling above the correct abstraction.

Direct, looped, and packed implementations may be ordinary backends that only
implement `contract_op`. They form an operation-specific selector axis rather
than permanent entries in every storage policy's general backend list. The
planned `backend_selector_default<contract_op<...>, StoragePolicy>` composes
that contraction list around `StoragePolicy::backend_selector()`, allowing the
same contraction planners to delegate lower GEMM and elementwise work to
BLAS/CPU or cuBLAS/CUDA execution backends.

An ordered contraction list initially provides priority-based runtime selection
through clean decline. A later cost-based selector may itself be a contraction
backend: it prepares each viable strategy without submitting work, compares
their runtime costs, and executes exactly one prepared plan. This does not
change the user override contract. A
`backend_selector_override<contract_op<...>, StoragePolicy>` specialization
replaces the complete Uni20 contraction list, for example to disable one
strategy as a temporary workaround.

## Forward Work

The next operation-level additions are:

1. A replaceable-output overwrite operation distinct from fixed-output
   contraction.
2. Async Tensor wrappers using the same normalized descriptor dispatch.
3. Looped or batched BLAS contraction plans for residual M/N/K groups.
4. CUDA reference and provider implementations.
5. BlockTensor lowering in which symmetry-aware worklists submit these dense
   contractions without erasing block keys or leg metadata.
