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

`DirectGemmContractionBackend` implements the first direct no-copy lowering. It
builds joint M, N, and K stride groups from normalized mdspec mappings. When
each group collapses to at most one dimension, it projects the operands to
rank-two `layout_stride` mdspecs without acquiring their handles and dispatches
`gemm_op` through the execution selector retained by the backend. The selected
GEMM backend therefore owns host or CUDA acquisition and provider lowering.
The projection retains each descriptor, accessor, and mapping, so default and
representable conjugating accessors use the same validation as direct GEMM.
An extent-one axis never blocks group merging because its reported stride is
unobservable; the merged group retains the non-singleton axis's strides.
Nonmergeable groups cleanly fall through to the next contraction backend.

`LoopedGemmContractionBackend` handles the next no-copy case. After joint
stride merging, exactly one of M or N may retain two descriptors while the
other groups contain at most one. The backend chooses one residual descriptor
as an outer loop and projects the other dimensions to rank-two mdspecs. Each
iteration advances copied operand metadata to a disjoint output slice and
dispatches `gemm_op` through the retained execution selector. Immediate
mdspans advance through their accessor offset policy; deferred descriptors must
provide `offset_by`, which preserves storage identity for later acquisition.
K is deliberately not looped in this checkpoint, so every slice uses the
original `beta` and no partial K accumulation state is required.

Both GEMM planners currently require each projected rank-two operand to have
strictly positive strides and expose a unit-stride axis. Negative strides need
origin rebasing, while a zero stride ordinarily denotes a broadcast dimension.
A zero-sized `layout_right` input can also report zero for a surviving outer
stride. Consequently a mathematically valid `K == 0` contraction may decline
during projection rather than reaching a GEMM backend. The current host backend
list then falls through to `CpuReferenceBackend`, which applies `C = beta*C`
without reading either input. A tensor with no unit-stride axis can otherwise
participate only when slicing away a residual M or N descriptor reveals a valid
matrix projection.

In `CpuReferenceBackend`, `alpha == 0` avoids input element reads and
`beta == 0` avoids reading output elements. Empty contracted extents therefore
produce the correctly scaled zero product without accessing the inputs. A
retained GEMM selector may handle this through its reference GEMM backend; a
provider-only direct backend can instead decline to the outer contraction
fallback.

## Execution Strategies

The mathematical operation and its axis descriptor do not select one physical
algorithm. Backends may implement the following hierarchy:

| Strategy | Intended role | Status |
|---|---|---|
| Generic indexed loop | Correctness path for arbitrary mappings and compatible accessors. | Implemented by `CpuReferenceBackend`. |
| Stride-grouped reference loop | Reduce mapping arithmetic for default-accessor strided views using merged M/N/K groups. | Existing historical code; not yet integrated into dispatch. |
| Direct GEMM | Collapse M, N, and K groups and dispatch one rank-two GEMM through the storage-selected execution backends. | Implemented by `DirectGemmContractionBackend`. |
| Looped GEMM | Loop over one residual M or N descriptor around rank-two GEMM dispatches. | Implemented by `LoopedGemmContractionBackend`; multiple residual dimensions and residual K remain planned. |
| Pack-GEMM-unpack | Materialize favorable grouped layouts when packing cost is justified. | Planned; requires temporary-storage and cost policy. |

Direct GEMM, looped GEMM, and packing are backend lowerings of `contract_op`; the Tensor
front end must not replace the semantic operation with transpose/reshape/GEMM.
This leaves storage placement, provider capabilities, future automatic
differentiation, and block-sparse scheduling above the correct abstraction.

Direct, looped, and packed implementations are ordinary backends that only
implement `contract_op`. They form an operation-specific selector axis rather
than permanent entries in every storage policy's general backend list.
`backend_selector_default<contract_op<...>, StoragePolicy>` installs the direct
and looped backends with copies of the storage selector they use for `gemm_op`,
then appends the original storage-selector entries as outer contraction
fallbacks. Host vector storage therefore reaches the CPU reference contraction
when GEMM strategies decline, while CUDA storage retains cuBLAS/CUDA execution
backends without changing either contraction planner.

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
3. Multiple residual M/N dimensions, residual-K accumulation, and batched GEMM
   contraction plans.
4. CUDA reference and provider implementations.
5. BlockTensor lowering in which symmetry-aware worklists submit these dense
   contractions without erasing block keys or leg metadata.
