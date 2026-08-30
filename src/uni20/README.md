# src/uni20

This directory contains the C++ source for the Uni20 library. Headers under
module subdirectories are developer-facing unless the owning module documents a
narrower boundary. Implementation details should stay inside the owning module
rather than leaking through the top-level namespace.

## Directory Map

Every active tracked source directory has a short local `README.md` that
explains ownership, important entry points, and where not to put cross-layer
logic. Keep these files brief; detailed design notes belong in `/docs`.

- [`core/`](core/): scalar concepts, scalar aliases, scalar traits, numeric limits, and
  small math/type utilities.
- [`common/`](common/): diagnostics, trace/check infrastructure, presentation helpers,
  and common containers/utilities.
- [`mdspan/`](mdspan/): configured mdspan integration, structural concepts,
  accessors, lazy transform views, stride helpers, and backend-neutral iteration
  plans.
- [`backend/`](backend/): backend-library wrappers and manifests. Current subdirectories
  include BLAS/LAPACK support, the CUDA runtime foundation, cuBLAS GEMM, and
  cuSOLVER scaffolding.
- [`kernel/`](kernel/): low-level tensor kernel entry points over resolved views. Kernel
  code should remain below tensor/symmetry semantics.
- [`linalg/`](linalg/): dense linear-algebra front ends and operation descriptors built on
  the backend/kernel layers, including dispatched elementwise operations.
- [`krylov/`](krylov/): matrix-free Krylov eigensolvers, dense projected subspace helpers,
  and Krylov/Taylor exponential-action algorithms.
- [`async/`](async/): epoch-ordered async runtime, schedulers, tasks, buffers, and async
  value/storage machinery.
- [`storage/`](storage/): storage abstractions used below tensor views and tensor objects.
- [`tensor/`](tensor/): owning dense tensors, tensor views, factories,
  and front-end operations.
- [`symmetry/`](symmetry/): quantum-number, block-space, U(1), and symmetry-factor
  infrastructure.
- [`tensor_network/`](tensor_network/): canonical MPS/MPO/center aliases and
  tensor-network algorithms built from BlockTensor operations.
- [`models/`](models/): concrete physical-model constructors that produce
  symmetry-aware finite MPS and MPO owners.

## Layering Notes

- Raw dense primitives operate on resolved storage/views and should not depend on
  quantum-number or block-sparse semantics. Symmetric tensor operations should
  lower into dense primitive programs after sector/block logic is resolved.
- Dense linear-algebra backends receive explicit operation tags and resolved
  mdspan-like views. Tensor, async, and symmetry layers own higher-level
  metadata such as storage policy, backend selection, and quantum numbers.
- Krylov algorithms are matrix-free in application space: they use opaque vector
  operations plus small dense projected problems rather than inspecting tensor
  storage directly.
- The generated `uni20/config.hpp` is part of the source include path during a
  configured build. Source headers that depend on configured options should
  include it explicitly.

## References

- Contributor and style rules: [`/AGENTS.md`](../../AGENTS.md)
- Project documentation index: [`/docs/README.md`](../../docs/)
- Architecture overview: [`/docs/architecture/overview.md`](../../docs/architecture/overview.md)
- Scalar policy: [`/docs/tensor/scalar_policy.md`](../../docs/tensor/scalar_policy.md)
- Backend dispatch notes: [`/docs/architecture/backend_dispatch.md`](../../docs/architecture/backend_dispatch.md)
- Kernel dispatch notes: [`/docs/architecture/kernel_dispatch.md`](../../docs/architecture/kernel_dispatch.md)
- Raw primitive lowering: [`/docs/symmetry/raw_primitives_and_lowering.md`](../../docs/symmetry/raw_primitives_and_lowering.md)
- Trace as reduction: [`/docs/linalg/trace_reduction.md`](../../docs/linalg/trace_reduction.md)
- Mdspan/LAPACK plan: [`/docs/linalg/mdspan_dispatch.md`](../../docs/linalg/mdspan_dispatch.md)
- Krylov algorithms: [`/docs/krylov/algorithms.md`](../../docs/krylov/algorithms.md)
