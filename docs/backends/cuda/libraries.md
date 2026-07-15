# CUDA Backend Library Compatibility

**Status:** background provider and version survey for future CUDA backends, not
current Tensor backend behavior.

These notes record how uni20 should handle CUDA-adjacent vendor libraries such
as cuBLAS, cuSOLVER, cuTENSOR, cuTensorNet, cuStateVec, and cuDensityMat.
See also [`runtime_resolution.md`](runtime_resolution.md)
for the loader/rpath policy that determines which shared libraries are found
before these runtime probes can run.

The main rule is that library discovery is not enough.  A CUDA backend adapter
must distinguish:

- whether headers for a particular API are available at compile time;
- which shared library is resolved at runtime;
- which semantic version that runtime library reports;
- which CUDA toolkit and driver ABI the library was built for;
- whether the active CUDA device architecture is supported;
- whether the specific operation, scalar type, and compute mode are supported.

For example, a library can be a valid CUDA 12 library and still be unusable on
GV100 because the current release has dropped `sm_70` kernels.  This happened
locally with cuTENSOR 2.6 and cuQuantum 26.03 on Polaron.

## Compatibility Probes

Backend adapters should provide lightweight runtime probes.  These probes should
be explicit and cacheable; they should not run repeatedly on every operation.

Useful probe levels are:

- library version query, where the vendor API exposes one;
- handle/context creation on the active CUDA device;
- minimal plan creation for representative operation families;
- minimal execution for libraries where handle creation can succeed in a
  CPU-only or reduced-capability mode.

Handle creation is a useful first filter but is not always sufficient.  Some
libraries may create a handle while rejecting a particular tensor contraction,
factorization, precision mode, or JIT policy later.

The result should be represented as capabilities, not as a single boolean:

```text
library: cutensor
version: 2.2.0
cuda_abi: 12
device: sm_70
capabilities:
  dense_contraction_f64: yes
  block_sparse_contraction: unknown
  jit_kernels: no
```

Operations should dispatch through these capabilities and fall back when the
optimized backend is unavailable.

## Versioned Installations

uni20 should allow multiple vendor library versions to coexist.  This is needed
for older but still useful GPU architectures such as Volta, where the newest
library release may no longer support the device.

The preferred model is:

- CMake discovers candidate roots such as `CUTENSOR_ROOT` and `CUQUANTUM_ROOT`;
- each backend adapter records the include path, library path, and detected
  version used for a build;
- runtime diagnostics report the resolved shared library and vendor-reported
  version;
- tests can force a specific backend root or disable a backend entirely;
- fallback paths remain tested even when an optimized backend is present.

Do not assume that the system default `ldconfig` result is the desired backend.
For reproducible builds and benchmarks, explicit roots are better than ambient
library search order.

## Polaron GV100 Observations

These are empirical results from the dual GV100 Polaron machine.  They are not
a complete compatibility matrix.

Failed on `Quadro GV100 sm_70`:

- cuTENSOR 2.6.0.4, CUDA 12 package:
  `cutensorCreate()` returned `CUTENSOR_STATUS_NOT_SUPPORTED`.
- cuQuantum 26.03.2, CUDA 12 package:
  `cutensornetCreate()` returned `CUTENSORNET_STATUS_NOT_SUPPORTED`.

Worked on `Quadro GV100 sm_70`:

- cuTENSOR 2.2.0, CUDA 12 library:
  handle creation, double-precision contraction planning, execution, and a CPU
  reference check passed.
- cuQuantum 25.06.0, CUDA 12 libraries:
  `cutensornetCreate()`, `custatevecCreate()`, and `cudensitymatCreate()` all
  succeeded.

Current local roots used for the working archived stack:

```text
CUTENSOR_ROOT=/usr/local/cutensor-2.2.0
CUQUANTUM_ROOT=/usr/local/cuquantum-25.06.0
```

## Dispatch Implication

The backend-dispatch model should treat vendor libraries as optional
capabilities.  A public uni20 operation should not require cuTENSOR or
cuTensorNet unless the operation explicitly promises that backend.

The normal path should be:

```text
operation request
  -> compile-time API eligibility
  -> cached runtime capability check
  -> backend operation attempt
  -> semantic fallback when unavailable
```

This keeps Volta-era systems viable while still allowing newer machines to use
newer vendor libraries.
