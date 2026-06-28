# CUDA Runtime Resolution Strategy

Status: design note, not current implemented behavior. This note complements
[`cuda_backend_libraries.md`](cuda_backend_libraries.md): that document describes
runtime capability probes and backend fallback, while this one describes how the
dynamic loader finds CUDA-adjacent shared libraries in the first place. The
runtime fallback discussion here uses the `kernel_maybe_can(...)` /
`try_kernel(...)` split from [`kernel_dispatch.md`](kernel_dispatch.md).

## Problem

`CMAKE_CUDA_COMPILER` selects the CUDA toolkit used at configure and link time,
but it does not by itself control which shared libraries are loaded at runtime.
On Linux, linking against a CUDA library usually records only the SONAME, such as
`libcudart.so.12`, `libcublas.so.12`, or `libcusolver.so.12`. The dynamic loader
then resolves those names using rpath/runpath, `LD_LIBRARY_PATH`, and the system
library cache.

This means a build against `/usr/local/cuda-12` can silently load CUDA libraries
from an apt installation in `/usr/lib/x86_64-linux-gnu` if the installed
extension or executable does not carry an appropriate runtime search path.

The same issue applies to optional NVIDIA libraries such as cuTENSOR and
cuTensorNet/cuQuantum.

## Loader Policy Versus Backend Fallback

There are two different failure boundaries:

- **Loader/runtime-stack resolution.** If the dynamic loader finds an incompatible
  CUDA, cuBLAS, cuSOLVER, cuTENSOR, or cuTensorNet library, Uni20 may fail before
  any Uni20 diagnostic code runs. This can be a hard import-time or startup
  failure.
- **Kernel/backend capability.** If the process starts successfully and a backend
  can probe the loaded libraries, the kernel-dispatch layer can mark an optimized
  backend unavailable at runtime and use the next eligible backend in the ordered
  list.

Uni20 should handle both. The loader policy should make accidental library mixing
unlikely. Runtime probes should then make known incompatibilities visible to
dispatch, so optional CUDA-adjacent backends can be avoided at runtime where a
fallback exists. If a user forces a backend or requests an operation that promises
that backend, an incompatible runtime should fail with a clear diagnostic.

In kernel-dispatch terms:

- `kernel_maybe_can(...)` checks type-level facts, such as whether operands are
  CUDA-resident, scalar types are supported by the backend family, and the
  operation shape is meaningful for that backend.
- `try_kernel(...)` checks runtime facts before side effects, such as the loaded
  library version, active device architecture, handle/context availability,
  peer-access availability, and backend-specific capability probes.

A version conflict should therefore usually be a `try_kernel(...) == false`
runtime decline, not a hard failure, when a later backend in the ordered list can
provide the requested Uni20 semantics. A forced one-entry backend list, or an API
that promises a specific backend, has no semantic fallback and should fail loudly
when `try_kernel(...)` declines.

## Strategy

Separate the policies for local developer builds and distributable packages.

For local developer builds, it is acceptable to pin the selected toolkit with an
absolute rpath. This makes debugging predictable and avoids accidental mixing
between an apt CUDA install and a tarball CUDA install.

For relocatable packages and wheels, do not rely on absolute build-machine paths.
Use one of:

- package-repaired relative rpaths for bundled shared libraries;
- relative rpaths to runtime libraries supplied by package dependencies;
- a documented runtime-library discovery/loading policy;
- an explicit `LD_LIBRARY_PATH` workaround for unmanaged local installs.

Do not assume that `INSTALL_RPATH_USE_LINK_PATH TRUE` is universally correct. It
can be appropriate for a local install, but it is usually the wrong default for a
relocatable wheel unless the wheel repair step rewrites the paths.

## RPATH Versus RUNPATH

Modern Linux linkers usually emit `DT_RUNPATH` by default. `RUNPATH` applies to
direct dependencies, but not necessarily to indirect dependencies. CUDA libraries
often bring in transitive dependencies, so a partial runpath can still produce a
mixed runtime stack.

For package builds, every bundled ELF object that has non-system dependencies may
need its own `$ORIGIN`-relative runpath after repair. Setting the Python extension
module's runpath is not enough if a bundled helper library or vendor library then
loads its own dependencies through ambient system paths.

For tightly pinned local builds, `DT_RPATH` via `-Wl,--disable-new-dtags` may be
useful because it also affects transitive dependency resolution. Treat this as an
explicit local reproducibility mode, not a default package policy. Old-style
`RPATH` can also make `LD_LIBRARY_PATH` overrides ineffective or surprising, so it
is a poor default when users need an escape hatch.

## Runtime Diagnostics

Runtime checks are useful, but they are diagnostics, not a substitute for loader
policy. If the wrong library causes an import-time or load-time symbol failure,
Uni20 code may never run.

When runtime initialization does succeed, report enough information to diagnose
mismatches:

- configured CUDA toolkit root and compile-time CUDA version;
- configured library paths or package dependency source, when known;
- `cudaRuntimeGetVersion()` and `cudaDriverGetVersion()`;
- cuBLAS/cuSOLVER versions when those libraries are enabled;
- `cutensorGetVersion()` when cuTENSOR is enabled;
- `cutensornetGetVersion()` and `cutensornetGetCudartVersion()` when cuTensorNet
  is enabled;
- on Linux, `dladdr()` paths for representative symbols from each loaded shared
  library.

The path check is important because two libraries can have compatible major
versions while still coming from different toolkit roots.

## cuTENSOR And cuTensorNet

Keep cuTENSOR and cuTensorNet policy separate.

cuTENSOR is the dense tensor algebra library. It can be used directly for dense
contractions and permutations.

cuTensorNet is part of cuQuantum. It provides tensor-network operations such as
contraction planning and tensor QR/SVD. It has its own version line, but it also
depends on cuTENSOR.

Observed compatibility boundary from the local GV100/SM 7.0 investigation:

- cuTENSOR supports SM 7.0 through the 2.2.x line.
- cuTENSOR 2.3.0 removes SM 7.0 support.
- recent cuTensorNet/cuQuantum releases require newer cuTENSOR and start at
  compute capability 7.5.
- older cuTensorNet versions did support SM 7.0, including the tensor
  SVD/truncation APIs, but that requires pinning an older compatible stack.

Do not conflate:

- dropping CUDA 11;
- requiring cuTENSOR >= 2.3;
- dropping SM 7.0 / V100 / GV100.

These are separate policy decisions.

## Practical Default

For Uni20, prefer this default:

- local developer builds may pin absolute CUDA/cuTENSOR/cuTensorNet library
  paths;
- package builds must be relocatable and should use relative rpaths, repaired
  wheels, or explicit runtime dependency policy;
- runtime initialization should emit clear diagnostics when the loaded libraries
  do not match the configured expectations;
- `try_kernel(...)` runtime checks should mark known incompatible combinations
  unavailable before side effects, especially SM 7.0 with cuTENSOR >= 2.3 or
  recent cuTensorNet;
- forced backends and backend-promising operations should hard-fail with a clear
  diagnostic when the required runtime stack is incompatible.
