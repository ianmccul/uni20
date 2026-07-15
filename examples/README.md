# Uni20 Examples

This directory contains runnable programs that demonstrate implemented Uni20
APIs, diagnostics, numerical probes, and retained development experiments.
Examples are developer guidance and validation aids; tests and canonical docs
remain authoritative for exact contracts.

## Build And Run

Examples are enabled by default. An explicit configuration and target build is:

```bash
cmake -S . -B build -DUNI20_BUILD_EXAMPLES=ON
cmake --build build --target kernel_dispatch_example
./build/examples/kernel_dispatch_example
```

`examples/CMakeLists.txt` is the authoritative target map. Some examples are
also registered with CTest as executable documentation.

## Directory Map

- [AD](ad/README.md): early reverse-mode and gradient-descent experiments.
- [Async](async/README.md): buffers, await paths, schedulers, Tensor kernels,
  failures, and DAG diagnostics.
- [Common](common/README.md): build information, floating-point test helpers,
  and trace diagnostics.
- [Krylov](krylov/README.md): symmetric/nonsymmetric Matrix Market drivers and
  exponential-action probes.
- [Linear algebra](linalg/README.md): provider reporting, kernel dispatch, and
  Tensor GEMM.
- [Mdspan](mdspan/README.md): configured mdspan and formatting basics.
- [Presentation](presentation/README.md): reports, tables, glyph policies,
  text art, diagnostics, and mdspan previews.
- [Python](python/README.md): Python build-information smoke example.

## Failure Examples

Several programs deliberately fail or abort to demonstrate diagnostics. Their
local README identifies this behavior; do not use a zero exit status as the
expected result for those targets.

See [Getting Started](../docs/getting_started.md) for the general build workflow
and the [documentation index](../docs/README.md) for subsystem contracts.
