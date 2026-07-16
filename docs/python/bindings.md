# Uni20 Python Bindings

Status: current implementation and build guide.

The current `uni20` extension is a nanobind smoke module. It exposes `greet()`
and generated build metadata; it does not yet expose Tensor values, numerical
operations, NumPy or DLPack interop, or native async values.

## Error Mode

Importing the extension configures recoverable Uni20 `ERROR`, `ERROR_IF`, and
`trace::raise(...)` paths to throw exceptions process-wide. Nanobind translates
exceptions crossing a synchronous binding boundary into Python exceptions.

Internal invariant failures reported by `CHECK`, `PRECONDITION`, or `PANIC`
still abort. Future Tensor bindings must validate user-controlled values before
entering native paths whose contracts assume valid input. Future asynchronous
bindings must retain exceptions in native async state and deliver them through
the Python result instead of allowing them to escape a worker thread.

Kernel bindings should use `dynamic_dispatch_kernel(...)` where an operation
may be absent from a configured build. This preserves the native compile-time
dispatch contract while translating an unavailable binding instantiation into
`KernelDispatchError`.

## Future Binding Boundary

The Python layer will bind Uni20 operations; it will not define a parallel
tensor, storage, device, dispatch, or scheduler architecture.

The native contracts remain authoritative:

- `Tensor` and `TensorView` define values, views, ownership, accessors, and
  storage-derived backend selection.
- `Async<T>` defines asynchronous value and alias semantics.
- mdspan accessors define the values observed through a view. A pointer-shaped
  data handle alone does not permit direct NumPy, DLPack, BLAS, or LAPACK
  access.
- Python-visible views must retain the native owner needed to keep their data
  alive.
- Python-owned objects retained by asynchronous work need an interpreter-safe
  destruction path.

The representation of arbitrary-rank tensors, NumPy/DLPack API spelling,
device exposure, and Python async API remain open design questions. They should
be decided when implementing the corresponding binding, against the then
current native API and protocol versions.

See [Tensor Operations](../tensor/operations.md),
[Async Storage](../async/storage.md), and
[Kernel Dispatch](../architecture/kernel_dispatch.md).

## Prerequisites

Before configuring the project ensure the following are available:

- A supported C++23 compiler: GCC 13 or newer, or upstream Clang 19 or newer.
- CMake 3.24 or newer.
- Python 3.11 or newer with development-module headers. On Debian-based systems:

  ```bash
  sudo apt-get install python3-dev python3-venv
  ```

Uni20 reuses the same BLAS/LAPACK dependencies as the core C++ library. The
configuration step discovers Python first and then requires nanobind 2.13.0 or
newer from a system installation, falling back to nanobind v2.13.0 through
CMake `FetchContent`.

## Configure and Build

Enable the bindings when configuring:

```bash
cmake -S . -B build -DUNI20_BUILD_PYTHON=ON
```

Build the extension target directly, or build the default target graph:

```bash
cmake --build build --target uni20_python
```

The compiled module is written under `build/bindings/python/`. On Linux and macOS the filename follows the normal ABI-tagged extension naming convention such as `uni20.cpython-312-x86_64-linux-gnu.so`.

## Run the Module

The sample module currently exports a `greet()` helper together with build metadata. Add the build output directory to `PYTHONPATH` and import it directly:

```bash
export PYTHONPATH="$(pwd)/build/bindings/python:${PYTHONPATH}"
python -c "import uni20; print(uni20.greet())"
```

Expected output:

```text
Hello from uni20!
```

The [Python example](../../examples/python/) provides the same setup as a
runnable script.

## Build Metadata

The Python `buildinfo()` function exposes the same generated metadata as the C++ `<uni20/buildinfo.hpp>` API:

```bash
python -c "import pprint, uni20; pprint.pp(uni20.buildinfo())"
```

For a formatted string:

```bash
python -c "import uni20; print(uni20.buildinfo_pretty())"
```

For the C++ API and pretty-print example, see
[Build Information](../development/build_information.md).

## Presentation Boundary

Future Tensor bindings use the common C++ presentation data for human-facing
output while keeping terminal text, plain text, and notebook HTML as separate
renderers.

Design rules for future Python display:

- `repr(obj)` should be stable, plain, and safe for large tensors.
- `str(obj)` and `print(obj)` may use presentation defaults, but should avoid surprising exhaustive output.
- `_repr_html_()` or `_repr_mimebundle_()` should be used for rich Jupyter display when available.
- `obj.pretty(...)` should expose explicit controls such as `width`, `glyphs`, `charset`, `color`, selected axes, precision, and preview limits.
- Environment variables such as `UNI20_GLYPHS`, `UNI20_CHARSET`, `UNI20_COLOR`, `NO_COLOR`, and `COLUMNS` should provide defaults, but per-call Python overrides should win.

Tensor and mdspan display in Python must be preview-first. Do not bind a tensor `repr` to the current exhaustive mdspan formatter by default. Add an explicit preview policy first, with limits such as maximum elements, edge items, maximum rows/columns, maximum slices, and an opt-in full-output mode.

Future arithmetic bindings should follow the
[Python Dtype Promotion](dtype_promotion.md) design note rather than acquiring
promotion behavior accidentally from nanobind conversions.

## Run Tests

The Python bindings ship with lightweight smoke tests that import the compiled extension and validate both `greet()` and the generated build information:

```bash
cmake --build build --target uni20_python
ctest --test-dir build --output-on-failure -R "python.bindings"
```

If you prefer to run a smoke test directly, pass the directory containing the compiled extension so the test can add it to `sys.path`:

```bash
python tests/python/test_greet.py build/bindings/python
```

Adjust the second argument if your build directory differs.

## Packaging Status

The repository currently builds a top-level `uni20` extension through CMake.
It does not yet ship a `pyproject.toml`, wheel-building backend, generated
stubs, or packaged Python distribution.
