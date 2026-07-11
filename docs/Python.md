# Uni20 Python Bindings

The `uni20` extension exposes a subset of the Uni20 C++ API to Python via [nanobind](https://github.com/wjakob/nanobind). This guide covers the prerequisites, build steps, and the current smoke-test workflow.

Importing the extension configures recoverable Uni20 `ERROR` diagnostics to throw exceptions process-wide. Nanobind translates exceptions crossing a synchronous binding boundary into Python exceptions. Internal invariant failures reported by `CHECK`, `PRECONDITION`, or `PANIC` still abort. Future asynchronous bindings must catch exceptions at their C++ task boundaries and deliver them through the Python-facing result rather than allowing them to escape a worker thread.

Future kernel bindings should lower through `dynamic_dispatch_kernel(...)` when a concrete operation may be unavailable in a particular build. Normal C++ `try_dispatch_kernel(...)` and `dispatch_kernel(...)` calls are intentionally ill-formed when their aggregate type probe is `no`; the dynamic boundary instead converts that static rejection into `ERROR`, and therefore into a Python exception. Python argument validation should still reject invalid user values before entering C++ code whose internal contracts are enforced by `CHECK` or `PRECONDITION`.

## Prerequisites

Before configuring the project ensure the following are available:

- A C++23-capable compiler such as GCC 13+, Clang 16+, or MSVC 19.36+.
- CMake 3.24 or newer.
- Python 3.8 or newer with interpreter and development headers. On Debian-based systems:
  ```bash
  sudo apt-get install python3-dev python3-venv
  ```

Uni20 reuses the same BLAS/LAPACK dependencies as the core C++ library. The configuration step discovers Python first and then resolves `nanobind` from a system installation or via CMake `FetchContent`.

## Configure and build

Enable the bindings when configuring:

```bash
cmake -S . -B build -DUNI20_BUILD_PYTHON=ON
```

Build the extension target directly, or build the default target graph:

```bash
cmake --build build --target uni20_python
```

The compiled module is written under `build/bindings/python/`. On Linux and macOS the filename follows the normal ABI-tagged extension naming convention such as `uni20.cpython-312-x86_64-linux-gnu.so`.

## Run the sample bindings

The sample module currently exports a `greet()` helper together with build metadata. Add the build output directory to `PYTHONPATH` and import it directly:

```bash
export PYTHONPATH="$(pwd)/build/bindings/python:${PYTHONPATH}"
python -c "import uni20; print(uni20.greet())"
```

Expected output:

```text
Hello from uni20!
```

## Build metadata

The Python `buildinfo()` function exposes the same generated metadata as the C++ `<uni20/buildinfo.hpp>` API:

```bash
python -c "import pprint, uni20; pprint.pp(uni20.buildinfo())"
```

For a formatted string:

```bash
python -c "import uni20; print(uni20.buildinfo_pretty())"
```

For the C++ API and pretty-print example, see [buildinfo.md](buildinfo.md).

## Presentation and notebook display roadmap

Current Python bindings expose only lightweight smoke-test functionality and build metadata. Future tensor bindings should use the common C++ presentation layer for human-facing text, but they should keep terminal rendering, plain text rendering, and notebook HTML rendering as separate adapters.

Design rules for future Python display:

- `repr(obj)` should be stable, plain, and safe for large tensors.
- `str(obj)` and `print(obj)` may use presentation defaults, but should avoid surprising exhaustive output.
- `_repr_html_()` or `_repr_mimebundle_()` should be used for rich Jupyter display when available.
- `obj.pretty(...)` should expose explicit controls such as `width`, `glyphs`, `charset`, `color`, selected axes, precision, and preview limits.
- Environment variables such as `UNI20_GLYPHS`, `UNI20_CHARSET`, `UNI20_COLOR`, `NO_COLOR`, and `COLUMNS` should provide defaults, but per-call Python overrides should win.

Tensor and mdspan display in Python must be preview-first. Do not bind a tensor `repr` to the current exhaustive mdspan formatter by default. Add an explicit preview policy first, with limits such as maximum elements, edge items, maximum rows/columns, maximum slices, and an opt-in full-output mode.

Future tensor bindings should also follow a documented dtype policy before exposing arithmetic. See [Python Dtype Promotion Policy](python_dtype_promotion.md) for the current design note.

## Running tests

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

## Packaging status

The repository currently builds the extension through CMake only. It does not yet ship a `pyproject.toml` or wheel-building backend, so `pip install .` style packaging is not documented here.
