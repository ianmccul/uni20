# Uni20 Python Bindings

The `uni20` extension exposes a subset of the Uni20 C++ API to Python via [nanobind](https://github.com/wjakob/nanobind). This guide covers the prerequisites, build steps, and the current smoke-test workflow.

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
