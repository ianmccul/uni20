# Uni20 Python Bindings

The `uni20` extension exposes a subset of the Uni20 C++ API to Python via [nanobind](https://github.com/wjakob/nanobind). This guide explains the prerequisites, build steps, and example usage for the module.

## Prerequisites

Uni20 relies on CMake to orchestrate the build and reuses the same system dependencies as the C++ library. Before configuring the project ensure the following packages are available:

- A C++23-capable compiler (GCC 13+, Clang 16+, or MSVC 19.36+).
- CMake 3.24 or newer.
- Python 3.8 or newer with the development headers and libraries. On Debian-based systems install them via:
  ```bash
  sudo apt-get install python3-dev python3-venv
  ```

## Configure the build

Enable the Python bindings when running `cmake`. You only need to configure once unless you change toolchains or cache variables.

```bash
cmake -S . -B build -DUNI20_BUILD_PYTHON=ON
```

The configuration step locates your Python interpreter and development headers. If a system-wide nanobind installation exists it will be reused; otherwise Uni20 fetches a copy automatically.

## Build the module

Compile both the C++ library and the Python extension:

```bash
cmake --build build --target uni20
```

CMake emits the loadable extension to `build/bindings/python/`. On Linux and macOS the file is named `uni20.cpython-<abi>.so`; on Windows it is `uni20.cp<abi>.pyd`.

To rebuild after making source changes rerun the same build command. Ninja and other generators only recompile files that changed.

## Run the sample bindings

The sample module currently exports a single `greet()` function. Add the build output directory to `PYTHONPATH` and import it directly:

```bash
export PYTHONPATH="$(pwd)/build/bindings/python:${PYTHONPATH}"
python -c "import uni20; print(uni20.greet())"
```

The script should print:

```
Hello from uni20!
```

Alternatively, you can install the extension into a virtual environment using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install build
python -m build --wheel --outdir dist bindings/python
pip install dist/uni20-*.whl
python -c "import uni20; print(uni20.greet())"
```

## Running tests

After building you can run the test suite via CTest. The Python bindings currently ship with a lightweight smoke test that imports the extension and verifies the `greet()` helper returns the expected string:

```bash
cmake --build build
cd build/tests
ctest --output-on-failure -R python.bindings
```

Running `ctest --output-on-failure` from the same directory executes the Python smoke test alongside the native GoogleTest suites.

If you prefer to run the Python smoke test directly, invoke the test module and pass the directory containing the compiled extension so it can be added to `sys.path`:

```bash
python tests/python/test_greet.py build/bindings/python
```

Adjust the second argument if your build directory differs; for example, pass `out/bindings/python` when using `-B out` during configuration.
