# MPLAPACK Binary128 Setup

Uni20 can optionally use MPLAPACK 3.0.0 or newer as its binary128 BLAS/LAPACK
backend. This enables `uni20::float128`, its complex counterpart, and the
scalar-generic tensor, BlockTensor, Krylov, and tensor-network paths backed by
those provider operations.

The default dependency mode is `AUTO`: CMake prefers a compatible installed
MPLAPACK package with the `binary128` component and otherwise fetches the
MPLAPACK 3.0.0 release. Configure the ordinary fetched path with:

```bash
cmake -S . -B ./build_codex/build_gcc13_debug_mplapack \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUNI20_ENABLE_MPLAPACK=ON
```

The fetched build enables only the reference binary128 backend. MPLAPACK's
other scalar backends, optimized duplicate library, examples, tests,
benchmarks, CUDA, and OpenCL targets remain disabled.

## Dependency Selection

Use `UNI20_USE_SYSTEM_MPLAPACK` to select the normal Uni20 dependency modes:

| value | behavior |
| --- | --- |
| `AUTO` | Prefer an installed MPLAPACK 3.0.0 or newer package with `binary128`; otherwise fetch `v3.0.0`. |
| `ON` | Require a compatible installed package and fail if it is unavailable. |
| `OFF` | Always fetch the pinned `v3.0.0` source. |

MPLAPACK 3.0.0 exposes one self-contained library per precision. Uni20 consumes:

```text
mplapack::mplapack_binary128
```

The pre-release split between separate `mpblas_binary128` and
`mplapack_binary128` libraries is not part of the supported package contract.

## Optional System Build

To provide a system package explicitly, build MPLAPACK 3.0.0 or newer as
GNU++23 with only the required backend:

```bash
cmake -S ../mplapack -B ./build_codex/mplapack_binary128 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build_codex/mplapack_binary128_install" \
  -DMPLAPACK_CXX_STANDARD=23 \
  -DMPLAPACK_CXX_EXTENSIONS=ON \
  -DMPLAPACK_ENABLE_DOUBLE=OFF \
  -DMPLAPACK_ENABLE_MPFR=OFF \
  -DMPLAPACK_ENABLE_GMP=OFF \
  -DMPLAPACK_ENABLE_QD=OFF \
  -DMPLAPACK_ENABLE_DD=OFF \
  -DMPLAPACK_ENABLE_BINARY80=OFF \
  -DMPLAPACK_ENABLE_BINARY128=ON \
  -DMPLAPACK_ENABLE_OPT=OFF \
  -DMPLAPACK_ENABLE_CUDA=OFF \
  -DMPLAPACK_ENABLE_OPENCL=OFF \
  -DMPLAPACK_BUILD_EXAMPLES=OFF \
  -DMPLAPACK_BUILD_TESTS=OFF \
  -DMPLAPACK_BUILD_BENCHMARKS=OFF

cmake --build ./build_codex/mplapack_binary128 --parallel
cmake --install ./build_codex/mplapack_binary128
```

The configure output should report a real binary128 backend, for example:

```text
binary128: _Float128 + strfromf128
```

Then require that installed package when configuring Uni20:

```bash
cmake -S . -B ./build_codex/build_gcc13_debug_mplapack \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUNI20_ENABLE_MPLAPACK=ON \
  -DUNI20_USE_SYSTEM_MPLAPACK=ON \
  -DCMAKE_PREFIX_PATH="$PWD/build_codex/mplapack_binary128_install"
```

Pointing directly at the MPLAPACK build tree also works while developing
MPLAPACK itself:

```bash
cmake -S . -B ./build_codex/build_gcc13_debug_mplapack \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUNI20_ENABLE_MPLAPACK=ON \
  -DUNI20_USE_SYSTEM_MPLAPACK=ON \
  -Dmplapack_DIR="$PWD/build_codex/mplapack_binary128"
```

## Validate

Build and run the MPLAPACK-gated Uni20 probes:

```bash
cmake --build ./build_codex/build_gcc13_debug_mplapack --parallel 36 \
  --target \
    uni20_mplapack_binary128_tests \
    uni20_linalg_mplapack_binary128_tests \
    uni20_krylov_mplapack_binary128_tests \
    spin_half_heisenberg_dmrg_example

ctest --test-dir ./build_codex/build_gcc13_debug_mplapack \
  --output-on-failure \
  --parallel 36 \
  -R "MplapackBinary128|SpinHalfHeisenbergDmrg.*Float128"
```

## Supported Upstream State

Uni20 requires the public MPLAPACK 3.0.0 package surface. The fetched dependency
is pinned to tag `v3.0.0`; compatible installed packages must report version
3.0.0 or newer and provide the `binary128` component.

Do not build a system MPLAPACK package for Uni20 in GNU++17 mode. The package
may build successfully in GNU++17 mode, but the
installed headers can record GNU++17 feature-test results. When those headers
are later consumed from Uni20's C++23 build, fallback overload decisions may not
match the consuming translation unit. Building MPLAPACK itself as GNU++23 keeps
the installed configuration aligned with Uni20. The fetched configuration sets
this automatically.

## Troubleshooting

If `UNI20_USE_SYSTEM_MPLAPACK=ON` cannot find MPLAPACK, check the package path:

```bash
find ./build_codex/mplapack_binary128_install -name mplapackConfig.cmake
```

Then pass either the install prefix through `CMAKE_PREFIX_PATH` or the directory
containing `mplapackConfig.cmake` through `mplapack_DIR`.

If Uni20 configures but binary128 wrappers are disabled or tests are skipped,
inspect the generated MPLAPACK configuration:

```bash
rg "MPLAPACK_BINARY128_MODE" \
  ./build_codex/mplapack_binary128/include/mplapack_config.h \
  ./build_codex/mplapack_binary128_install/include/mplapack/mplapack_config.h
```

`MPLAPACK_BINARY128_MODE_LDBL` is not accepted as Uni20's binary128 backend;
use a real binary128 mode such as `_Float128` or quadmath.
