# MPLAPACK Binary128 Setup

Uni20 can optionally use MPLAPACK as an experimental binary128 backend for
selected scalar, dense projected linear algebra, Krylov, and exponential-action
tests. This is an opt-in developer configuration. Uni20 does not download or
build MPLAPACK during its own configure step.

The recommended workflow is:

1. Build MPLAPACK once as a separate GNU++23 package with only the binary128
   backend enabled.
2. Install that package into a local prefix.
3. Configure Uni20 with `UNI20_ENABLE_MPLAPACK=ON` and point CMake at the
   installed MPLAPACK package.

## Build MPLAPACK

From a Uni20 checkout with an MPLAPACK checkout next to it, build current
MPLAPACK master or a release containing the binary128 CMake detection fixes:

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

## Configure Uni20

Use the installed MPLAPACK prefix:

```bash
cmake -S . -B ./build_codex/build_gcc13_debug_mplapack \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUNI20_ENABLE_MPLAPACK=ON \
  -DCMAKE_PREFIX_PATH="$PWD/build_codex/mplapack_binary128_install"
```

Pointing directly at the MPLAPACK build tree also works while developing
MPLAPACK itself:

```bash
cmake -S . -B ./build_codex/build_gcc13_debug_mplapack \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUNI20_ENABLE_MPLAPACK=ON \
  -Dmplapack_DIR="$PWD/build_codex/mplapack_binary128"
```

The package must provide both CMake targets:

```text
mplapack::mpblas_binary128
mplapack::mplapack_binary128
```

## Validate

Build and run the MPLAPACK-gated Uni20 probes:

```bash
cmake --build ./build_codex/build_gcc13_debug_mplapack --parallel 36 \
  --target \
    uni20_mplapack_binary128_tests \
    uni20_linalg_mplapack_binary128_tests \
    uni20_krylov_mplapack_binary128_tests

ctest --test-dir ./build_codex/build_gcc13_debug_mplapack \
  --output-on-failure \
  --parallel 36 \
  -R "MplapackBinary128"
```

## Validated Upstream State

On 2026-06-30 in the Asia/Taipei timezone, Uni20 was validated against
MPLAPACK master commit `308abcccd5798f56a5a3cb033a8af035886b8823`.
That checkout configured, built, installed, and passed Uni20's MPLAPACK-gated
binary128 tests without any local MPLAPACK patch.

Do not build MPLAPACK for Uni20 in GNU++17 mode. Current upstream autoconf and
the current CMake detector may build successfully in GNU++17 mode, but the
installed headers can record GNU++17 feature-test results. When those headers
are later consumed from Uni20's C++23 build, fallback overload decisions may not
match the consuming translation unit. Building MPLAPACK itself as GNU++23 keeps
the installed configuration aligned with Uni20.

## Troubleshooting

If Uni20 cannot find MPLAPACK, check the package path:

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

For the current Uni20 probes, `MPLAPACK_BINARY128_MODE_LDBL` is not useful; use
a real binary128 mode such as `_Float128` or quadmath.
