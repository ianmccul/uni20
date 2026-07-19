# CUDA Examples

- `cuda_hello_world_example.cpp` reports whether CUDA was enabled in the
  current Uni20 build. CUDA-enabled builds show the compile-time, runtime, and
  driver versions, enumerate visible devices, display their cached hardware
  capabilities, and exercise the stream/completion/idle-stream-pool
  foundation. CPU-only builds show the CMake option and runtime requirements
  needed to enable the CUDA path.

Build and run the example with:

```bash
cmake -S . -B build-cuda -DUNI20_ENABLE_CUDA=ON
cmake --build build-cuda --target cuda_hello_world_example
./build-cuda/examples/cuda_hello_world_example
```

See the [examples index](../), the introductory [CUDA Buffers
guide](../../docs/backends/cuda/buffers.md), and the [CUDA backend documentation
index](../../docs/backends/cuda/).
