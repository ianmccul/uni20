#include <uni20/buildinfo.hpp>
#include <uni20/common/display.hpp>
#include <uni20/config.hpp>

#include <fmt/format.h>

#include <string>
#include <utility>

#if UNI20_BACKEND_CUDA
#include <uni20/backend/cuda/device.hpp>
#include <uni20/backend/cuda/runtime.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>
#endif

namespace
{

namespace presentation = uni20::presentation;

[[nodiscard]] std::string compiler_description()
{
  auto const info = uni20::build_info::current();
  return fmt::format("{} {}", info.cxx_compiler_id, info.cxx_compiler_version);
}

void append_build_fields(presentation::report_builder& report)
{
  auto const info = uni20::build_info::current();
  report.field("Uni20 build type", info.build_type.empty() ? "unspecified" : info.build_type)
      .field("C++ compiler", compiler_description());
}

void emit(presentation::report_builder report) { uni20::display::emit(std::move(report), uni20::display::stream::out); }

#if UNI20_BACKEND_CUDA

[[nodiscard]] std::string cuda_version(int encoded)
{
  if (encoded <= 0)
  {
    return "unavailable";
  }
  return fmt::format("{}.{}", encoded / 1000, encoded % 1000 / 10);
}

[[nodiscard]] std::string cuda_status(cudaError_t status)
{
  char const* const name = cudaGetErrorName(status);
  char const* const description = cudaGetErrorString(status);
  return fmt::format("{}: {}", name != nullptr ? name : "unknown CUDA error",
                     description != nullptr ? description : "no description available");
}

[[nodiscard]] std::string uuid_string(std::array<std::uint8_t, 16> const& uuid)
{
  std::string result;
  result.reserve(36);
  for (std::size_t i = 0; i < uuid.size(); ++i)
  {
    if (i == 4 || i == 6 || i == 8 || i == 10)
    {
      result.push_back('-');
    }
    fmt::format_to(std::back_inserter(result), "{:02x}", static_cast<unsigned int>(uuid[i]));
  }
  return result;
}

[[nodiscard]] std::string bytes(std::size_t value)
{
  double constexpr gibibyte = 1024.0 * 1024.0 * 1024.0;
  return fmt::format("{:.2f} GiB ({} bytes)", static_cast<double>(value) / gibibyte, value);
}

[[nodiscard]] char const* supported(bool value) { return value ? "yes" : "no"; }

struct SmokeResult
{
    bool passed;
    std::string summary;
};

[[nodiscard]] SmokeResult smoke_test(uni20::cuda::DeviceResources& resources)
{
  auto& pool = resources.streams();
  uni20::cuda::Completion completion;
  {
    auto stream = pool.try_acquire();
    if (!stream)
    {
      return {.passed = false, .summary = "idle stream acquisition failed"};
    }

    completion = stream->record_completion();
  }
  completion.synchronize();
  pool.synchronize();

  bool const passed = completion.ready() && pool.idle_stream_count() == 1 && pool.leased_stream_count() == 0 &&
                      pool.pending_stream_count() == 0;
  return {.passed = passed,
          .summary = passed ? "stream completion and idle-pool check passed"
                            : "runtime resource state did not return to idle"};
}

void append_package_table(presentation::report_builder& report)
{
  report.table("Uni20 CUDA package")
      .grid()
      .column("component", presentation::table_alignment::left)
      .column("state", presentation::table_alignment::left)
      .column("scope", presentation::table_alignment::left)
      .row("CMake support", "enabled", "UNI20_ENABLE_CUDA=ON")
      .row("backend target", "linked", "uni20_backend_cuda")
      .row("runtime errors", "available", "structured diagnostics through the presentation layer")
      .row("device registry", "available", "validated identities and cached immutable capabilities")
      .row("runtime resources", "available", "scoped global lifetime and canonical per-device resources")
      .row("Tensor CUDA storage", "available", "opaque device mdspans and cuBLAS GEMM lowering")
      .row("CUDA task scheduler", "available", "unified debug and oneTBB host/multi-device schedulers")
      .row("async CUDA GEMM", "planned", "non-blocking resource admission is not yet wired to Tensor GEMM");
}

void append_device_table(presentation::report_builder& report, uni20::cuda::Device device, SmokeResult const& smoke)
{
  auto const& capabilities = device.capabilities();
  report.table(fmt::format("Device {}: {}", device.ordinal(), capabilities.name))
      .grid()
      .column("property", presentation::table_alignment::left)
      .column("value", presentation::table_alignment::left)
      .row("runtime smoke check", smoke.summary)
      .row("UUID", uuid_string(capabilities.uuid))
      .row("PCI address", fmt::format("{:04x}:{:02x}:{:02x}.0", capabilities.pci_domain, capabilities.pci_bus,
                                      capabilities.pci_device))
      .row("compute capability",
           fmt::format("{}.{}", capabilities.compute_capability_major, capabilities.compute_capability_minor))
      .row("global memory", bytes(capabilities.total_global_memory))
      .row("streaming multiprocessors", capabilities.multiprocessor_count)
      .row("warp size", capabilities.warp_size)
      .row("maximum threads per block", capabilities.max_threads_per_block)
      .row("maximum threads per multiprocessor", capabilities.max_threads_per_multiprocessor)
      .row("shared memory per block", fmt::format("{} bytes", capabilities.shared_memory_per_block))
      .row("shared memory per multiprocessor", fmt::format("{} bytes", capabilities.shared_memory_per_multiprocessor))
      .row("asynchronous copy engines", capabilities.async_engine_count)
      .row("concurrent kernels", supported(capabilities.concurrent_kernels))
      .row("unified virtual addressing", supported(capabilities.unified_addressing))
      .row("host-memory mapping", supported(capabilities.can_map_host_memory))
      .row("managed memory", supported(capabilities.managed_memory))
      .row("concurrent managed access", supported(capabilities.concurrent_managed_access))
      .row("direct pageable host-memory access", supported(capabilities.pageable_memory_access))
      .row("stream-ordered memory pools", supported(capabilities.memory_pools_supported))
      .row("stream priorities", supported(capabilities.stream_priorities_supported))
      .row("cooperative launch", supported(capabilities.cooperative_launch))
      .row("ECC enabled", supported(capabilities.ecc_enabled));
}

int run_cuda_example()
{
  presentation::report_builder report("Uni20 CUDA hello world");
  append_build_fields(report);
  report.field("CUDA backend", "uni20_backend_cuda");

  int runtime_version = 0;
  int driver_version = 0;
  int device_count = 0;
  cudaError_t const runtime_status = cudaRuntimeGetVersion(&runtime_version);
  cudaError_t const driver_status = cudaDriverGetVersion(&driver_version);
  cudaError_t const discovery_status = cudaGetDeviceCount(&device_count);

  report.field("CUDA headers", cuda_version(CUDART_VERSION));
  if (runtime_status == cudaSuccess)
  {
    report.field("CUDA runtime", cuda_version(runtime_version));
  }
  if (driver_status == cudaSuccess)
  {
    report.field("driver-supported CUDA", cuda_version(driver_version));
  }

  if (runtime_status != cudaSuccess || driver_status != cudaSuccess || discovery_status != cudaSuccess)
  {
    report.status(presentation::semantic_glyph::failure,
                  "the CUDA backend is linked, but runtime initialization failed");
    auto& failures = report.table("Runtime diagnostics");
    failures.grid()
        .column("operation", presentation::table_alignment::left)
        .column("result", presentation::table_alignment::left);
    if (runtime_status != cudaSuccess)
    {
      failures.row("cudaRuntimeGetVersion", cuda_status(runtime_status));
    }
    if (driver_status != cudaSuccess)
    {
      failures.row("cudaDriverGetVersion", cuda_status(driver_status));
    }
    if (discovery_status != cudaSuccess)
    {
      failures.row("cudaGetDeviceCount", cuda_status(discovery_status));
    }
    failures.row("required at run time", "a compatible NVIDIA driver and a visible CUDA device");
    emit(std::move(report));
    return 1;
  }

  report.field("visible devices", device_count);
  if (device_count == 0)
  {
    report.status(presentation::semantic_glyph::warning, "the CUDA backend is linked, but no CUDA devices are visible")
        .table("Runtime requirements")
        .grid()
        .column("item", presentation::table_alignment::left)
        .column("requirement", presentation::table_alignment::left)
        .row("driver", "a CUDA-compatible NVIDIA driver must be loaded")
        .row("device visibility", "at least one device must be exposed to this process")
        .row("environment", "check CUDA_VISIBLE_DEVICES when device visibility is restricted");
    append_package_table(report);
    emit(std::move(report));
    return 0;
  }

  auto const current_device = uni20::cuda::Device::current();
  auto const devices = uni20::cuda::Device::enumerate();
  auto cuda_lifetime = uni20::cuda::initialize({.streams_per_device = 1});
  report.field("current device", current_device.ordinal());
  report.field("runtime default device", *cuda_lifetime.default_device());

  std::vector<SmokeResult> smoke_results;
  smoke_results.reserve(devices.size());
  bool all_passed = true;
  for (auto const device : devices)
  {
    smoke_results.push_back(smoke_test(uni20::cuda::device_resources(device.ordinal())));
    all_passed = all_passed && smoke_results.back().passed;
  }

  report.status(all_passed ? presentation::semantic_glyph::success : presentation::semantic_glyph::failure,
                all_passed ? "CUDA device discovery and Uni20 runtime checks passed"
                           : "one or more Uni20 CUDA runtime checks failed");
  append_package_table(report);
  for (std::size_t i = 0; i < devices.size(); ++i)
  {
    append_device_table(report, devices[i], smoke_results[i]);
  }

  emit(std::move(report));
  return all_passed ? 0 : 1;
}

#else

int run_cuda_example()
{
  presentation::report_builder report("Uni20 CUDA hello world");
  report.status(presentation::semantic_glyph::skipped, "CUDA support is not configured in this Uni20 build");
  append_build_fields(report);
  report.field("CUDA backend", "disabled");
  report.table("Enable CUDA support")
      .grid()
      .column("item", presentation::table_alignment::left)
      .column("requirement", presentation::table_alignment::left)
      .row("CUDA language", "-DUNI20_ENABLE_CUDA=ON")
      .row("CUDA backend", "-DUNI20_BACKEND_CUDA=ON (the default when CUDA is enabled)")
      .row("CUDA toolkit", "CMake must find nvcc and the CUDA runtime library")
      .row("run-time driver", "a compatible NVIDIA driver is required to use a device")
      .row("run-time hardware", "at least one CUDA device must be visible to the process")
      .row("example target", "cmake --build <build-dir> --target cuda_hello_world_example");
  emit(std::move(report));
  return 0;
}

#endif

} // namespace

int main() { return run_cuda_example(); }
