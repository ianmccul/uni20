#pragma once

/**
 * \file device.hpp
 * \ingroup backend_cuda
 * \brief CUDA device discovery and immutable hardware capability snapshots.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace uni20::cuda
{

/// \brief Immutable hardware properties cached for one CUDA device ordinal.
/// \details `Device::capabilities()` returns the process-wide cached instance.
///          Provider-library support and mutable execution resources are not
///          device hardware capabilities and are tracked separately.
struct DeviceCapabilities
{
    /// \brief CUDA runtime device name.
    std::string name{};

    /// \brief Stable 16-byte CUDA device UUID.
    std::array<std::uint8_t, 16> uuid{};

    /// \brief Total device global memory in bytes.
    std::size_t total_global_memory = 0;

    /// \brief CUDA compute-capability major version.
    int compute_capability_major = 0;

    /// \brief CUDA compute-capability minor version.
    int compute_capability_minor = 0;

    /// \brief Number of streaming multiprocessors.
    int multiprocessor_count = 0;

    /// \brief Hardware warp size in threads.
    int warp_size = 0;

    /// \brief Maximum threads in one thread block.
    int max_threads_per_block = 0;

    /// \brief Maximum resident threads per streaming multiprocessor.
    int max_threads_per_multiprocessor = 0;

    /// \brief Maximum shared memory per thread block in bytes.
    std::size_t shared_memory_per_block = 0;

    /// \brief Maximum shared memory per streaming multiprocessor in bytes.
    std::size_t shared_memory_per_multiprocessor = 0;

    /// \brief Number of asynchronous copy engines.
    int async_engine_count = 0;

    /// \brief PCI domain identifier.
    int pci_domain = 0;

    /// \brief PCI bus identifier.
    int pci_bus = 0;

    /// \brief PCI device identifier.
    int pci_device = 0;

    /// \brief Whether the device supports concurrent kernel execution.
    bool concurrent_kernels = false;

    /// \brief Whether the device uses unified virtual addressing.
    bool unified_addressing = false;

    /// \brief Whether the device can map page-locked host memory.
    bool can_map_host_memory = false;

    /// \brief Whether the device supports managed-memory allocation.
    bool managed_memory = false;

    /// \brief Whether CPU and device may concurrently access managed memory.
    bool concurrent_managed_access = false;

    /// \brief Whether the device can directly access pageable host memory.
    bool pageable_memory_access = false;

    /// \brief Whether stream-ordered CUDA memory pools are supported.
    bool memory_pools_supported = false;

    /// \brief Whether CUDA stream priorities are supported.
    bool stream_priorities_supported = false;

    /// \brief Whether cooperative kernel launch is supported.
    bool cooperative_launch = false;

    /// \brief Whether ECC is enabled on the device.
    bool ecc_enabled = false;
};

/// \brief Validated CUDA device identity with cached immutable capabilities.
/// \details A `Device` is a cheap value containing only a device ordinal.
///          `get()` validates that ordinal and initializes its process-wide
///          capability cache. It does not own streams, schedulers, provider
///          handles, memory pools, or allocations.
class Device {
  public:
    /// \brief Validate and return one CUDA device.
    /// \param ordinal CUDA runtime device ordinal.
    /// \return Cheap device value referring to the cached device record.
    [[nodiscard]] static Device get(int ordinal);

    /// \brief Return the number of CUDA devices visible to this process.
    [[nodiscard]] static int count();

    /// \brief Discover every visible CUDA device in ordinal order.
    /// \return Validated device values with initialized capability caches.
    [[nodiscard]] static std::vector<Device> enumerate();

    /// \brief Return the CUDA device currently selected on the calling thread.
    [[nodiscard]] static Device current();

    /// \brief Return this device's CUDA runtime ordinal.
    [[nodiscard]] constexpr int ordinal() const noexcept { return ordinal_; }

    /// \brief Return this device's cached immutable hardware capabilities.
    [[nodiscard]] DeviceCapabilities const& capabilities() const;

    friend constexpr bool operator==(Device const&, Device const&) = default;

  private:
    explicit constexpr Device(int ordinal) noexcept : ordinal_(ordinal) {}

    int ordinal_;
};

} // namespace uni20::cuda
