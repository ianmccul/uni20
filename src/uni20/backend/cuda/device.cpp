#include <uni20/backend/cuda/device.hpp>

#include <uni20/backend/cuda/cuda_error.hpp>

#include <cuda_runtime_api.h>

#include <memory>
#include <mutex>
#include <optional>

namespace uni20::cuda
{
namespace
{

DeviceCapabilities query_capabilities(int ordinal)
{
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, ordinal), "cudaGetDeviceProperties", ordinal);

  DeviceCapabilities result;
  result.name = properties.name;
  for (std::size_t i = 0; i < result.uuid.size(); ++i)
  {
    result.uuid[i] = static_cast<std::uint8_t>(static_cast<unsigned char>(properties.uuid.bytes[i]));
  }
  result.total_global_memory = properties.totalGlobalMem;
  result.compute_capability_major = properties.major;
  result.compute_capability_minor = properties.minor;
  result.multiprocessor_count = properties.multiProcessorCount;
  result.warp_size = properties.warpSize;
  result.max_threads_per_block = properties.maxThreadsPerBlock;
  result.max_threads_per_multiprocessor = properties.maxThreadsPerMultiProcessor;
  result.shared_memory_per_block = properties.sharedMemPerBlock;
  result.shared_memory_per_multiprocessor = properties.sharedMemPerMultiprocessor;
  result.async_engine_count = properties.asyncEngineCount;
  result.pci_domain = properties.pciDomainID;
  result.pci_bus = properties.pciBusID;
  result.pci_device = properties.pciDeviceID;
  result.concurrent_kernels = properties.concurrentKernels != 0;
  result.unified_addressing = properties.unifiedAddressing != 0;
  result.can_map_host_memory = properties.canMapHostMemory != 0;
  result.managed_memory = properties.managedMemory != 0;
  result.concurrent_managed_access = properties.concurrentManagedAccess != 0;
  result.pageable_memory_access = properties.pageableMemoryAccess != 0;
  result.memory_pools_supported = properties.memoryPoolsSupported != 0;
  result.stream_priorities_supported = properties.streamPrioritiesSupported != 0;
  result.cooperative_launch = properties.cooperativeLaunch != 0;
  result.ecc_enabled = properties.ECCEnabled != 0;
  return result;
}

class DeviceRegistry {
  public:
    int count()
    {
      std::call_once(count_once_, [this] {
        int count = 0;
        check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        count_ = count;
        entries_.reserve(static_cast<std::size_t>(count_));
        for (int ordinal = 0; ordinal < count_; ++ordinal)
        {
          entries_.push_back(std::make_unique<Entry>());
        }
      });
      return count_;
    }

    DeviceCapabilities const& capabilities(int ordinal)
    {
      int const device_count = this->count();
      if (ordinal < 0 || ordinal >= device_count)
      {
        raise_runtime_error(cudaErrorInvalidDevice, "cudaGetDeviceProperties", ordinal);
      }

      Entry& entry = *entries_[static_cast<std::size_t>(ordinal)];
      std::call_once(entry.once, [&entry, ordinal] { entry.capabilities.emplace(query_capabilities(ordinal)); });
      return *entry.capabilities;
    }

  private:
    struct Entry
    {
        std::once_flag once;
        std::optional<DeviceCapabilities> capabilities;
    };

    std::once_flag count_once_;
    int count_ = 0;
    std::vector<std::unique_ptr<Entry>> entries_;
};

DeviceRegistry& device_registry()
{
  static DeviceRegistry registry;
  return registry;
}

} // namespace

Device Device::get(int ordinal)
{
  (void)device_registry().capabilities(ordinal);
  return Device(ordinal);
}

int Device::count() { return device_registry().count(); }

std::vector<Device> Device::enumerate()
{
  int const device_count = Device::count();
  std::vector<Device> devices;
  devices.reserve(static_cast<std::size_t>(device_count));
  for (int ordinal = 0; ordinal < device_count; ++ordinal)
  {
    devices.push_back(Device::get(ordinal));
  }
  return devices;
}

Device Device::current()
{
  int ordinal = -1;
  check(cudaGetDevice(&ordinal), "cudaGetDevice");
  return Device::get(ordinal);
}

DeviceCapabilities const& Device::capabilities() const { return device_registry().capabilities(ordinal_); }

} // namespace uni20::cuda
