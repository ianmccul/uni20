#include "buildinfo.hpp"
#include "registry.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <string>
#include <utility>

namespace nb = nanobind;

namespace
{

/// \brief Returns a dictionary with details about the current CMake build.
nb::dict buildinfo()
{
  using namespace uni20::python::build_info;

  nb::dict info;
  info["generator"] = nb::str(kGenerator.data(), kGenerator.size());
  info["build_type"] = nb::str(kBuildType.data(), kBuildType.size());
  info["system_name"] = nb::str(kSystemName.data(), kSystemName.size());
  info["system_version"] = nb::str(kSystemVersion.data(), kSystemVersion.size());
  info["system_processor"] = nb::str(kSystemProcessor.data(), kSystemProcessor.size());
  info["cxx_compiler_id"] = nb::str(kCompilerId.data(), kCompilerId.size());
  info["cxx_compiler_version"] = nb::str(kCompilerVersion.data(), kCompilerVersion.size());
  info["cxx_compiler_path"] = nb::str(kCompilerPath.data(), kCompilerPath.size());

  auto populate_entries = [](nb::dict& target, auto const& entries) {
    for (auto const& entry : entries)
    {
      nb::dict metadata;
      metadata["value"] = nb::str(entry.value.data(), entry.value.size());
      if (!entry.help.empty())
      {
        metadata["help"] = nb::str(entry.help.data(), entry.help.size());
      }

      target[nb::str(entry.key.data(), entry.key.size())] = std::move(metadata);
    }
  };

  nb::dict build_options;
  populate_entries(build_options, kBuildOptions);
  info["build_options"] = std::move(build_options);

  nb::dict detected_environment;
  populate_entries(detected_environment, kDetectedEnvironment);
  info["detected_environment"] = std::move(detected_environment);
  return info;
}

void register_core(nb::module_& module)
{
  module.def("buildinfo", &buildinfo, "Return build system metadata for the current uni20 build.");

  // Any other core utilities can go here
  // e.g., module.def("set_log_level", &set_log_level);
}

} // namespace

// Register into the global registry
static RegisterBinding core_reg([](nb::module_& module) { register_core(module); });
