#include "registry.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <uni20/buildinfo.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace nb = nanobind;

namespace
{
using namespace nb::literals;

/// \brief Returns a dictionary with details about the current CMake build.
nb::dict buildinfo()
{
  auto const build_info = uni20::build_info::current();

  auto to_python_string = [](std::string_view text) { return nb::str(text.data(), text.size()); };

  nb::dict info;
  info["generator"] = to_python_string(build_info.generator);
  info["build_type"] = to_python_string(build_info.build_type);
  info["system_name"] = to_python_string(build_info.system_name);
  info["system_version"] = to_python_string(build_info.system_version);
  info["system_processor"] = to_python_string(build_info.system_processor);
  info["cxx_compiler_id"] = to_python_string(build_info.cxx_compiler_id);
  info["cxx_compiler_version"] = to_python_string(build_info.cxx_compiler_version);
  info["cxx_compiler_path"] = to_python_string(build_info.cxx_compiler_path);

  auto populate_entries = [&to_python_string](nb::dict& target, auto const& entries) {
    for (auto const& entry : entries)
    {
      nb::dict metadata;
      metadata["value"] = to_python_string(entry.value);
      if (!entry.help.empty())
      {
        metadata["help"] = to_python_string(entry.help);
      }

      target[to_python_string(entry.key)] = std::move(metadata);
    }
  };

  nb::dict build_options;
  populate_entries(build_options, build_info.build_options);
  info["build_options"] = std::move(build_options);

  nb::dict detected_environment;
  populate_entries(detected_environment, build_info.detected_environment);
  info["detected_environment"] = std::move(detected_environment);
  return info;
}

/// \brief Returns a human-readable Python rendering of the current build metadata.
std::string buildinfo_pretty()
{
  nb::object formatted = nb::module_::import_("pprint").attr("pformat")(buildinfo(), "sort_dicts"_a = false);
  return nb::cast<std::string>(formatted);
}

void register_core(nb::module_& module)
{
  module.def("buildinfo", &buildinfo, "Return build system metadata for the current uni20 build.");
  module.def("buildinfo_pretty", &buildinfo_pretty,
             "Return formatted build system metadata for the current uni20 build.");

  // Any other core utilities can go here
  // e.g., module.def("set_log_level", &set_log_level);
}

} // namespace

// Register into the global registry
static RegisterBinding core_reg([](nb::module_& module) { register_core(module); });
