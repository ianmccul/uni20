#include "registry.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

static void register_core(nb::module_& m)
{
  m.def("version", [] { return std::string("0.1.0"); });

  // Any other core utilities can go here
  // e.g., m.def("set_log_level", &set_log_level);
}

// Register into the global registry
static RegisterBinding core_reg([](nb::module_& m){ register_core(m); });
