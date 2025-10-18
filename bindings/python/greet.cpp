#include "registry.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <string>

namespace nb = nanobind;

namespace
{

/// \brief Returns a simple greeting string.
std::string greet() { return std::string("Hello from uni20!"); }

void register_greet(nb::module_& module) { module.def("greet", &greet, "A function that returns a greeting string."); }

} // namespace

static RegisterBinding greet_reg([](nb::module_& module) { register_greet(module); });
