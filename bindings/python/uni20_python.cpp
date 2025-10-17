#include "registry.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(uni20_python, module)
{
  module.doc() = "uni20 Python bindings (hello world example)";

  for (auto const& fn : BindingRegistry::list()) {
    fn(module);
  }
}
