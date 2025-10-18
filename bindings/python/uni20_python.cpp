#include "registry.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(uni20, module)
{
  module.doc() = "uni20 Python bindings";

  for (auto const& fn : BindingRegistry::list())
  {
    fn(module);
  }
}
