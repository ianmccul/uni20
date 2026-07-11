#include "registry.hpp"

#include <nanobind/nanobind.h>

#include <uni20/common/trace.hpp>

namespace nb = nanobind;

NB_MODULE(uni20, module)
{
  // A recoverable Uni20 error must cross the extension boundary as a Python
  // exception. CHECK and PANIC remain unconditional invariant failures.
  trace::get_formatting_options().set_errors_abort(false);

  module.doc() = "uni20 Python bindings";

  for (auto const& fn : BindingRegistry::list())
  {
    fn(module);
  }
}
