#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>

/// \brief Returns a simple greeting string.
std::string greet() { return std::string("Hello from uni20!"); }

NB_MODULE(uni20_python, module)
{
  module.doc() = "uni20 Python bindings (hello world example)";
  module.def("greet", &greet, "A function that returns a greeting string.");
}
