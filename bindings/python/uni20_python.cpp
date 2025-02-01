#include <pybind11/pybind11.h>

namespace py = pybind11;

// A simple function that returns a greeting.
std::string greet() {
    return "Hello from uni20!";
}

// This macro creates the Python module named 'uni20_python'
PYBIND11_MODULE(uni20_python, m) {
    m.doc() = "uni20 Python bindings (hello world example)"; // Optional module docstring

    // Expose the greet() function to Python.
    m.def("greet", &greet, "A function that returns a greeting string");
}
