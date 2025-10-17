#pragma once

#include <vector>
#include <functional>
#include <nanobind/nanobind.h>

/// \brief Global registry for Python binding registration functions.
/// Each submodule contributes a registration lambda at static initialization time.
struct BindingRegistry {
  using RegFn = std::function<void(nanobind::module_&)>;

  static auto& list() {
    static std::vector<RegFn> functions;
    return functions;
  }
};

/// \brief Helper type that registers a function at static initialization.
/// Constructed once per submodule.
struct RegisterBinding {
  explicit RegisterBinding(BindingRegistry::RegFn fn) {
    BindingRegistry::list().push_back(std::move(fn));
  }
};
