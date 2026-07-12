#include <uni20/common/presentation.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/dispatch.hpp>

#include <cstdio>
#include <fmt/core.h>
#include <string_view>

namespace example
{
using uni20::linalg::kernel_types_maybe;
using uni20::linalg::KernelTypeAcceptance;

struct demo_op
{
    static constexpr std::string_view name = "dispatch_error_demo";
};

struct unavailable_backend
{
    static constexpr std::string_view name = "unavailable";
};

struct declining_backend
{
    static constexpr std::string_view name = "declining";
};

consteval auto kernel_accepts_types(declining_backend const&, demo_op const&, int&) { return kernel_types_maybe; }

bool try_kernel(declining_backend, demo_op, int&) { return false; }

void print_error(std::string_view heading, uni20::linalg::KernelDispatchError const& error)
{
  fmt::print("\n{}\n", heading);
  auto policy = trace::get_formatting_options().presentation_policy();
  fmt::print("{}\n", uni20::presentation::render_terminal(trace::format_diagnostic(error), policy, stdout));
}

} // namespace example

int main()
{
  using namespace example;
  using uni20::linalg::backend_list;
  using uni20::linalg::KernelDispatchError;
  using uni20::linalg::KernelDispatchFailure;

  // Native C++ defaults to aborting on checked errors. Bindings and other
  // recoverable boundaries select throw mode so they can translate the error.
  auto& formatting = trace::get_formatting_options();
  bool const previous_errors_abort = formatting.errors_abort();
  formatting.set_errors_abort(false);

  int value = 0;
  bool saw_runtime_decline = false;
  try
  {
    auto backends = backend_list{unavailable_backend{}, declining_backend{}};
    uni20::linalg::dispatch_kernel(backends, demo_op{}, value);
  }
  catch (KernelDispatchError const& error)
  {
    saw_runtime_decline = error.failure() == KernelDispatchFailure::all_candidates_declined;
    print_error("Checked dispatch: runtime candidates exhausted", error);
  }

  bool saw_type_rejection = false;
  try
  {
    auto backends = backend_list{unavailable_backend{}};
    uni20::linalg::dynamic_dispatch_kernel(backends, demo_op{}, value);
  }
  catch (KernelDispatchError const& error)
  {
    saw_type_rejection = error.failure() == KernelDispatchFailure::no_eligible_backend;
    print_error("Dynamic dispatch: no backend accepts the types", error);
  }

  formatting.set_errors_abort(previous_errors_abort);
  return saw_runtime_decline && saw_type_rejection ? 0 : 1;
}
