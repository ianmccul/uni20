#include <uni20/linalg/dispatch.hpp>

#include <fmt/core.h>
#include <span>
#include <string_view>
#include <vector>

namespace example
{
using uni20::linalg::kernel_types_maybe;
using uni20::linalg::kernel_types_yes;
using uni20::linalg::KernelTypeAcceptance;

struct scale_in_place_op
{
    static constexpr std::string_view name = "scale_in_place";
};

struct blocked_backend
{
    static constexpr std::string_view name = "blocked";
};

struct scalar_backend
{
    static constexpr std::string_view name = "scalar";
};

consteval auto kernel_accepts_types(blocked_backend const&, scale_in_place_op const&, std::span<double>&, double const&)
{
  return kernel_types_maybe;
}

bool try_kernel(blocked_backend, scale_in_place_op, std::span<double> values, double factor)
{
  if (values.size() % 4 != 0)
  {
    return false;
  }

  fmt::print("blocked backend accepted {} values\n", values.size());
  for (double& value : values)
  {
    value *= factor;
  }
  return true;
}

consteval auto kernel_accepts_types(scalar_backend const&, scale_in_place_op const&, std::span<double>&, double const&)
{
  return kernel_types_yes;
}

bool try_kernel(scalar_backend, scale_in_place_op, std::span<double> values, double factor)
{
  fmt::print("scalar backend accepted {} values\n", values.size());
  for (double& value : values)
  {
    value *= factor;
  }
  return true;
}

constexpr std::string_view acceptance_name(KernelTypeAcceptance acceptance)
{
  switch (acceptance)
  {
    case KernelTypeAcceptance::no:
      return "no";
    case KernelTypeAcceptance::maybe:
      return "maybe";
    case KernelTypeAcceptance::yes:
      return "yes";
  }
  return "unknown";
}

} // namespace example

int main()
{
  using namespace example;
  using uni20::linalg::backend_list;

  std::vector<double> short_storage{1.0, 2.0, 3.0};
  std::span<double> short_values(short_storage);
  auto blocked_only = backend_list{blocked_backend{}};

  auto const blocked_acceptance =
      uni20::linalg::probe_dispatch_kernel(blocked_only, scale_in_place_op{}, short_values, 2.0);
  fmt::print("blocked-only type acceptance: {}\n", acceptance_name(blocked_acceptance));

  // A `maybe` backend can decline an individual runtime instance before side effects.
  bool const blocked_success = uni20::linalg::try_dispatch_kernel(blocked_only, scale_in_place_op{}, short_values, 2.0);
  fmt::print("blocked-only runtime result: {}\n", blocked_success ? "performed" : "declined");

  auto with_fallback = backend_list{blocked_backend{}, scalar_backend{}};
  auto const fallback_acceptance =
      uni20::linalg::probe_dispatch_kernel(with_fallback, scale_in_place_op{}, short_values, 2.0);
  fmt::print("fallback-list type acceptance: {}\n", acceptance_name(fallback_acceptance));

  // The blocked backend declines three values, so dispatch continues to the scalar backend.
  uni20::linalg::dispatch_kernel(with_fallback, scale_in_place_op{}, short_values, 2.0);

  std::vector<double> blocked_storage{1.0, 2.0, 3.0, 4.0};
  std::span<double> blocked_values(blocked_storage);
  // Four values satisfy the first backend, so the fallback is not called.
  uni20::linalg::dispatch_kernel(with_fallback, scale_in_place_op{}, blocked_values, 3.0);

  bool const short_ok = short_storage == std::vector<double>{2.0, 4.0, 6.0};
  bool const blocked_ok = blocked_storage == std::vector<double>{3.0, 6.0, 9.0, 12.0};
  return short_ok && blocked_ok ? 0 : 1;
}
