#include <uni20/linalg/dispatch.hpp>

#include <gtest/gtest.h>

#include <string_view>
#include <vector>

namespace
{
using uni20::linalg::backend_list;
using uni20::linalg::kernel_types_maybe;
using uni20::linalg::kernel_types_yes;
using uni20::linalg::KernelAttempt;
using uni20::linalg::KernelTypeAcceptance;

struct diagnostic_test_op
{
    static constexpr std::string_view name = "diagnostic_test";
};

struct ineligible_backend
{
    static constexpr std::string_view name = "ineligible";
};

struct declining_backend
{
    static constexpr std::string_view name = "declining";
};

struct accepting_backend
{
    static constexpr std::string_view name = "accepting";
};

struct unreached_backend
{
    static constexpr std::string_view name = "unreached";
};

consteval auto kernel_accepts_types(declining_backend const&, diagnostic_test_op const&, int&)
{
  return kernel_types_maybe;
}

consteval auto kernel_accepts_types(accepting_backend const&, diagnostic_test_op const&, int&)
{
  return kernel_types_yes;
}

consteval auto kernel_accepts_types(unreached_backend const&, diagnostic_test_op const&, int&)
{
  return kernel_types_yes;
}

KernelAttempt try_kernel(declining_backend, diagnostic_test_op, int&) { return KernelAttempt::unsupported_layout; }

KernelAttempt try_kernel(accepting_backend, diagnostic_test_op, int& value)
{
  ++value;
  return KernelAttempt::success;
}

KernelAttempt try_kernel(unreached_backend, diagnostic_test_op, int& value)
{
  value += 100;
  return KernelAttempt::success;
}
} // namespace

TEST(KernelDispatchDiagnosticsTest, DisabledByDefaultAndScopedSinkRestoresState)
{
  namespace diagnostics = uni20::linalg::dispatch_diagnostics;
  diagnostics::reset_sink();
  EXPECT_FALSE(diagnostics::enabled());

  std::vector<diagnostics::event> events;
  {
    diagnostics::scoped_sink capture([&](diagnostics::event const& diagnostic) { events.push_back(diagnostic); });
    EXPECT_TRUE(diagnostics::enabled());

    int value = 4;
    auto const backends =
        backend_list{ineligible_backend{}, declining_backend{}, accepting_backend{}, unreached_backend{}};
    EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(backends, diagnostic_test_op{}, value));
    EXPECT_EQ(value, 5);
  }

  EXPECT_FALSE(diagnostics::enabled());
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].operation, "diagnostic_test");
  EXPECT_TRUE(events[0].succeeded());
  ASSERT_TRUE(events[0].selected_backend().has_value());
  EXPECT_EQ(*events[0].selected_backend(), "accepting");
  ASSERT_EQ(events[0].backend_attempts.size(), 4);
  EXPECT_EQ(events[0].backend_attempts[0].type_acceptance, KernelTypeAcceptance::no);
  EXPECT_FALSE(events[0].backend_attempts[0].runtime_result.has_value());
  EXPECT_EQ(events[0].backend_attempts[1].runtime_result, KernelAttempt::unsupported_layout);
  EXPECT_EQ(events[0].backend_attempts[2].runtime_result, KernelAttempt::success);
  EXPECT_EQ(events[0].backend_attempts[3].type_acceptance, KernelTypeAcceptance::yes);
  EXPECT_FALSE(events[0].backend_attempts[3].runtime_result.has_value());
}

TEST(KernelDispatchDiagnosticsTest, CheckedDispatchEmitsSelectedBackendReport)
{
  namespace diagnostics = uni20::linalg::dispatch_diagnostics;
  diagnostics::reset_sink();

  std::vector<diagnostics::event> events;
  diagnostics::scoped_sink capture([&](diagnostics::event const& diagnostic) { events.push_back(diagnostic); });

  int value = 8;
  uni20::linalg::dispatch_kernel(backend_list{declining_backend{}, accepting_backend{}}, diagnostic_test_op{}, value);

  ASSERT_EQ(events.size(), 1);
  ASSERT_TRUE(events[0].selected_backend().has_value());
  EXPECT_EQ(*events[0].selected_backend(), "accepting");
  auto const report = uni20::linalg::diagnostic_report(events[0]);
  ASSERT_EQ(report.statuses().size(), 1);
  EXPECT_EQ(report.statuses()[0].first, uni20::presentation::semantic_glyph::success);
  EXPECT_EQ(report.statuses()[0].second, "Kernel dispatch for 'diagnostic_test' selected 'accepting'");
  ASSERT_EQ(report.tables().size(), 1);
  EXPECT_EQ(report.tables()[0].title(), "Ordered backend candidates");
}
