#include "common/trace.hpp"

#include <gtest/gtest.h>

#include <regex>
#include <vector>

namespace
{
trace::FormattingOptions make_test_options()
{
  auto opts = trace::get_formatting_options("trace-format-test");
  opts.set_color_output(trace::FormattingOptions::ColorOptions::no);
  return opts;
}
} // namespace

TEST(TraceFormatting, FloatingPointPrecision)
{
  auto opts = make_test_options();
  opts.fp_precision_float32 = 2;
  opts.fp_precision_float64 = 4;

  EXPECT_EQ("3.14", trace::formatValue(3.14159f, opts));
  EXPECT_EQ("2.7183", trace::formatValue(2.718281828, opts));
  EXPECT_EQ("1.23-6.79i", trace::formatValue(std::complex<float>{1.2345f, -6.789f}, opts));
  EXPECT_EQ("-0.1250+42.5000i", trace::formatValue(std::complex<double>{-0.125, 42.5}, opts));
}

TEST(TraceFormatting, NullRepresentations)
{
  auto opts = make_test_options();
  EXPECT_EQ("(null)", trace::formatValue(std::string_view{}, opts));
  const char* null_ptr = nullptr;
  EXPECT_EQ("(null)", trace::formatValue(null_ptr, opts));
}

TEST(TraceFormatting, ContainerFormatting)
{
  auto opts = make_test_options();

  std::vector<std::string> single_line{"1", "2", "3"};
  EXPECT_EQ("[ 1, 2, 3 ]", trace::formatContainerToString(single_line));

  auto formatted_single = trace::formatItemString({"values", false}, single_line, opts, 80);
  EXPECT_EQ("values = [ 1, 2, 3 ]", formatted_single);

  std::vector<std::string> multi_line{"first\nsecond", "third"};
  auto formatted_container = trace::formatContainerToString(multi_line);
  EXPECT_EQ("[\nfirst\n  second,\n  third\n]", formatted_container);

  auto formatted_multi = trace::formatItemString({"values", false}, multi_line, opts, 80);
  EXPECT_EQ("\nvalues = [\n         first\n           second,\n           third\n         ]", formatted_multi);
}

TEST(TraceFormatting, TimestampMatchesPattern)
{
  auto timestamp = trace::format_timestamp();
  std::regex timestamp_pattern(R"((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?\.\d{6}))");
  EXPECT_TRUE(std::regex_match(timestamp, timestamp_pattern)) << timestamp;
} 
