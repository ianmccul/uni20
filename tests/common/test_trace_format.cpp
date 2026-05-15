#include <uni20/common/trace.hpp>
#include <uni20/common/mdspan.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace
{
class EnvVarGuard {
  public:
    explicit EnvVarGuard(std::string name) : name_(std::move(name))
    {
      if (char const* value = std::getenv(name_.c_str()))
      {
        original_ = value;
      }
    }

    EnvVarGuard(EnvVarGuard const&) = delete;
    EnvVarGuard& operator=(EnvVarGuard const&) = delete;

    ~EnvVarGuard()
    {
      if (original_)
      {
        ::setenv(name_.c_str(), original_->c_str(), 1);
      }
      else
      {
        ::unsetenv(name_.c_str());
      }
    }

    void set(std::string const& value) const { ::setenv(name_.c_str(), value.c_str(), 1); }

    void unset() const { ::unsetenv(name_.c_str()); }

  private:
    std::string name_;
    std::optional<std::string> original_;
};

trace::FormattingOptions make_test_options()
{
  auto opts = trace::get_formatting_options("trace-format-test");
  opts.set_color_output(trace::FormattingOptions::ColorOptions::no);
  return opts;
}

std::unique_ptr<std::FILE, decltype(&std::fclose)> make_terminal_stream()
{
  int const fd = ::open("/dev/ptmx", O_RDWR | O_NOCTTY);
  if (fd < 0)
  {
    return {nullptr, &std::fclose};
  }

  std::FILE* stream = ::fdopen(fd, "w");
  if (stream == nullptr)
  {
    ::close(fd);
    return {nullptr, &std::fclose};
  }

  return {stream, &std::fclose};
}

bool contains_ansi(std::string const& text) { return text.find("\033[") != std::string::npos; }
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

TEST(TraceFormatting, ContainerFormattingUsesDisplayCellWidth)
{
  auto opts = make_test_options();
  opts.presentation_policy().charset = uni20::presentation::text_charset::utf8;

  std::vector<std::string> cjk_width{"x", "\xE4\xB8\xAD"};
  EXPECT_EQ("[  x, \xE4\xB8\xAD ]", trace::formatContainerToString(cjk_width, opts));
}

TEST(TraceFormatting, MultilineItemIndentUsesDisplayCellWidth)
{
  auto opts = make_test_options();

  auto formatted = trace::formatItemString({"\xE4\xB8\xAD", false}, "first\nsecond", opts, 80);
  EXPECT_EQ("\n\xE4\xB8\xAD = first\n     second", formatted);
}

TEST(TraceFormatting, TimestampMatchesPattern)
{
  auto timestamp = trace::format_timestamp();
  std::regex timestamp_pattern(R"((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{9}))");
  EXPECT_TRUE(std::regex_match(timestamp, timestamp_pattern)) << timestamp;
}

TEST(TraceFormatting, NoColorSuppressesAutoColorOnTerminal)
{
  EnvVarGuard no_color("NO_COLOR");
  no_color.set("1");

  auto stream = make_terminal_stream();
  if (stream == nullptr)
  {
    GTEST_SKIP() << "pseudo-terminal stream is unavailable";
  }

  auto opts = trace::get_formatting_options("trace-format-no-color-auto");
  opts.set_output_stream(stream.get());
  opts.set_color_output(trace::FormattingOptions::ColorOptions::autocolor);

  EXPECT_FALSE(opts.should_show_color());
}

TEST(TraceFormatting, EmptyNoColorDoesNotSuppressAutoColorOnTerminal)
{
  EnvVarGuard no_color("NO_COLOR");
  no_color.set("");

  auto stream = make_terminal_stream();
  if (stream == nullptr)
  {
    GTEST_SKIP() << "pseudo-terminal stream is unavailable";
  }

  auto opts = trace::get_formatting_options("trace-format-empty-no-color-auto");
  opts.set_output_stream(stream.get());
  opts.set_color_output(trace::FormattingOptions::ColorOptions::autocolor);

  EXPECT_TRUE(opts.should_show_color());
}

TEST(TraceFormatting, NoColorDoesNotOverrideExplicitColorYes)
{
  EnvVarGuard no_color("NO_COLOR");
  no_color.set("1");

  auto opts = trace::get_formatting_options("trace-format-no-color-explicit-yes");
  opts.set_color_output(trace::FormattingOptions::ColorOptions::yes);

  EXPECT_TRUE(opts.should_show_color());
}

TEST(TraceFormatting, FormatStyleUsesPresentationRendererForColor)
{
  auto opts = trace::get_formatting_options("trace-format-renderer-color");

  opts.set_color_output(trace::FormattingOptions::ColorOptions::yes);
  EXPECT_TRUE(contains_ansi(opts.format_style("TRACE", "TRACE")));

  opts.set_color_output(trace::FormattingOptions::ColorOptions::no);
  EXPECT_FALSE(contains_ansi(opts.format_style("TRACE", "TRACE")));
}

TEST(TraceFormatting, PlainFileModeSuppressesAutoColor)
{
  auto file = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::tmpfile(), &std::fclose);
  ASSERT_NE(file, nullptr);

  auto opts = trace::get_formatting_options("trace-format-plain-file");
  opts.set_output_stream(file.get());
  opts.set_color_output(trace::FormattingOptions::ColorOptions::autocolor);

  EXPECT_FALSE(opts.should_show_color());
  EXPECT_FALSE(contains_ansi(opts.format_style("ERROR", "ERROR")));
}

TEST(TraceFormatting, TraceLineUsesPresentationCharsetPolicy)
{
  auto& opts = trace::get_formatting_options("trace-format-ascii-line");
  std::ostringstream oss;

  opts.set_sink([&oss](std::string msg) { oss << msg; });
  opts.set_color_output(trace::FormattingOptions::ColorOptions::no);
  opts.presentation_policy().charset = uni20::presentation::text_charset::ascii_escape;

  std::string value = "\xE4\xB8\xAD\xE6\x96\x87";
  trace::TraceModuleCall("trace-format-ascii-line", "value", __FILE__, __LINE__, value);

  auto const output = oss.str();
  EXPECT_NE(output.find("value = \\u4E2D\\u6587"), std::string::npos) << output;
  EXPECT_EQ(output.find(value), std::string::npos) << output;

  opts.presentation_policy().charset = uni20::presentation::text_charset::utf8;
  opts.set_output_stream(stderr);
}

TEST(TraceFormatting, SemanticGlyphsUseFormatterPolicy)
{
  auto opts = make_test_options();
  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::unicode;

  EXPECT_EQ(opts.format_glyph(uni20::presentation::semantic_glyph::warning, "ERROR"), "\xE2\x96\xB2");
  EXPECT_EQ(opts.format_glyph(uni20::presentation::semantic_glyph::arrow_right, "TRACE"), "\xE2\x86\x92");

  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::emoji;
  EXPECT_EQ(opts.format_glyph(uni20::presentation::semantic_glyph::warning, "ERROR"), "\xF0\x9F\x9A\xA8");

  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::ascii;
  EXPECT_EQ(opts.format_glyph(uni20::presentation::semantic_glyph::success, "TRACE"), "[OK]");
  EXPECT_EQ(opts.format_glyph(uni20::presentation::semantic_glyph::failure, "ERROR"), "[FAIL]");
  EXPECT_EQ(opts.format_glyph(uni20::presentation::semantic_glyph::arrow_right, "TRACE"), "->");
  EXPECT_EQ(opts.format_glyph(uni20::presentation::semantic_glyph::tree_branch, "TRACE"), "|-");
}

TEST(TraceFormatting, TracePayloadSeparatorUsesGlyphPolicy)
{
  static constexpr char const* module = "TRACE_FORMAT_SEPARATOR";
  auto& opts = trace::get_formatting_options(module);
  std::ostringstream oss;

  opts.set_sink([&oss](std::string msg) { oss << msg; });
  opts.set_color_output(trace::FormattingOptions::ColorOptions::no);
  opts.timestamp = false;
  opts.threadId = trace::FormattingOptions::ThreadIdOptions::no;

  int value = 42;

  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::unicode;
  trace::TraceModuleCall(module, "value", __FILE__, __LINE__, value);
  EXPECT_NE(oss.str().find(" \xE2\x86\x92 value = 42"), std::string::npos) << oss.str();

  oss.str("");
  oss.clear();
  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::ascii;
  value = 43;
  trace::TraceModuleCall(module, "value", __FILE__, __LINE__, value);
  EXPECT_NE(oss.str().find(" -> value = 43"), std::string::npos) << oss.str();

  oss.str("");
  oss.clear();
  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::emoji;
  value = 44;
  trace::TraceModuleCall(module, "value", __FILE__, __LINE__, value);
  EXPECT_NE(oss.str().find(" \xE2\x9E\xA1\xEF\xB8\x8F value = 44"), std::string::npos) << oss.str();

  opts.set_output_stream(stderr);
}

TEST(TraceFormatting, DiagnosticHeadersUseSemanticGlyphPolicy)
{
  auto opts = make_test_options();

  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::emoji;
  auto check = trace::detail::make_diagnostic_header(opts, "CHECK", "CHECK", __FILE__, __LINE__);
  auto panic = trace::detail::make_diagnostic_header(opts, "PANIC", "PANIC", __FILE__, __LINE__);
  EXPECT_NE(opts.render(check).find("\xE2\x9D\x8C CHECK at "), std::string::npos);
  EXPECT_NE(opts.render(panic).find("\xF0\x9F\x9A\xA8 PANIC at "), std::string::npos);

  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::unicode;
  check = trace::detail::make_diagnostic_header(opts, "CHECK", "CHECK", __FILE__, __LINE__);
  panic = trace::detail::make_diagnostic_header(opts, "PANIC", "PANIC", __FILE__, __LINE__);
  EXPECT_NE(opts.render(check).find("\xE2\x9C\x97 CHECK at "), std::string::npos);
  EXPECT_NE(opts.render(panic).find("\xE2\x96\xB2 PANIC at "), std::string::npos);

  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::ascii;
  auto error = trace::detail::make_diagnostic_header(opts, "ERROR", "ERROR", __FILE__, __LINE__);
  auto precondition =
      trace::detail::make_diagnostic_header(opts, "PRECONDITION", "PRECONDITION", __FILE__, __LINE__);
  EXPECT_NE(opts.render(error).find("[FAIL] ERROR at "), std::string::npos);
  EXPECT_NE(opts.render(precondition).find("[WARN] PRECONDITION at "), std::string::npos);
}

TEST(TraceFormatting, GlobalEnvironmentConfiguresPresentationPolicy)
{
  EnvVarGuard glyphs("UNI20_GLYPHS");
  EnvVarGuard charset("UNI20_CHARSET");
  EnvVarGuard columns("COLUMNS");

  glyphs.set("ascii");
  charset.set("ascii_escape");
  columns.set("12");

  trace::FormattingOptions opts;
  opts.set_color_output(trace::FormattingOptions::ColorOptions::no);

  EXPECT_EQ(opts.presentation_policy().glyphs, uni20::presentation::glyph_set::ascii);
  EXPECT_EQ(opts.presentation_policy().charset, uni20::presentation::text_charset::ascii_escape);
  EXPECT_EQ(opts.terminal_width, 12);
  EXPECT_EQ(opts.format_glyph(uni20::presentation::semantic_glyph::arrow_right, "TRACE"), "->");
  EXPECT_EQ(trace::formatItemString({"value", false}, "\xE4\xB8\xAD", opts, 80), "value = \\u4E2D");
  EXPECT_EQ(trace::formatItemString({"long_name", false}, "value", opts, opts.terminal_width), "\nlong_name = value");
}

TEST(TraceFormatting, GlobalColorEnvironmentConfiguresPresentationPolicy)
{
  EnvVarGuard color("UNI20_COLOR");
  EnvVarGuard no_color("NO_COLOR");
  no_color.set("1");

  color.set("no");
  trace::FormattingOptions disabled;
  EXPECT_EQ(disabled.presentation_policy().color, uni20::presentation::color_mode::never);
  EXPECT_FALSE(disabled.should_show_color());

  color.set("yes");
  trace::FormattingOptions enabled;
  EXPECT_EQ(enabled.presentation_policy().color, uni20::presentation::color_mode::always);
  EXPECT_TRUE(enabled.should_show_color());
}

TEST(TraceFormatting, MdspanValuesUsePresentationTensorArt)
{
  auto opts = make_test_options();
  std::array<int, 6> data{1, 20, 300, 4000, 5, 60};
  stdex::mdspan<int, stdex::extents<std::size_t, 2, 3>> matrix(data.data());

  EXPECT_EQ(trace::formatValue(matrix, opts), "shape=(2, 3)\n"
                                              "\xE2\x8E\xA1    1 20 300 \xE2\x8E\xA4\n"
                                              "\xE2\x8E\xA3 4000  5  60 \xE2\x8E\xA6");
}

TEST(TraceFormatting, MdspanRealAndComplexValuesUseTracePrecision)
{
  auto opts = make_test_options();
  opts.fp_precision_float64 = 2;

  std::array<double, 4> real_data{1.0, 2.5, 10.25, -3.0};
  stdex::mdspan<double, stdex::extents<std::size_t, 2, 2>> real_matrix(real_data.data());

  EXPECT_EQ(trace::formatValue(real_matrix, opts), "shape=(2, 2)\n"
                                                  "\xE2\x8E\xA1  1.00  2.50 \xE2\x8E\xA4\n"
                                                  "\xE2\x8E\xA3 10.25 -3.00 \xE2\x8E\xA6");

  opts.fp_precision_float64 = 1;
  std::array<std::complex<double>, 2> complex_data{std::complex<double>{1.0, -2.5},
                                                  std::complex<double>{0.0, 3.0}};
  stdex::mdspan<std::complex<double>, stdex::extents<std::size_t, 2>> complex_vector(complex_data.data());

  EXPECT_EQ(trace::formatValue(complex_vector, opts), "shape=(2)\n[ 1.0-2.5i 0.0+3.0i ]");
}

TEST(TraceFormatting, MdspanValuesCanSelectPresentationMatrixAxes)
{
  auto opts = make_test_options();
  opts.mdspan_format_policy().matrix_axes = uni20::presentation::mdspan_matrix_axes{0, 2};

  std::array<int, 12> data{};
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<int>(i);

  stdex::mdspan<int, stdex::extents<std::size_t, 2, 3, 2>> tensor(data.data());

  EXPECT_EQ(trace::formatValue(tensor, opts), "shape=(2, 3, 2)\n"
                                             "slice [:, 0, :]\n"
                                             "\xE2\x8E\xA1 0 1 \xE2\x8E\xA4\n"
                                             "\xE2\x8E\xA3 6 7 \xE2\x8E\xA6\n"
                                             "\n"
                                             "slice [:, 1, :]\n"
                                             "\xE2\x8E\xA1 2 3 \xE2\x8E\xA4\n"
                                             "\xE2\x8E\xA3 8 9 \xE2\x8E\xA6\n"
                                             "\n"
                                             "slice [:, 2, :]\n"
                                             "\xE2\x8E\xA1  4  5 \xE2\x8E\xA4\n"
                                             "\xE2\x8E\xA3 10 11 \xE2\x8E\xA6");
}
