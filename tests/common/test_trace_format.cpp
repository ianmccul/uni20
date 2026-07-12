#include <uni20/common/mdspan.hpp>
#include <uni20/common/trace.hpp>

#include "env_var_guard.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{
using uni20::test::EnvVarGuard;

struct FileCloser
{
    void operator()(std::FILE* file) const
    {
      if (file != nullptr)
      {
        std::fclose(file);
      }
    }
};

using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

trace::FormattingOptions make_test_options()
{
  auto opts = trace::get_formatting_options("trace-format-test");
  opts.set_color_output(trace::FormattingOptions::ColorOptions::no);
  return opts;
}

FilePtr make_terminal_stream()
{
  int const fd = ::open("/dev/ptmx", O_RDWR | O_NOCTTY);
  if (fd < 0)
  {
    return {};
  }

  std::FILE* stream = ::fdopen(fd, "w");
  if (stream == nullptr)
  {
    ::close(fd);
    return {};
  }

  return FilePtr{stream};
}

bool contains_ansi(std::string const& text) { return text.find("\033[") != std::string::npos; }
} // namespace

TEST(TraceFormatting, FloatingPointPrecision)
{
  auto opts = make_test_options();
  opts.fp_precision_float32 = 2;
  opts.fp_precision_float64 = 4;
  opts.mdspan_format_policy().numeric.long_double_precision = 3;

  EXPECT_EQ("3.14", trace::formatValue(3.14159f, opts));
  EXPECT_EQ("2.7183", trace::formatValue(2.718281828, opts));
  EXPECT_EQ("1.23-6.79i", trace::formatValue(uni20::complex<float>{1.2345f, -6.789f}, opts));
  EXPECT_EQ("-0.1250+42.5000i", trace::formatValue(uni20::complex<double>{-0.125, 42.5}, opts));
  EXPECT_EQ("1.250", trace::formatValue(1.25L, opts));
  EXPECT_EQ("1.250-0.500i", trace::formatValue(uni20::complex<long double>{1.25L, -0.5L}, opts));
}

#if UNI20_HAS_FLOAT128
TEST(TraceFormatting, Float128UsesUni20ScalarFormattingAndConfiguredPrecision)
{
  using Real = uni20::float128;
  using Complex = uni20::complex<Real>;

  auto opts = make_test_options();
  opts.fp_precision_float128 = 36;
  Real const value = uni20::parse_real<Real>("1.000000000000000000000000000000001");
  std::string const formatted = trace::formatValue(value, opts);

  EXPECT_EQ(formatted, uni20::presentation::format_real(value, opts.numeric_format_policy()));
  EXPECT_EQ(uni20::parse_real<Real>(formatted), value);

  Complex const complex_value{value, -value};
  EXPECT_EQ(trace::formatValue(complex_value, opts),
            uni20::presentation::format_complex(complex_value, opts.numeric_format_policy()));
}
#endif

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

TEST(TraceFormatting, RectangularNestedContainersUsePresentationTensorArt)
{
  auto opts = make_test_options();
  std::vector<std::vector<int>> matrix{{1, 20, 300}, {4000, 5, 60}};

  EXPECT_EQ(trace::formatContainerToString(trace::formatValue(matrix, opts), opts),
            "shape=(2, 3)\n"
            "\xE2\x8E\xA1    1 20 300 \xE2\x8E\xA4\n"
            "\xE2\x8E\xA3 4000  5  60 \xE2\x8E\xA6");
}

TEST(TraceFormatting, RaggedNestedContainersKeepListFormatting)
{
  auto opts = make_test_options();
  std::vector<std::vector<int>> ragged{{1, 2}, {3}};

  EXPECT_EQ(trace::formatContainerToString(trace::formatValue(ragged, opts), opts), "[ [ 1, 2 ],\n  [ 3 ] ]");
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

  opts.set_color_output(uni20::presentation::color_mode::always);
  EXPECT_TRUE(contains_ansi(opts.format_style("TRACE", "TRACE")));

  opts.set_color_output(uni20::presentation::color_mode::never);
  EXPECT_FALSE(contains_ansi(opts.format_style("TRACE", "TRACE")));
}

TEST(TraceFormatting, PlainFileModeSuppressesAutoColor)
{
  auto file = FilePtr{std::tmpfile()};
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

  EXPECT_EQ(opts.format_glyph(uni20::presentation::semantic_glyph::warning, "ERROR"), "\xE2\x9A\xA0");
  EXPECT_EQ(opts.format_glyph(uni20::presentation::semantic_glyph::arrow_right, "TRACE"), "\xE2\x86\x92");

  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::emoji;
  EXPECT_EQ(opts.format_glyph(uni20::presentation::semantic_glyph::warning, "ERROR"), "\xE2\x9A\xA0\xEF\xB8\x8F");

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
  EXPECT_NE(opts.render(check).find("\xF0\x9F\x9A\xA8 CHECK at "), std::string::npos);
  EXPECT_NE(opts.render(panic).find("\xF0\x9F\x9A\xA8 PANIC at "), std::string::npos);

  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::unicode;
  check = trace::detail::make_diagnostic_header(opts, "CHECK", "CHECK", __FILE__, __LINE__);
  panic = trace::detail::make_diagnostic_header(opts, "PANIC", "PANIC", __FILE__, __LINE__);
  EXPECT_NE(opts.render(check).find("\xE2\x80\xBC CHECK at "), std::string::npos);
  EXPECT_NE(opts.render(panic).find("\xE2\x80\xBC PANIC at "), std::string::npos);

  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::ascii;
  auto error = trace::detail::make_diagnostic_header(opts, "ERROR", "ERROR", __FILE__, __LINE__);
  auto precondition = trace::detail::make_diagnostic_header(opts, "PRECONDITION", "PRECONDITION", __FILE__, __LINE__);
  EXPECT_NE(opts.render(error).find("[FAIL] ERROR at "), std::string::npos);
  EXPECT_NE(opts.render(precondition).find("[FATAL] PRECONDITION at "), std::string::npos);
}

TEST(TraceFormatting, GlobalEnvironmentConfiguresPresentationPolicy)
{
  EnvVarGuard glyphs("UNI20_GLYPHS");
  EnvVarGuard charset("UNI20_CHARSET");
  EnvVarGuard columns("COLUMNS");
  EnvVarGuard float128_precision("UNI20_FP_PRECISION_FLOAT128");

  glyphs.set("ascii");
  charset.set("ascii_escape");
  columns.set("12");
  float128_precision.set("27");

  trace::FormattingOptions opts;
  opts.set_color_output(trace::FormattingOptions::ColorOptions::no);

  EXPECT_EQ(opts.presentation_policy().glyphs, uni20::presentation::glyph_set::ascii);
  EXPECT_EQ(opts.presentation_policy().charset, uni20::presentation::text_charset::ascii_escape);
  EXPECT_EQ(opts.terminal_width, 12);
  EXPECT_EQ(opts.fp_precision_float128, 27);
  EXPECT_EQ(opts.numeric_format_policy().float128_precision, 27);
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
  std::array<uni20::complex<double>, 2> complex_data{uni20::complex<double>{1.0, -2.5},
                                                     uni20::complex<double>{0.0, 3.0}};
  stdex::mdspan<uni20::complex<double>, stdex::extents<std::size_t, 2>> complex_vector(complex_data.data());

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

TEST(TraceFormatting, NormalMdspanPreviewFitsTerminalWidthWithoutDump)
{
  auto dump_dir =
      std::filesystem::temp_directory_path() /
      fmt::format("uni20-trace-format-test-{}", std::chrono::steady_clock::now().time_since_epoch().count());
  EnvVarGuard dump_dir_env("UNI20_TRACE_DUMP_DIR", dump_dir.string());
  EnvVarGuard dump_env("UNI20_TRACE_DUMP");
  dump_env.unset();

  auto opts = make_test_options();
  opts.terminal_width = 32;
  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::ascii;
  opts.mdspan_preview_policy().full_element_limit = 4;
  opts.mdspan_preview_policy().edge_items = 1;

  std::array<int, 20> data{};
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<int>(i);

  stdex::mdspan<int, stdex::extents<std::size_t, 4, 5>> matrix(data.data());
  auto text = trace::formatParameterListTextForContext("matrix", opts, trace::trace_format_context::normal, matrix);
  auto const rendered = opts.render(text);

  EXPECT_NE(rendered.find("preview elided"), std::string::npos) << rendered;
  EXPECT_EQ(rendered.find("full data:"), std::string::npos) << rendered;

  std::size_t line_start = 0;
  while (line_start <= rendered.size())
  {
    auto const line_end = rendered.find('\n', line_start);
    auto const line = line_end == std::string::npos
                          ? std::string_view(rendered).substr(line_start)
                          : std::string_view(rendered).substr(line_start, line_end - line_start);
    if (!line.empty())
    {
      EXPECT_LE(uni20::presentation::display_width(line, opts.presentation_policy()),
                static_cast<std::size_t>(opts.terminal_width))
          << line;
    }
    if (line_end == std::string::npos) break;
    line_start = line_end + 1;
  }

  if (std::filesystem::exists(dump_dir))
  {
    EXPECT_TRUE(std::filesystem::is_empty(dump_dir));
    std::filesystem::remove_all(dump_dir);
  }
}

TEST(TraceFormatting, FatalMdspanPreviewWritesFullDump)
{
  auto dump_dir =
      std::filesystem::temp_directory_path() /
      fmt::format("uni20-trace-format-test-{}", std::chrono::steady_clock::now().time_since_epoch().count());
  EnvVarGuard dump_dir_env("UNI20_TRACE_DUMP_DIR", dump_dir.string());
  EnvVarGuard dump_env("UNI20_TRACE_DUMP");
  dump_env.unset();

  auto opts = make_test_options();
  opts.terminal_width = 32;
  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::ascii;
  opts.mdspan_preview_policy().full_element_limit = 4;
  opts.mdspan_preview_policy().edge_items = 1;

  std::array<int, 20> data{};
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<int>(i);

  stdex::mdspan<int, stdex::extents<std::size_t, 4, 5>> matrix(data.data());
  auto text = trace::formatParameterListTextForContext("matrix", opts, trace::trace_format_context::fatal, matrix);
  auto const rendered = opts.render(text);

  EXPECT_NE(rendered.find("preview elided"), std::string::npos) << rendered;
  auto const marker = std::string("full data: ");
  auto const path_pos = rendered.find(marker);
  ASSERT_NE(path_pos, std::string::npos) << rendered;

  auto path = rendered.substr(path_pos + marker.size());
  if (auto const newline = path.find('\n'); newline != std::string::npos) path.resize(newline);
  while (!path.empty() && std::isspace(static_cast<unsigned char>(path.back())))
    path.pop_back();

  ASSERT_TRUE(std::filesystem::exists(path)) << path;
  std::ifstream in(path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  auto const dump = buffer.str();
  EXPECT_NE(dump.find("# shape=(4, 5)"), std::string::npos) << dump;
  EXPECT_NE(dump.find("[3, 4]\t19"), std::string::npos) << dump;
  EXPECT_FALSE(contains_ansi(dump)) << dump;

  std::filesystem::remove_all(dump_dir);
}

TEST(TraceFormatting, MdspanPreviewHighlightsNonFiniteScalarsWhenColorEnabled)
{
  auto opts = make_test_options();
  opts.set_color_output(uni20::presentation::color_mode::always);
  opts.presentation_policy().glyphs = uni20::presentation::glyph_set::ascii;

  std::array<double, 4> data{1.0, std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()};
  stdex::mdspan<double, stdex::extents<std::size_t, 2, 2>> matrix(data.data());

  auto plain_opts = make_test_options();
  plain_opts.set_color_output(uni20::presentation::color_mode::never);
  plain_opts.presentation_policy().glyphs = uni20::presentation::glyph_set::ascii;

  auto const plain = trace::formatValue(matrix, plain_opts);
  auto const colored = trace::formatValue(matrix, opts);

  EXPECT_FALSE(contains_ansi(plain));
  EXPECT_TRUE(contains_ansi(colored));
  EXPECT_NE(colored.find("\033[1;31minf\033[0m"), std::string::npos);
  EXPECT_NE(colored.find("\033[1;31m-inf\033[0m"), std::string::npos);
  EXPECT_NE(colored.find("\033[1;31mnan\033[0m"), std::string::npos);
  EXPECT_EQ(uni20::presentation::display_width(colored, opts.presentation_policy()),
            uni20::presentation::display_width(plain, plain_opts.presentation_policy()));
}
