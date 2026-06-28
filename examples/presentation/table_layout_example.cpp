#include "presentation_example_common.hpp"

#include <algorithm>
#include <cstddef>
#include <fmt/core.h>
#include <limits>

namespace
{
namespace demo = uni20::examples::presentation_demo;
namespace presentation = uni20::presentation;

[[nodiscard]] presentation::report_builder solver_report()
{
  presentation::report_builder report("Krylov solve");
  report.status(presentation::semantic_glyph::success, "converged")
      .field("matrix", "demo")
      .field("dimension", 128)
      .field("tolerance", "1.0e-12");

  report.table("Solver Summary")
      .grid(presentation::table_rule_style::double_line)
      .column("solver", presentation::table_alignment::left)
      .column("matvecs")
      .column("ritz value")
      .column("residual")
      .row("native", 185, "-1.732050807568877", "1.0e-15")
      .row("arpack", 238, "-1.732050807568102", "8.0e-13")
      .separator()
      .row({{"notes", 1},
            {"spanning cells are useful for solver diagnostics that need a short label followed by wider prose.", 3,
             presentation::table_alignment::left}});

  return report;
}

[[nodiscard]] presentation::report_builder advanced_table_report()
{
  presentation::report_builder report("Advanced table layout");
  report.status(presentation::semantic_glyph::info, "spans, manual rules, mixed borders, and decimal alignment");

  report.table("Phase ledger")
      .outer_border()
      .column_separators()
      .header_separator()
      .top_separator(presentation::table_rule_style::double_line)
      .column("phase", presentation::table_alignment::left)
      .column("description", presentation::table_alignment::left)
      .column("cost", presentation::table_alignment::decimal)
      .column("state", presentation::table_alignment::center)
      .row("parse", "read sparse blocks", "1.250", "ok")
      .row("factor", "symbolic setup", "12", "cached")
      .separator()
      .row({{"solve", 1},
            {"Krylov restart used a spanning note cell; the separator above shows where vertical rules stop.", 2,
             presentation::table_alignment::left},
            {"warn", 1, presentation::table_alignment::center}})
      .separator(presentation::table_rule_style::double_line)
      .row("cleanup", "release temporaries", "0.03125", "done");

  report.table("Decimal alignment")
      .grid()
      .column("quantity", presentation::table_alignment::left)
      .column("estimate", presentation::table_alignment::decimal)
      .column("observed", presentation::table_alignment::decimal)
      .column("note", presentation::table_alignment::left)
      .row("residual", "1.0e-15", "8.0e-13", "scientific strings still align before the decimal")
      .row("runtime", "12", "12.375", "integers align as if the decimal follows the value")
      .row("overflow", presentation::format_real(std::numeric_limits<double>::infinity(), {}),
           presentation::format_real(-std::numeric_limits<double>::infinity(), {}),
           "non-finite real values use deterministic lowercase spelling")
      .row("invalid", presentation::format_real(std::numeric_limits<double>::quiet_NaN(), {}), "nan",
           "NaN renders as nan and centers in decimal columns")
      .row({{"status", 1},
            {"-", 1, presentation::table_alignment::center},
            {"-", 1, presentation::table_alignment::center},
            {"per-cell center overrides", 1}});

  return report;
}

[[nodiscard]] presentation::report_builder fixed_width_wrapping_report()
{
  presentation::report_builder report("Fixed-width wrapped table");
  report.status(presentation::semantic_glyph::info, "wrap_width = 44 display cells");

  report.table("Cell wrapping")
      .grid()
      .column("case", presentation::table_alignment::left)
      .column("content", presentation::table_alignment::left)
      .row("ascii", "A long diagnostic message wraps inside this cell instead of after the table is rendered.")
      .row("unicode", "display width counts \xE4\xB8\xAD\xE6\x96\x87 and \xF0\x9F\x9A\x80 before choosing where each "
                      "cell line breaks.")
      .row("numbers", "right-aligned numeric columns can stay compact while nearby text columns wrap.");

  return report;
}

[[nodiscard]] presentation::report_builder terminal_width_wrapping_report(std::size_t terminal_width)
{
  presentation::report_builder report("Terminal-width table");
  report.status(presentation::semantic_glyph::info,
                fmt::format("detected width = {} display cells; resize the terminal and rerun", terminal_width));

  report.table("Resize experiment")
      .grid()
      .column("item", presentation::table_alignment::left)
      .column("observation", presentation::table_alignment::left)
      .column("value")
      .row("policy", "The table consumes output_policy::wrap_width before final rendering, so borders remain intact.",
           "stable")
      .row("wide terminal", "More width is assigned to natural columns, reducing the number of wrapped cell lines.",
           "fewer rows")
      .row("narrow terminal", "The widest cells shrink first; entries with whitespace are preferred for wrapping.",
           "more rows")
      .row("glyphs",
           "Semantic box drawing still falls back through the selected glyph policy: \xE2\x9C\x93 "
           "\xE2\x86\x92 \xE4\xB8\xAD\xE6\x96\x87 \xF0\x9F\x9A\x80.",
           "cells")
      .row("width: check", "isolated checkmark: \xE2\x9C\x93", "probe")
      .row("width: arrow", "isolated right arrow: \xE2\x86\x92", "probe")
      .row("width: cjk", "isolated CJK text: \xE4\xB8\xAD\xE6\x96\x87", "probe")
      .row("width: emoji", "isolated emoji: \xF0\x9F\x9A\x80", "probe")
      .row("width: mixed", "mixed sequence: \xE2\x9C\x93 \xE2\x86\x92 \xE4\xB8\xAD\xE6\x96\x87 \xF0\x9F\x9A\x80",
           "probe");

  return report;
}

} // namespace

int main()
{
  auto unicode_policy = demo::forced_demo_policy(presentation::glyph_set::unicode, 80);

  fmt::print("{}\n", presentation::render_plain(solver_report(), unicode_policy));
  fmt::print("{}\n", presentation::render_plain(advanced_table_report(), unicode_policy));

  auto fixed_table_policy = unicode_policy;
  fixed_table_policy.wrap_width = 44;
  fmt::print("{}\n", presentation::render_plain(fixed_width_wrapping_report(), fixed_table_policy));

  auto terminal_table_policy = unicode_policy;
  terminal_table_policy.wrap_width = std::max<std::size_t>(demo::detected_terminal_width(), 24);
  fmt::print("{}\n", presentation::render_plain(terminal_width_wrapping_report(*terminal_table_policy.wrap_width),
                                                terminal_table_policy));
}
