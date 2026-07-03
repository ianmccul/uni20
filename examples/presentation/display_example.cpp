#include <uni20/common/display.hpp>
#include <uni20/common/presentation.hpp>

#include <utility>

namespace
{
namespace display = uni20::display;
namespace presentation = uni20::presentation;
} // namespace

int main()
{
  display::info("starting DMRG setup for {} sites", 32);
  display::success("loaded {} Hamiltonian terms", 3);
  display::warning("using dense validation path for the first sweep");
  display::partial("validated {} of {} optional inputs", 2, 3);
  display::deferred("GPU kernel selection will be decided at runtime");
  display::skipped("CUDA backend unavailable in this build");

  display::emit(display::status_cell(presentation::semantic_glyph::info,
                                     "status cells are non-emitting values until passed to display::emit"),
                display::stream::out);

  auto Highlight = presentation::style("Cyan;Bold");
  auto Good = presentation::style("Green;Bold");
  auto Caution = presentation::style("Yellow;Bold");
  display::emit(Highlight("custom style objects format values: residual {:.2e}", 3.2e-7), display::stream::out);

  presentation::report_builder setup("DMRG setup");
  setup.status(presentation::semantic_glyph::success, "ready")
      .field("sites", 32)
      .field("target sweeps", 8)
      .field("max bond", 256)
      .field("backend", "cpu");

  setup.table("Hamiltonian terms")
      .grid()
      .column("term", presentation::table_alignment::left)
      .column("coefficient", presentation::table_alignment::decimal)
      .column("range", presentation::table_alignment::left)
      .row("Sz Sz", "1.0", "nearest neighbor")
      .row("S+ S-", "0.5", "nearest neighbor")
      .row("S- S+", "0.5", "nearest neighbor");

  display::emit(std::move(setup), display::stream::out);

  presentation::report_builder summary("Review disposition glyphs");
  summary.table("Status meanings")
      .grid()
      .column("state", presentation::table_alignment::left)
      .column("meaning", presentation::table_alignment::left)
      .row("success", "completed or converged")
      .row("failure", "failed or still open")
      .row("partial", "some progress, not complete")
      .row("deferred", "intentionally postponed")
      .row("skipped", "not applicable or unavailable");

  display::emit(std::move(summary), display::stream::out);

  auto sweeps = display::table("Streaming DMRG sweeps");
  sweeps.wrap_width(72)
      .column("sweep", display::width::fixed(5))
      .column("energy", display::width::fixed(18), presentation::table_alignment::decimal)
      .column("dE", display::width::fixed(12), presentation::table_alignment::decimal)
      .column("bond", display::width::fixed(6))
      .column("status", display::width::share(1), presentation::table_alignment::left);

  sweeps.row(1, "-12.345678901234", "-", 128, Good(presentation::semantic_glyph::success, "accepted"));
  sweeps.row(2, "-12.456789012345", "-1.111e-1", 256,
             Caution(presentation::semantic_glyph::warning, "bond cap reached; truncation error 3.2e-7"));
  sweeps.row(3, "-12.467000000000", "-1.021e-2", 256, Good(presentation::semantic_glyph::success, "accepted"));
}
