#include "presentation_example_common.hpp"

#include <uni20/common/mdspan.hpp>
#include <uni20/common/presentation_mdspan.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fmt/core.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{
namespace demo = uni20::examples::presentation_demo;
namespace presentation = uni20::presentation;

template <typename MDS>
void print_full(std::string const& title, MDS const& mds, presentation::output_policy const& policy,
                presentation::mdspan_format_options const& options = {})
{
  fmt::print("{}\n{}\n\n", title, presentation::format_mdspan(mds, policy, options));
}

template <typename MDS, typename Formatter>
void print_custom(std::string const& title, MDS const& mds, presentation::output_policy const& policy,
                  Formatter&& formatter, presentation::mdspan_format_options const& options = {})
{
  fmt::print("{}\n{}\n\n", title,
             presentation::format_mdspan(mds, policy, std::forward<Formatter>(formatter), options));
}

template <typename MDS>
void print_preview(std::string const& title, MDS const& mds, presentation::output_policy const& policy,
                   presentation::mdspan_preview_options const& options)
{
  auto const preview = presentation::format_mdspan_preview(mds, policy, options);
  fmt::print("{}\n{}\n\n", title, preview.text);
}

[[nodiscard]] std::vector<int> sequence(std::size_t count, int offset = 0)
{
  std::vector<int> values(count);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = offset + static_cast<int>(i);
  return values;
}

[[nodiscard]] std::vector<double> mixed_sign_rank3_data(std::size_t slices, std::size_t rows, std::size_t columns)
{
  std::vector<double> values(slices * rows * columns);
  for (std::size_t slice = 0; slice < slices; ++slice)
  {
    for (std::size_t row = 0; row < rows; ++row)
    {
      for (std::size_t column = 0; column < columns; ++column)
      {
        auto const linear = (slice * rows + row) * columns + column;
        auto const sign = ((slice + row + column) % 2 == 0) ? 1.0 : -1.0;
        values[linear] = sign * (static_cast<double>(linear + 1) / 7.0);
      }
    }
  }
  return values;
}

[[nodiscard]] std::size_t terminal_edge_budget(std::size_t width) { return std::clamp<std::size_t>(width / 9, 4, 32); }

[[nodiscard]] std::size_t terminal_slice_budget(std::size_t width) { return width >= 120 ? 6 : 4; }

} // namespace

int main()
{
  auto const terminal_width = std::max<std::size_t>(demo::detected_terminal_width(), 32);
  auto const terminal_edge_items = terminal_edge_budget(terminal_width);
  auto const terminal_max_slices = terminal_slice_budget(terminal_width);
  auto terminal_policy = demo::terminal_demo_policy(terminal_width);

  auto fixed_policy = terminal_policy;
  fixed_policy.wrap_width = 52;

  fmt::print("presentation mdspan example\n");
  fmt::print("terminal preview width: {}\n", terminal_width);
  fmt::print("terminal preview budget: edge_items <= {}, max_slices = {}\n", terminal_edge_items, terminal_max_slices);
  fmt::print("{}\n\n", demo::width_ruler(std::min<std::size_t>(terminal_width, 120)));

  std::array<int, 8> vector_data{0, 1, 1, 2, 3, 5, 8, 13};
  stdex::mdspan<int, stdex::extents<std::size_t, 8>> vector(vector_data.data());
  print_full("rank-1 vector, exhaustive", vector, terminal_policy);

  std::array<int, 12> matrix_data{};
  for (std::size_t i = 0; i < matrix_data.size(); ++i)
    matrix_data[i] = static_cast<int>(10 * (i / 4) + (i % 4));

  stdex::mdspan<int, stdex::extents<std::size_t, 3, 4>> matrix(matrix_data.data());
  print_full("rank-2 matrix, exhaustive", matrix, terminal_policy);

  std::array<std::string, 6> label_data{"alpha", "beta", "gamma", "delta", "epsilon", "zeta"};
  stdex::mdspan<std::string, stdex::extents<std::size_t, 2, 3>> labels(label_data.data());
  print_custom("rank-2 string labels, custom formatter", labels, terminal_policy,
               [](std::string const& value) { return value; });

  std::array<int, 24> rank3_data{};
  for (std::size_t i = 0; i < rank3_data.size(); ++i)
    rank3_data[i] = static_cast<int>(i);

  stdex::mdspan<int, stdex::extents<std::size_t, 2, 3, 4>> rank3(rank3_data.data());
  print_full("rank-3 tensor, exhaustive default slices", rank3, terminal_policy);

  presentation::mdspan_format_options axis_options;
  axis_options.matrix_axes = presentation::mdspan_matrix_axes{0, 2};
  print_full("rank-3 tensor, axes 0 and 2 as matrix axes", rank3, terminal_policy, axis_options);

  using complex_t = uni20::complex<double>;
  std::array<complex_t, 6> complex_data{complex_t{1.0, -2.5},
                                        complex_t{0.0, 3.0},
                                        complex_t{-4.25, 0.125},
                                        complex_t{2.0, -0.0},
                                        complex_t{std::numeric_limits<double>::infinity(), 1.0},
                                        complex_t{-0.5, std::numeric_limits<double>::quiet_NaN()}};
  stdex::mdspan<complex_t, stdex::extents<std::size_t, 2, 3>> complex_matrix(complex_data.data());

  presentation::mdspan_format_options complex_options;
  complex_options.numeric.notation = presentation::real_notation::fixed;
  complex_options.numeric.float64_precision = 2;
  print_full("rank-2 complex matrix, fixed precision", complex_matrix, terminal_policy, complex_options);

  auto large_matrix_data = sequence(18 * 22, 1000);
  using dynamic_matrix_extents = stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent>;
  stdex::mdspan<int, dynamic_matrix_extents, stdex::layout_right> large_matrix(large_matrix_data.data(), 18, 22);

  presentation::mdspan_preview_options fixed_preview;
  fixed_preview.full_element_limit = 48;
  fixed_preview.edge_items = 2;
  fixed_preview.max_slices = 4;
  print_preview("large dynamic matrix preview, fixed width 52", large_matrix, fixed_policy, fixed_preview);

  auto rank3_preview_data = mixed_sign_rank3_data(6, 9, 64);
  using dynamic_rank3_extents =
      stdex::extents<std::size_t, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>;
  stdex::mdspan<double, dynamic_rank3_extents, stdex::layout_right> large_rank3(rank3_preview_data.data(), 6, 9, 64);

  presentation::mdspan_preview_options terminal_preview;
  terminal_preview.full_element_limit = 64;
  terminal_preview.edge_items = terminal_edge_items;
  terminal_preview.max_slices = terminal_max_slices;
  terminal_preview.format.numeric.notation = presentation::real_notation::fixed;
  terminal_preview.format.numeric.float64_precision = 3;
  print_preview("large rank-3 preview, terminal width", large_rank3, terminal_policy, terminal_preview);

  auto narrow_policy = terminal_policy;
  narrow_policy.wrap_width = 28;
  presentation::mdspan_preview_options narrow_preview = fixed_preview;
  narrow_preview.edge_items = 1;
  print_preview("same large matrix preview, narrow width 28", large_matrix, narrow_policy, narrow_preview);
}
