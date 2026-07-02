#pragma once

#include "presentation.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::presentation
{

/// \brief Selects whether mdspan/tensor art includes shape metadata.
enum class mdspan_shape_mode
{
  none,
  prefix
};

/// \brief Matrix axes used when rendering rank-2-or-higher tensors.
struct mdspan_matrix_axes
{
    std::size_t row = 0;
    std::size_t column = 1;
};

/// \brief Formatting controls for mdspan and tensor display art.
///
/// Full tensor rendering is intentionally exhaustive. Preview, clipping, or
/// elision should be added as an explicit non-default mode rather than changing
/// the meaning of `format_mdspan`.
struct mdspan_format_options
{
    mdspan_shape_mode shape = mdspan_shape_mode::prefix;
    bool label_slices = true;
    std::optional<mdspan_matrix_axes> matrix_axes = std::nullopt;
    numeric_format_options numeric;
};

/// \brief Bounded preview controls for mdspan and tensor display.
struct mdspan_preview_options
{
    mdspan_format_options format;
    std::size_t full_element_limit = 256;
    std::size_t edge_items = 3;
    std::size_t max_slices = 4;
};

/// \brief Result of bounded mdspan preview rendering.
struct mdspan_preview_result
{
    std::string text;
    bool elided = false;
    std::vector<std::size_t> extents;
    std::size_t element_count = 0;
};

/// \brief Detect mdspan-like objects without depending on trace or tensor.
/// \tparam MDS Candidate mdspan type.
template <typename MDS>
concept mdspan_like = requires(std::remove_cvref_t<MDS> const& mds) {
  typename std::remove_cvref_t<MDS>::element_type;
  typename std::remove_cvref_t<MDS>::index_type;
  { std::remove_cvref_t<MDS>::rank() } -> std::convertible_to<std::size_t>;
  { mds.extent(std::size_t{0}) } -> std::convertible_to<std::size_t>;
};

/// \brief Render every mdspan/tensor element as display-cell-aligned text art.
/// \tparam MDS Mdspan-like object type.
/// \tparam ElementFormatter Callable that formats each scalar element to UTF-8 text.
/// \param mds Mdspan-like object to render.
/// \param policy Output policy controlling glyphs, charset fallback, and width.
/// \param formatter Callable invoked as `formatter(element)`.
/// \param options Matrix/tensor art controls.
/// \return Rendered matrix or higher-order tensor art containing every element.
template <mdspan_like MDS, typename ElementFormatter>
  requires(!std::same_as<std::remove_cvref_t<ElementFormatter>, mdspan_format_options>)
[[nodiscard]] std::string format_mdspan(MDS const& mds, output_policy const& policy, ElementFormatter&& formatter,
                                        mdspan_format_options const& options = {});

/// \brief Render every mdspan/tensor element using default scalar numeric formatting.
/// \tparam MDS Mdspan-like object type.
/// \param mds Mdspan-like object to render.
/// \param policy Output policy controlling glyphs, charset fallback, and width.
/// \param options Matrix/tensor art and numeric formatting controls.
/// \return Rendered matrix or higher-order tensor art containing every element.
template <mdspan_like MDS>
[[nodiscard]] std::string format_mdspan(MDS const& mds, output_policy const& policy,
                                        mdspan_format_options const& options = {});

/// \brief Render a bounded mdspan/tensor preview without visiting omitted elements.
/// \tparam MDS Mdspan-like object type.
/// \tparam ElementFormatter Callable that formats each scalar element to UTF-8 text.
/// \param mds Mdspan-like object to render.
/// \param policy Output policy controlling glyphs, charset fallback, and width.
/// \param formatter Callable invoked only for displayed elements.
/// \param options Preview, matrix/tensor art, and numeric controls.
/// \return Rendered preview plus metadata describing whether values were elided.
template <mdspan_like MDS, typename ElementFormatter>
  requires(!std::same_as<std::remove_cvref_t<ElementFormatter>, mdspan_preview_options>)
[[nodiscard]] mdspan_preview_result format_mdspan_preview(MDS const& mds, output_policy const& policy,
                                                          ElementFormatter&& formatter,
                                                          mdspan_preview_options const& options = {});

/// \brief Render a bounded mdspan/tensor preview using default scalar numeric formatting.
/// \tparam MDS Mdspan-like object type.
/// \param mds Mdspan-like object to render.
/// \param policy Output policy controlling glyphs, charset fallback, and width.
/// \param options Preview, matrix/tensor art, and numeric controls.
/// \return Rendered preview plus metadata describing whether values were elided.
template <mdspan_like MDS>
[[nodiscard]] mdspan_preview_result format_mdspan_preview(MDS const& mds, output_policy const& policy,
                                                          mdspan_preview_options const& options = {});

namespace detail
{

template <typename MDS> using mdspan_type = std::remove_cvref_t<MDS>;

template <typename MDS> inline constexpr std::size_t mdspan_rank_v = mdspan_type<MDS>::rank();

template <typename MDS> using mdspan_index_type = typename mdspan_type<MDS>::index_type;

[[nodiscard]] inline std::string render_piece(semantic_glyph glyph, output_policy const& policy)
{
  return render_text(render_glyph(glyph, policy), policy);
}

[[nodiscard]] inline std::string render_piece(std::string_view text, output_policy const& policy)
{
  return render_text(text, policy);
}

template <typename MDS, std::size_t Rank>
decltype(auto) mdspan_at(MDS const& mds, std::array<mdspan_index_type<MDS>, Rank> const& indices)
{
  static_assert(Rank == mdspan_rank_v<MDS>);
  return mds[indices];
}

template <typename MDS, typename ElementFormatter, std::size_t Rank>
[[nodiscard]] std::string format_element(MDS const& mds, std::array<mdspan_index_type<MDS>, Rank> const& indices,
                                         output_policy const& policy, ElementFormatter& formatter)
{
  decltype(auto) value = mdspan_at(mds, indices);
  auto text = std::string(std::invoke(formatter, value));
  return render_text(style_nonfinite_scalar(std::move(text), value, policy), policy);
}

[[nodiscard]] inline std::string shape_string(std::vector<std::size_t> const& extents)
{
  std::string out = "shape=(";
  for (std::size_t i = 0; i < extents.size(); ++i)
  {
    if (i > 0) out += ", ";
    out += std::to_string(extents[i]);
  }
  out += ")";
  return out;
}

[[nodiscard]] inline std::string element_count_string(std::size_t element_count)
{
  return "elements=" + std::to_string(element_count);
}

[[nodiscard]] inline std::string preview_notice_string() { return "preview elided"; }

[[nodiscard]] inline std::size_t checked_product(std::vector<std::size_t> const& values)
{
  std::size_t product = 1;
  for (auto const value : values)
  {
    if (value != 0 && product > std::numeric_limits<std::size_t>::max() / value)
      return std::numeric_limits<std::size_t>::max();
    product *= value;
  }
  return product;
}

struct resolved_matrix_axes
{
    std::size_t row = 0;
    std::size_t column = 1;
    std::vector<std::size_t> slice_axes;
};

[[nodiscard]] inline resolved_matrix_axes resolve_matrix_axes(std::size_t rank, mdspan_format_options const& options)
{
  auto axes = options.matrix_axes.value_or(mdspan_matrix_axes{rank - 2, rank - 1});
  if (axes.row >= rank || axes.column >= rank)
  {
    throw std::invalid_argument("mdspan matrix axes must be valid tensor axes");
  }
  if (axes.row == axes.column)
  {
    throw std::invalid_argument("mdspan matrix row and column axes must be distinct");
  }

  resolved_matrix_axes result{axes.row, axes.column, {}};
  result.slice_axes.reserve(rank - 2);
  for (std::size_t axis = 0; axis < rank; ++axis)
  {
    if (axis != axes.row && axis != axes.column)
    {
      result.slice_axes.push_back(axis);
    }
  }
  return result;
}

template <typename MDS, std::size_t Rank>
[[nodiscard]] std::string slice_label(std::array<mdspan_index_type<MDS>, Rank> const& indices,
                                      resolved_matrix_axes const& axes)
{
  std::string out = "slice [";
  for (std::size_t i = 0; i < Rank; ++i)
  {
    if (i > 0) out += ", ";
    if (i == axes.row || i == axes.column)
    {
      out += ":";
    }
    else
    {
      out += std::to_string(static_cast<std::size_t>(indices[i]));
    }
  }
  out += "]";
  return out;
}

[[nodiscard]] inline std::string pad_cell(std::string const& value, std::size_t width, output_policy const& policy)
{
  auto const value_width = display_width(value, policy);
  if (value_width >= width) return value;
  return std::string(width - value_width, ' ') + value;
}

[[nodiscard]] inline std::string join_row(std::vector<std::string> const& row,
                                          std::vector<std::size_t> const& column_widths, output_policy const& policy)
{
  std::string out;
  for (std::size_t col = 0; col < row.size(); ++col)
  {
    if (col > 0) out.push_back(' ');
    out += pad_cell(row[col], column_widths[col], policy);
  }
  return out;
}

[[nodiscard]] inline std::string matrix_left(std::size_t row, std::size_t rows, output_policy const& policy)
{
  if (rows <= 1) return "[";
  if (row == 0) return render_piece(semantic_glyph::matrix_top_left, policy);
  if (row + 1 == rows) return render_piece(semantic_glyph::matrix_bottom_left, policy);
  return render_piece(semantic_glyph::matrix_middle_left, policy);
}

[[nodiscard]] inline std::string matrix_right(std::size_t row, std::size_t rows, output_policy const& policy)
{
  if (rows <= 1) return "]";
  if (row == 0) return render_piece(semantic_glyph::matrix_top_right, policy);
  if (row + 1 == rows) return render_piece(semantic_glyph::matrix_bottom_right, policy);
  return render_piece(semantic_glyph::matrix_middle_right, policy);
}

[[nodiscard]] inline std::string format_matrix_rows(std::vector<std::vector<std::string>> const& rows,
                                                    output_policy const& policy)
{
  if (rows.empty()) return "[]";

  std::size_t columns = 0;
  for (auto const& row : rows)
    columns = std::max(columns, row.size());

  if (columns == 0) return "[]";

  std::vector<std::size_t> column_widths(columns, 0);
  for (auto const& row : rows)
  {
    for (std::size_t col = 0; col < row.size(); ++col)
      column_widths[col] = std::max(column_widths[col], display_width(row[col], policy));
  }

  std::string out;
  for (std::size_t row = 0; row < rows.size(); ++row)
  {
    if (row > 0) out.push_back('\n');
    out += matrix_left(row, rows.size(), policy);
    out.push_back(' ');
    out += join_row(rows[row], column_widths, policy);
    out.push_back(' ');
    out += matrix_right(row, rows.size(), policy);
  }
  return out;
}

[[nodiscard]] inline std::size_t rendered_max_line_width(std::string_view text, output_policy const& policy)
{
  std::size_t max_width = 0;
  std::size_t line_start = 0;
  while (line_start <= text.size())
  {
    auto const line_end = text.find('\n', line_start);
    auto const line =
        line_end == std::string_view::npos ? text.substr(line_start) : text.substr(line_start, line_end - line_start);
    max_width = std::max(max_width, display_width(line, policy));
    if (line_end == std::string_view::npos) break;
    line_start = line_end + 1;
  }
  return max_width;
}

[[nodiscard]] inline bool fits_wrap_width(std::string_view text, output_policy const& policy)
{
  return !policy.wrap_width.has_value() || rendered_max_line_width(text, policy) <= *policy.wrap_width;
}

[[nodiscard]] inline std::vector<std::optional<std::size_t>> selected_positions(std::size_t extent,
                                                                                std::size_t edge_items)
{
  std::vector<std::optional<std::size_t>> positions;
  if (extent == 0) return positions;

  edge_items = std::max<std::size_t>(edge_items, 1);
  if (extent <= 2 * edge_items)
  {
    positions.reserve(extent);
    for (std::size_t i = 0; i < extent; ++i)
      positions.push_back(i);
    return positions;
  }

  positions.reserve(2 * edge_items + 1);
  for (std::size_t i = 0; i < edge_items; ++i)
    positions.push_back(i);
  positions.push_back(std::nullopt);
  for (std::size_t i = extent - edge_items; i < extent; ++i)
    positions.push_back(i);
  return positions;
}

[[nodiscard]] inline bool positions_elide(std::vector<std::optional<std::size_t>> const& positions)
{
  return std::any_of(positions.begin(), positions.end(), [](auto const& position) { return !position.has_value(); });
}

template <typename MDS, typename ElementFormatter, std::size_t Rank>
[[nodiscard]] std::vector<std::string> collect_vector(MDS const& mds, std::array<mdspan_index_type<MDS>, Rank>& indices,
                                                      output_policy const& policy, ElementFormatter& formatter,
                                                      std::size_t column_axis)
{
  std::vector<std::string> row;
  auto const cols = static_cast<std::size_t>(mds.extent(column_axis));
  row.reserve(cols);
  for (std::size_t col = 0; col < cols; ++col)
  {
    indices[column_axis] = static_cast<mdspan_index_type<MDS>>(col);
    row.push_back(format_element(mds, indices, policy, formatter));
  }
  return row;
}

template <typename MDS, typename ElementFormatter, std::size_t Rank>
[[nodiscard]] std::vector<std::string>
collect_preview_vector(MDS const& mds, std::array<mdspan_index_type<MDS>, Rank>& indices, output_policy const& policy,
                       ElementFormatter& formatter, std::size_t column_axis, std::size_t edge_items, bool& elided)
{
  std::vector<std::string> row;
  auto const cols = static_cast<std::size_t>(mds.extent(column_axis));
  auto const positions = selected_positions(cols, edge_items);
  elided = elided || positions_elide(positions);
  row.reserve(positions.size());
  auto const ellipsis = render_piece(semantic_glyph::ellipsis, policy);
  for (auto const& col : positions)
  {
    if (!col.has_value())
    {
      row.push_back(ellipsis);
      continue;
    }
    indices[column_axis] = static_cast<mdspan_index_type<MDS>>(*col);
    row.push_back(format_element(mds, indices, policy, formatter));
  }
  return row;
}

template <typename MDS, typename ElementFormatter, std::size_t Rank>
[[nodiscard]] std::vector<std::vector<std::string>>
collect_matrix(MDS const& mds, std::array<mdspan_index_type<MDS>, Rank>& indices, output_policy const& policy,
               ElementFormatter& formatter, resolved_matrix_axes const& axes)
{
  std::vector<std::vector<std::string>> rows;
  auto const row_count = static_cast<std::size_t>(mds.extent(axes.row));
  rows.reserve(row_count);
  for (std::size_t row = 0; row < row_count; ++row)
  {
    indices[axes.row] = static_cast<mdspan_index_type<MDS>>(row);
    rows.push_back(collect_vector(mds, indices, policy, formatter, axes.column));
  }
  return rows;
}

template <typename MDS, typename ElementFormatter, std::size_t Rank>
[[nodiscard]] std::vector<std::vector<std::string>>
collect_preview_matrix(MDS const& mds, std::array<mdspan_index_type<MDS>, Rank>& indices, output_policy const& policy,
                       ElementFormatter& formatter, resolved_matrix_axes const& axes, std::size_t edge_items,
                       bool& elided)
{
  std::vector<std::vector<std::string>> rows;
  auto const row_count = static_cast<std::size_t>(mds.extent(axes.row));
  auto const row_positions = selected_positions(row_count, edge_items);
  auto const col_positions = selected_positions(static_cast<std::size_t>(mds.extent(axes.column)), edge_items);
  elided = elided || positions_elide(row_positions) || positions_elide(col_positions);
  rows.reserve(row_positions.size());

  auto const ellipsis = render_piece(semantic_glyph::ellipsis, policy);
  for (auto const& row : row_positions)
  {
    if (!row.has_value())
    {
      std::vector<std::string> omitted_row;
      omitted_row.assign(col_positions.size(), ellipsis);
      rows.push_back(std::move(omitted_row));
      continue;
    }

    std::vector<std::string> rendered_row;
    rendered_row.reserve(col_positions.size());
    indices[axes.row] = static_cast<mdspan_index_type<MDS>>(*row);
    for (auto const& col : col_positions)
    {
      if (!col.has_value())
      {
        rendered_row.push_back(ellipsis);
        continue;
      }
      indices[axes.column] = static_cast<mdspan_index_type<MDS>>(*col);
      rendered_row.push_back(format_element(mds, indices, policy, formatter));
    }
    rows.push_back(std::move(rendered_row));
  }

  return rows;
}

template <typename MDS, std::size_t Rank>
void apply_slice_linear_index(MDS const& mds, std::array<mdspan_index_type<MDS>, Rank>& indices,
                              resolved_matrix_axes const& axes, std::size_t linear_index)
{
  for (auto it = axes.slice_axes.rbegin(); it != axes.slice_axes.rend(); ++it)
  {
    auto const axis = *it;
    auto const extent = static_cast<std::size_t>(mds.extent(axis));
    if (extent == 0) return;
    indices[axis] = static_cast<mdspan_index_type<MDS>>(linear_index % extent);
    linear_index /= extent;
  }
}

template <typename MDS> [[nodiscard]] std::size_t slice_count(MDS const& mds, resolved_matrix_axes const& axes)
{
  std::vector<std::size_t> extents;
  extents.reserve(axes.slice_axes.size());
  for (auto const axis : axes.slice_axes)
    extents.push_back(static_cast<std::size_t>(mds.extent(axis)));
  return checked_product(extents);
}

[[nodiscard]] inline std::string fit_metadata_piece(std::string piece, output_policy const& policy)
{
  if (!policy.wrap_width.has_value() || display_width(piece, policy) <= *policy.wrap_width) return piece;
  return truncate_to_width(piece, *policy.wrap_width, policy, render_piece(semantic_glyph::ellipsis, policy));
}

[[nodiscard]] inline std::string preview_header(std::vector<std::size_t> const& extents, std::size_t element_count,
                                                bool elided, output_policy const& policy)
{
  std::vector<std::string> pieces;
  pieces.push_back(render_piece(shape_string(extents), policy));
  pieces.push_back(render_piece(element_count_string(element_count), policy));
  if (elided) pieces.push_back(render_piece(preview_notice_string(), policy));

  if (!policy.wrap_width.has_value())
  {
    std::string header;
    for (auto const& piece : pieces)
    {
      if (!header.empty()) header.push_back(' ');
      header += piece;
    }
    return header;
  }

  std::string header;
  std::string line;
  for (auto piece : pieces)
  {
    piece = fit_metadata_piece(std::move(piece), policy);
    if (line.empty())
    {
      line = std::move(piece);
      continue;
    }

    auto candidate = line + " " + piece;
    if (display_width(candidate, policy) <= *policy.wrap_width)
    {
      line = std::move(candidate);
      continue;
    }

    if (!header.empty()) header.push_back('\n');
    header += line;
    line = std::move(piece);
  }

  if (!line.empty())
  {
    if (!header.empty()) header.push_back('\n');
    header += line;
  }
  return header;
}

template <typename MDS, typename ElementFormatter, std::size_t Rank>
void append_slices(std::string& out, MDS const& mds, std::array<mdspan_index_type<MDS>, Rank>& indices,
                   output_policy const& policy, ElementFormatter& formatter, mdspan_format_options const& options,
                   resolved_matrix_axes const& axes, std::size_t slice_pos)
{
  if (slice_pos == axes.slice_axes.size())
  {
    if (!out.empty()) out += "\n\n";
    if (options.label_slices)
    {
      out += render_piece(slice_label<MDS>(indices, axes), policy);
      out.push_back('\n');
    }
    out += format_matrix_rows(collect_matrix(mds, indices, policy, formatter, axes), policy);
    return;
  }

  auto const axis = axes.slice_axes[slice_pos];
  auto const extent = static_cast<std::size_t>(mds.extent(axis));
  for (std::size_t i = 0; i < extent; ++i)
  {
    indices[axis] = static_cast<mdspan_index_type<MDS>>(i);
    append_slices(out, mds, indices, policy, formatter, options, axes, slice_pos + 1);
  }
}

template <typename MDS> [[nodiscard]] std::vector<std::size_t> extents_vector(MDS const& mds)
{
  constexpr std::size_t rank = mdspan_rank_v<MDS>;
  std::vector<std::size_t> extents;
  extents.reserve(rank);
  for (std::size_t dim = 0; dim < rank; ++dim)
    extents.push_back(static_cast<std::size_t>(mds.extent(dim)));
  return extents;
}

[[nodiscard]] inline std::string with_shape_prefix(std::string body, mdspan_shape_mode shape,
                                                   std::vector<std::size_t> const& extents, std::size_t element_count,
                                                   bool elided, output_policy const& policy)
{
  if (shape == mdspan_shape_mode::none || extents.empty()) return body;

  return preview_header(extents, element_count, elided, policy) + "\n" + body;
}

} // namespace detail

template <mdspan_like MDS, typename ElementFormatter>
  requires(!std::same_as<std::remove_cvref_t<ElementFormatter>, mdspan_format_options>)
std::string format_mdspan(MDS const& mds, output_policy const& policy, ElementFormatter&& formatter,
                          mdspan_format_options const& options)
{
  constexpr std::size_t rank = detail::mdspan_rank_v<MDS>;
  using index_type = detail::mdspan_index_type<MDS>;

  auto element_formatter = std::forward<ElementFormatter>(formatter);
  std::string body;
  std::array<index_type, rank> indices{};

  if constexpr (rank == 0)
  {
    body = detail::format_element(mds, indices, policy, element_formatter);
  }
  else if constexpr (rank == 1)
  {
    std::vector<std::vector<std::string>> rows;
    rows.push_back(detail::collect_vector(mds, indices, policy, element_formatter, 0));
    body = detail::format_matrix_rows(rows, policy);
  }
  else
  {
    auto const axes = detail::resolve_matrix_axes(rank, options);
    if (axes.slice_axes.empty())
    {
      body = detail::format_matrix_rows(detail::collect_matrix(mds, indices, policy, element_formatter, axes), policy);
    }
    else
    {
      detail::append_slices(body, mds, indices, policy, element_formatter, options, axes, 0);
    }
    if (body.empty()) body = "[]";
  }

  if (options.shape == mdspan_shape_mode::none || rank == 0) return body;
  return detail::render_piece(detail::shape_string(detail::extents_vector(mds)), policy) + "\n" + body;
}

template <mdspan_like MDS>
std::string format_mdspan(MDS const& mds, output_policy const& policy, mdspan_format_options const& options)
{
  return format_mdspan(
      mds, policy, [&options](auto const& value) { return format_scalar(value, options.numeric); }, options);
}

template <mdspan_like MDS, typename ElementFormatter>
  requires(!std::same_as<std::remove_cvref_t<ElementFormatter>, mdspan_preview_options>)
mdspan_preview_result format_mdspan_preview(MDS const& mds, output_policy const& policy, ElementFormatter&& formatter,
                                            mdspan_preview_options const& options)
{
  constexpr std::size_t rank = detail::mdspan_rank_v<MDS>;
  using index_type = detail::mdspan_index_type<MDS>;

  auto element_formatter = std::forward<ElementFormatter>(formatter);
  auto const extents = detail::extents_vector(mds);
  auto const element_count = detail::checked_product(extents);

  auto full_options = options.format;
  if (element_count <= options.full_element_limit)
  {
    auto full_text = format_mdspan(mds, policy, element_formatter, full_options);
    if (detail::fits_wrap_width(full_text, policy))
    {
      return mdspan_preview_result{std::move(full_text), false, extents, element_count};
    }
  }

  std::string body;

  auto build_body = [&](std::size_t edge_items) {
    std::array<index_type, rank> indices{};
    bool build_elided = false;
    std::string built;

    if constexpr (rank == 0)
    {
      built = detail::format_element(mds, indices, policy, element_formatter);
    }
    else if constexpr (rank == 1)
    {
      std::vector<std::vector<std::string>> rows;
      rows.push_back(
          detail::collect_preview_vector(mds, indices, policy, element_formatter, 0, edge_items, build_elided));
      built = detail::format_matrix_rows(rows, policy);
    }
    else
    {
      auto const axes = detail::resolve_matrix_axes(rank, full_options);
      if (axes.slice_axes.empty())
      {
        built = detail::format_matrix_rows(
            detail::collect_preview_matrix(mds, indices, policy, element_formatter, axes, edge_items, build_elided),
            policy);
      }
      else
      {
        auto const total_slices = detail::slice_count(mds, axes);
        auto const slice_positions =
            detail::selected_positions(total_slices, std::max<std::size_t>(options.max_slices / 2, 1));
        build_elided = build_elided || detail::positions_elide(slice_positions);
        auto const ellipsis = detail::render_piece(semantic_glyph::ellipsis, policy);
        for (auto const& slice : slice_positions)
        {
          if (!built.empty()) built += "\n\n";
          if (!slice.has_value())
          {
            built += ellipsis;
            continue;
          }
          detail::apply_slice_linear_index(mds, indices, axes, *slice);
          if (full_options.label_slices)
          {
            built += detail::render_piece(detail::slice_label<MDS>(indices, axes), policy);
            built.push_back('\n');
          }
          built += detail::format_matrix_rows(
              detail::collect_preview_matrix(mds, indices, policy, element_formatter, axes, edge_items, build_elided),
              policy);
        }
        if (built.empty()) built = "[]";
      }
    }

    return std::pair<std::string, bool>{std::move(built), build_elided};
  };

  auto edge_items = std::max<std::size_t>(options.edge_items, 1);
  bool body_elided = false;
  for (;;)
  {
    auto [candidate_body, candidate_elided] = build_body(edge_items);
    auto candidate =
        detail::with_shape_prefix(candidate_body, full_options.shape, extents, element_count, candidate_elided, policy);
    if (detail::fits_wrap_width(candidate, policy) || edge_items == 1)
    {
      body = std::move(candidate_body);
      body_elided = candidate_elided;
      break;
    }
    --edge_items;
  }

  auto text = detail::with_shape_prefix(body, full_options.shape, extents, element_count, body_elided, policy);
  if (!detail::fits_wrap_width(text, policy) && policy.wrap_width.has_value())
  {
    text = detail::preview_header(extents, element_count, true, policy) + "\n" +
           truncate_to_width("preview omitted: terminal width too narrow for one-cell tensor preview",
                             *policy.wrap_width, policy, detail::render_piece(semantic_glyph::ellipsis, policy));
    body_elided = true;
  }

  return mdspan_preview_result{std::move(text), body_elided, extents, element_count};
}

template <mdspan_like MDS>
mdspan_preview_result format_mdspan_preview(MDS const& mds, output_policy const& policy,
                                            mdspan_preview_options const& options)
{
  return format_mdspan_preview(
      mds, policy, [&options](auto const& value) { return format_scalar(value, options.format.numeric); }, options);
}

} // namespace uni20::presentation
