#pragma once

#include "presentation.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
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
  return render_text(std::invoke(formatter, mdspan_at(mds, indices)), policy);
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
                                          std::vector<std::size_t> const& column_widths,
                                          output_policy const& policy)
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

template <typename MDS, typename ElementFormatter, std::size_t Rank>
[[nodiscard]] std::vector<std::string> collect_vector(MDS const& mds,
                                                      std::array<mdspan_index_type<MDS>, Rank>& indices,
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
[[nodiscard]] std::vector<std::vector<std::string>> collect_matrix(MDS const& mds,
                                                                   std::array<mdspan_index_type<MDS>, Rank>& indices,
                                                                   output_policy const& policy,
                                                                   ElementFormatter& formatter,
                                                                   resolved_matrix_axes const& axes)
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

template <typename MDS>
[[nodiscard]] std::vector<std::size_t> extents_vector(MDS const& mds)
{
  constexpr std::size_t rank = mdspan_rank_v<MDS>;
  std::vector<std::size_t> extents;
  extents.reserve(rank);
  for (std::size_t dim = 0; dim < rank; ++dim)
    extents.push_back(static_cast<std::size_t>(mds.extent(dim)));
  return extents;
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
  return format_mdspan(mds, policy,
                       [&options](auto const& value) { return format_scalar(value, options.numeric); }, options);
}

} // namespace uni20::presentation
