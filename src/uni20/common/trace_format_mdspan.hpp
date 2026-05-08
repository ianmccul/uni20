#pragma once

#include "trace.hpp"

namespace trace
{

/// \brief Compatibility alias for mdspan-like values handled by trace formatting.
/// \tparam MDS Candidate mdspan-like type.
template <typename MDS>
concept MdspanLike = uni20::presentation::mdspan_like<MDS>;

/// \brief Format an mdspan-like value using trace scalar formatting and presentation tensor art.
/// \tparam MDS Mdspan-like object type.
/// \param mds Mdspan-like object to render.
/// \param opts Trace formatting options.
/// \return Display-cell-aligned matrix or higher-order tensor art.
template <MdspanLike MDS> [[nodiscard]] std::string format_mdspan_to_string(MDS const& mds, FormattingOptions const& opts)
{
  return formatValue(mds, opts);
}

} // namespace trace
