#pragma once

#include <uni20/level1/transform.hpp>

#include <utility>

/**
 * \defgroup level1_ops Level-1 tensor algorithms
 * \brief Element-wise tensor kernels that operate on strided mdspan views.
 */

namespace uni20
{

/// \brief Copy elements from a source mdspan into a destination mdspan.
/// \tparam MDSDst Destination mdspan type that models StridedMdspan.
/// \tparam MDSSrc Source mdspan type that models StridedMdspan.
/// \param dst Destination view receiving the copied elements.
/// \param src Source view providing the element values.
/// \ingroup level1_ops
template <MutableStridedMdspan MDSDst, StridedMdspan MDSSrc>
  requires requires(typename MDSDst::reference destination, typename MDSSrc::reference source) { destination = source; }
void assign(MDSDst dst, MDSSrc const& src)
{
  transform(dst, src,
            [](auto&& source) -> decltype(auto) { return std::forward<decltype(source)>(source); });
}

} // namespace uni20
