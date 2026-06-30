#pragma once

/**
 * \file common.hpp
 * \ingroup backend_lapack_mplapack
 * \brief Shared helpers for MPLAPACK LAPACK wrappers.
 */

#include <uni20/config.hpp>

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK
// clang-format off
#include <mplapack_config.h>
#include <mplapack_binary128.h>
// clang-format on
#endif

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace uni20::lapack::mplapack::detail
{

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK

inline std::size_t lapack_int_work_size(blas_int count)
{
  return static_cast<std::size_t>(std::max<blas_int>(1, count));
}

inline std::vector<mplapackint> make_mplapack_int_work(blas_int count)
{
  return std::vector<mplapackint>(lapack_int_work_size(count));
}

inline std::vector<mplapackint> to_mplapack_ints(blas_int const* values, blas_int count)
{
  std::vector<mplapackint> result = make_mplapack_int_work(count);
  for (blas_int i = 0; i < count; ++i)
  {
    result[static_cast<std::size_t>(i)] = static_cast<mplapackint>(values[i]);
  }
  return result;
}

inline void copy_from_mplapack_ints(std::vector<mplapackint> const& source, blas_int* target, blas_int count)
{
  for (blas_int i = 0; i < count; ++i)
  {
    target[i] = static_cast<blas_int>(source[static_cast<std::size_t>(i)]);
  }
}

inline blas_int checked_blas_int(std::size_t value)
{
  auto const converted = static_cast<blas_int>(value);
  if (static_cast<std::size_t>(converted) != value)
  {
    throw std::overflow_error("LAPACK workspace dimension does not fit BLAS integer type");
  }
  return converted;
}

inline blas_int gelsd_iwork_size(blas_int m, blas_int n)
{
  blas_int const minmn = std::min(m, n);
  if (minmn <= 0)
  {
    return 1;
  }

  // LAPACK's bound uses SMLSIZ from ILAENV. Using divisor 2 overestimates the
  // subdivision depth, avoiding a runtime dependency on ILAENV in this wrapper.
  std::size_t levels = 0;
  for (std::size_t size = static_cast<std::size_t>(minmn) / 2; size > 0; size /= 2)
  {
    ++levels;
  }
  std::size_t const minmn_size = static_cast<std::size_t>(minmn);
  std::size_t const required = 3 * minmn_size * levels + 11 * minmn_size;
  return checked_blas_int(std::max<std::size_t>(1, required));
}

#endif

} // namespace uni20::lapack::mplapack::detail
