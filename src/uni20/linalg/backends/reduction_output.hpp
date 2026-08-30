#pragma once

/**
 * \file reduction_output.hpp
 * \ingroup linalg
 * \brief Shared host-output handling for synchronous scalar reductions.
 */

#include <uni20/linalg/kernel_attempt.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace uni20::linalg::reduction_detail
{

/// \brief Report whether an output can receive a reduction value of fixed rank.
template <class Output, class Scalar, std::size_t Rank> consteval bool output_is_supported()
{
  using output_type = std::remove_cvref_t<Output>;
  if constexpr (uni20::MutableRankedMdspanLike<output_type, Rank>)
  {
    return std::same_as<typename output_type::value_type, Scalar>;
  }
  else if constexpr (Rank == 0)
  {
    return std::same_as<output_type, Scalar> && requires(Output& output, Scalar value) { output = value; };
  }
  else
  {
    return false;
  }
}

/// \brief Reduction output that is either a host scalar or host-writable mdspec.
template <class Output>
concept HostOutput = (!uni20::MdspecLike<Output>) || uni20::HostWritableMdspec<Output>;

template <class Output, bool = uni20::MdspecLike<Output>> struct HostOutputType
{
    using type = std::remove_cvref_t<Output>;
};

template <class Output> struct HostOutputType<Output, true>
{
    using type = uni20::host_write_mdspan_t<Output>;
};

/// \brief Resolved output type used by a host reduction backend probe.
template <class Output> using host_output_t = typename HostOutputType<Output>::type;

/// \brief Store a scalar in either a resolved rank-zero mdspan or C++ scalar.
template <class Output, class Scalar> void write_host_output(Output& output, Scalar value)
{
  if constexpr (uni20::MutableRankedMdspanLike<std::remove_cvref_t<Output>, 0>)
    output[] = value;
  else
    output = value;
}

/// \brief Invoke a synchronous reduction with a resolved host output.
template <class Output, class Function> KernelAttempt with_host_output(Output& output, Function&& function)
{
  if constexpr (uni20::MdspecLike<Output>)
  {
    auto output_access = acquire_host_write_access_sync(output);
    auto output_span = output_access.mdspan();
    return std::invoke(std::forward<Function>(function), output_span);
  }
  else
  {
    return std::invoke(std::forward<Function>(function), output);
  }
}

} // namespace uni20::linalg::reduction_detail
