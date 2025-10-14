#pragma once

/**
 * \defgroup kernel_cpu CPU kernel backends
 * \ingroup kernel
 * \brief Tag types and helpers specific to CPU tensor kernels.
 */

namespace uni20
{

/// \brief Tag that selects the CPU tensor-kernel implementation.
/// \ingroup kernel_cpu
struct cpu_tag
{};

} // namespace uni20
