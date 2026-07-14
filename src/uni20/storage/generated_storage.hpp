#pragma once

/**
 * \file generated_storage.hpp
 * \ingroup tensor
 * \brief Storage policy marker for compact generated tensor values.
 */

#include <uni20/linalg/backend_selector.hpp>

namespace uni20
{

/// \brief Backend-neutral storage policy for tensors generated from compact state.
/// \details Generated tensors have no addressable element buffer. Their
///          resolved accessors calculate values from logical indices and may
///          be accepted by any backend that understands those accessors.
struct GeneratedStorage
{
    using backend_selector_type = linalg::backend_list<linalg::CpuReferenceBackend>;

    /// \brief Return the host reference fallback used when no concrete operand selects a backend.
    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return backend_selector_type{linalg::CpuReferenceBackend{}};
    }
};

} // namespace uni20

namespace uni20::linalg
{

template <> inline constexpr bool enable_backend_neutral_storage<uni20::GeneratedStorage> = true;

} // namespace uni20::linalg
