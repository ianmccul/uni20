#pragma once

#include <string>

namespace uni20
{

/// \brief Reports the human-readable vendor string for the active BLAS backend.
/// \return Name of the runtime-selected BLAS implementation, such as "Intel MKL".
/// \ingroup backend_blas
std::string BLAS_Vendor();

/// \brief Retrieves the version identifier associated with the active BLAS backend.
/// \return Version string supplied by the vendor library, or an empty string when unavailable.
/// \ingroup backend_blas
std::string BLAS_Version();

} // namespace uni20
