
#include "blas_vendor.hpp"
#include "config.hpp"

#include <cstring>
#include <string>

namespace uni20
{

#ifdef UNI20_BLAS_VENDOR_MKL
// For MKL: Use the C interface provided by MKL.
extern "C" void MKL_Get_Version_String(char* buf, int len);

std::string BLAS_Vendor() { return "MKL"; }

std::string BLAS_Version()
{
  char buf[198];
  MKL_Get_Version_String(buf, 198);
  return std::string(buf);
}

#elif defined(UNI20_BLAS_VENDOR_OPENBLAS)
// For OpenBLAS: Use the function openblas_get_config().
extern "C" char* openblas_get_config();

std::string BLAS_Vendor() { return "OpenBLAS"; }

std::string BLAS_Version() { return openblas_get_config(); }

#else
// For generic BLAS (or if no version info extensions available)
std::string BLAS_Vendor() { return UNI20_BLAS_VENDOR; }

std::string BLAS_Version() { return "(no version information available)"; }
#endif

} // namespace uni20
