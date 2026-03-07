
#include <uni20/backend/blas/blas_vendor.hpp>
#include <fmt/core.h>

int main()
{
  fmt::print("BLAS vendor string: {}\n", uni20::BLAS_Vendor());
  fmt::print("BLAS version string: {}\n", uni20::BLAS_Version());
  return 0;
}
