
#include "backend/blas/blas_vendor.hpp"
#include <fmt/core.h>

int main()
{
  fmt::println("BLAS vendor string: {}", uni20::BLAS_Vendor());
  fmt::println("BLAS version string: {}", uni20::BLAS_Version());
  return 0;
}
