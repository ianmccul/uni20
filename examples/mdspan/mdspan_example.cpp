
#include <fmt/core.h>
#include <uni20/common/presentation_mdspan.hpp>
#include <uni20/mdspan/mdspan.hpp>

#include <array>
#include <complex>
#include <cstddef>

int main()
{
  auto policy = uni20::presentation::terminal_policy(stdout);

  std::array<int, 9> matrix_data{0, 1, 2, 3, 4, 5, 6, 7, 8};
  stdex::mdspan<int, stdex::extents<std::size_t, 3, 3>> matrix(matrix_data.data());

  fmt::print("Integer matrix (3x3):\n{}\n\n", uni20::presentation::format_mdspan(matrix, policy));

  std::array<int, 12> rank3_data{};
  for (std::size_t i = 0; i < rank3_data.size(); ++i)
    rank3_data[i] = static_cast<int>(i);

  stdex::mdspan<int, stdex::extents<std::size_t, 2, 3, 2>> rank3(rank3_data.data());
  fmt::print("Rank-3 tensor (2x3x2):\n{}\n\n", uni20::presentation::format_mdspan(rank3, policy));

  uni20::presentation::mdspan_format_options axis_options;
  axis_options.matrix_axes = uni20::presentation::mdspan_matrix_axes{0, 2};
  fmt::print("Rank-3 tensor with axes 0 and 2 as matrix axes:\n{}\n\n",
             uni20::presentation::format_mdspan(rank3, policy, axis_options));

  uni20::presentation::mdspan_format_options real_options;
  real_options.numeric.notation = uni20::presentation::real_notation::fixed;
  real_options.numeric.float64_precision = 3;

  std::array<double, 6> real_data{1.0, 2.0 / 3.0, -12.5, 1000.0, -0.0, 3.141592653589793};
  stdex::mdspan<double, stdex::extents<std::size_t, 2, 3>> real_matrix(real_data.data());
  fmt::print("Real matrix, fixed precision (2x3):\n{}\n\n",
             uni20::presentation::format_mdspan(real_matrix, policy, real_options));

  uni20::presentation::mdspan_format_options complex_options;
  complex_options.numeric.notation = uni20::presentation::real_notation::fixed;
  complex_options.numeric.float64_precision = 2;

  using complex_t = uni20::complex<double>;
  std::array<complex_t, 4> complex_data{complex_t{1.0, -2.5}, complex_t{0.0, 3.0}, complex_t{-4.25, 0.125},
                                        complex_t{2.0, -0.0}};
  stdex::mdspan<complex_t, stdex::extents<std::size_t, 2, 2>> complex_matrix(complex_data.data());
  fmt::print("Complex matrix, fixed precision (2x2):\n{}\n",
             uni20::presentation::format_mdspan(complex_matrix, policy, complex_options));

  return 0;
}
