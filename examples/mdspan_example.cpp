
#include <uni20/common/mdspan.hpp>
#include <uni20/common/presentation_mdspan.hpp>
#include <fmt/core.h>
#include <fmt/format.h>
#include <vector>

int main()
{
  // Create a vector to hold 3x3 matrix data.
  std::vector<int> data(9);
  for (int i = 0; i < 9; ++i)
  {
    data[i] = i; // Fill with values 0, 1, 2, ..., 8
  }

  // Create a 3x3 mdspan that wraps the vector.
  // Here we use a dynamic extents type with fixed dimensions.
  stdex::mdspan<int, stdex::extents<size_t, 3, 3>> matrix(data.data());

  // Print the matrix using Uni20 presentation formatting.
  auto policy = uni20::presentation::terminal_policy(stdout);
  fmt::print("Matrix (3x3):\n{}\n", uni20::presentation::format_mdspan(matrix, policy));

  // Also demonstrate fmt::format
  auto formatted_str = fmt::format("The first element is: {}\n", matrix[0, 0]);
  fmt::print("{}", formatted_str);

  return 0;
}
