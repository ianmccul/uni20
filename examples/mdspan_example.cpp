#include <fmt/core.h>
#include <fmt/format.h>
#include <experimental/mdspan>
#include <vector>

namespace stdex = std::experimental;

int main() {
    // Create a vector to hold 3x3 matrix data.
    std::vector<int> data(9);
    for (int i = 0; i < 9; ++i) {
        data[i] = i;  // Fill with values 0, 1, 2, ..., 8
    }

    // Create a 3x3 mdspan that wraps the vector.
    // Here we use a dynamic extents type with fixed dimensions.
    stdex::mdspan<int, stdex::extents<size_t, 3, 3>> matrix(data.data());

    // Print the matrix using fmt.
    fmt::print("Matrix (3x3):\n");
    for (size_t i = 0; i < matrix.extents().extent(0); ++i) {
        for (size_t j = 0; j < matrix.extents().extent(1); ++j) {
            fmt::print("{:3d} ", matrix[i, j]);
        }
        fmt::print("\n");
    }

    // Also demonstrate fmt::format
    auto formatted_str = fmt::format("The first element is: {}\n", matrix[0, 0]);
    fmt::print("{}", formatted_str);

    return 0;
}
