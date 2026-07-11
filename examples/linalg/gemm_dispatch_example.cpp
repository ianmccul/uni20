#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/tensor/tensor.hpp>

#include <fmt/core.h>
#include <initializer_list>
#include <string_view>

namespace
{
template <class Matrix> void fill_matrix(Matrix& matrix, std::initializer_list<double> values)
{
  auto value = values.begin();
  for (uni20::index_type row = 0; row < matrix.extent(0); ++row)
  {
    for (uni20::index_type col = 0; col < matrix.extent(1); ++col)
    {
      matrix[row, col] = *value;
      ++value;
    }
  }
}

constexpr std::string_view acceptance_name(uni20::linalg::KernelTypeAcceptance acceptance)
{
  switch (acceptance)
  {
    case uni20::linalg::KernelTypeAcceptance::no:
      return "no";
    case uni20::linalg::KernelTypeAcceptance::maybe:
      return "maybe";
    case uni20::linalg::KernelTypeAcceptance::yes:
      return "yes";
  }
  return "unknown";
}
} // namespace

int main()
{
  using tensor_type = uni20::Tensor<double, 2>;
  using extents_type = typename tensor_type::extents_type;

  tensor_type lhs(extents_type{2, 3});
  tensor_type rhs(extents_type{3, 2});
  tensor_type output(extents_type{2, 2});
  fill_matrix(lhs, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  fill_matrix(rhs, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  auto selector = output.backend_selector();
  auto output_span = output.mdspan();
  auto lhs_span = lhs.mdspan();
  auto rhs_span = rhs.mdspan();
  auto const acceptance = uni20::linalg::probe_dispatch_kernel(selector, uni20::linalg::gemm_op{}, output_span, 1.0,
                                                               lhs_span, rhs_span, 0.0);
  fmt::print("Tensor GEMM type acceptance: {}\n", acceptance_name(acceptance));

  // The owning tensors provide the selector; GEMM lowers their mdspans into the backend walk.
  uni20::linalg::gemm(output, 1.0, lhs, rhs, 0.0);

  fmt::print("result = [[{}, {}], [{}, {}]]\n", output[0, 0], output[0, 1], output[1, 0], output[1, 1]);
  bool const correct = output[0, 0] == 58.0 && output[0, 1] == 64.0 && output[1, 0] == 139.0 && output[1, 1] == 154.0;
  return correct ? 0 : 1;
}
