#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/tensor.hpp>

#include <iostream>

namespace
{
using matrix_type = uni20::DenseMatrix<double>;

matrix_type make_lhs()
{
  matrix_type result(2, 3);
  result[0, 0] = 1;
  result[0, 1] = 2;
  result[0, 2] = 3;
  result[1, 0] = 4;
  result[1, 1] = 5;
  result[1, 2] = 6;
  return result;
}

matrix_type make_rhs()
{
  matrix_type result(3, 2);
  result[0, 0] = 7;
  result[0, 1] = 8;
  result[1, 0] = 9;
  result[1, 1] = 10;
  result[2, 0] = 11;
  result[2, 1] = 12;
  return result;
}
} // namespace

int main()
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<matrix_type> lhs = make_lhs();
  uni20::async::Async<matrix_type> rhs = make_rhs();
  uni20::async::Async<matrix_type> output;
  uni20::async::Async<double> update_scale = 0.5;

  uni20::linalg::assign_product(output, lhs, rhs);
  uni20::linalg::add_product(uni20::linalg::CpuReferenceBackend{}, output, lhs, rhs, update_scale);

  auto const& result = output.get_wait(scheduler);
  std::cout << "output = 1.5 * lhs * rhs\n";
  for (uni20::index_type row = 0; row < result.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < result.cols(); ++col)
    {
      if (col != 0) std::cout << ' ';
      std::cout << result[row, col];
    }
    std::cout << '\n';
  }

  bool const correct = result.rows() == 2 && result.cols() == 2 && result[0, 0] == 87.0 && result[0, 1] == 96.0 &&
                       result[1, 0] == 208.5 && result[1, 1] == 231.0;
  return correct ? 0 : 1;
}
