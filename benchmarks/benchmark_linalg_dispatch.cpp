#include <benchmark/benchmark.h>
#include <uni20/linalg/linalg.hpp>
#include <uni20/tensor/tensor.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace
{
using tensor_type = uni20::Tensor<double, 2>;

struct GemmOperands
{
    explicit GemmOperands(std::size_t size) : output(size, size), lhs(size, size), rhs(size, size)
    {
      auto lhs_span = lhs.mdspan();
      auto rhs_span = rhs.mdspan();
      for (std::size_t row = 0; row < size; ++row)
      {
        for (std::size_t col = 0; col < size; ++col)
        {
          lhs_span[row, col] = 1.0 + static_cast<double>(row) + 2.0 * static_cast<double>(col);
          rhs_span[row, col] = 1.0 + 2.0 * static_cast<double>(row) - static_cast<double>(col);
        }
      }
    }

    tensor_type output;
    tensor_type lhs;
    tensor_type rhs;
};

void set_gemm_work(benchmark::State& state)
{
  auto const size = static_cast<std::int64_t>(state.range(0));
  state.SetItemsProcessed(state.iterations());
  state.SetLabel("one " + std::to_string(size) + "x" + std::to_string(size) + " GEMM");
  state.counters["flops"] =
      benchmark::Counter(static_cast<double>(state.iterations()) * 2.0 * static_cast<double>(size * size * size),
                         benchmark::Counter::kIsRate);
}

void CpuGemmTryKernel(benchmark::State& state)
{
  GemmOperands operands(static_cast<std::size_t>(state.range(0)));
  auto output = operands.output.mdspan();
  auto lhs = operands.lhs.mdspan();
  auto rhs = operands.rhs.mdspan();
  auto* lhs_data = lhs.data_handle();
  auto* rhs_data = rhs.data_handle();
  uni20::linalg::dispatch_diagnostics::reset_sink();

  for ([[maybe_unused]] auto _ : state)
  {
    benchmark::DoNotOptimize(lhs_data);
    benchmark::DoNotOptimize(rhs_data);
    auto attempt = uni20::linalg::try_kernel(uni20::linalg::CpuReferenceBackend{}, uni20::linalg::gemm_op{}, output,
                                             1.0, lhs, rhs, 0.0);
    benchmark::DoNotOptimize(attempt);
    benchmark::ClobberMemory();
  }

  set_gemm_work(state);
}

void CpuGemmDispatch(benchmark::State& state)
{
  GemmOperands operands(static_cast<std::size_t>(state.range(0)));
  auto output = operands.output.mdspan();
  auto lhs = operands.lhs.mdspan();
  auto rhs = operands.rhs.mdspan();
  auto* lhs_data = lhs.data_handle();
  auto* rhs_data = rhs.data_handle();
  uni20::linalg::dispatch_diagnostics::reset_sink();

  for ([[maybe_unused]] auto _ : state)
  {
    benchmark::DoNotOptimize(lhs_data);
    benchmark::DoNotOptimize(rhs_data);
    uni20::linalg::dispatch_kernel(uni20::linalg::CpuReferenceBackend{}, uni20::linalg::gemm_op{}, output, 1.0, lhs,
                                   rhs, 0.0);
    benchmark::ClobberMemory();
  }

  set_gemm_work(state);
}

void TensorAssignProduct(benchmark::State& state)
{
  GemmOperands operands(static_cast<std::size_t>(state.range(0)));
  auto* lhs_data = operands.lhs.storage().data();
  auto* rhs_data = operands.rhs.storage().data();
  uni20::linalg::dispatch_diagnostics::reset_sink();

  for ([[maybe_unused]] auto _ : state)
  {
    benchmark::DoNotOptimize(lhs_data);
    benchmark::DoNotOptimize(rhs_data);
    uni20::linalg::assign_product(operands.output, operands.lhs, operands.rhs);
    benchmark::ClobberMemory();
  }

  set_gemm_work(state);
}

void register_tiny_gemm_sizes(benchmark::internal::Benchmark* benchmark)
{
  benchmark->Arg(1)->Arg(2)->Arg(4)->Arg(16)->ArgName("size");
}
} // namespace

BENCHMARK(CpuGemmTryKernel)->Apply(register_tiny_gemm_sizes);
BENCHMARK(CpuGemmDispatch)->Apply(register_tiny_gemm_sizes);
BENCHMARK(TensorAssignProduct)->Apply(register_tiny_gemm_sizes);
