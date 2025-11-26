#include "async/async.hpp"
#include "async/tbb_scheduler.hpp"
#include "tensor/basic_tensor.hpp"
#include <benchmark/benchmark.h>

#include <vector>

using namespace uni20;
using namespace uni20::async;

namespace
{
using extents_type = stdex::dextents<index_type, 2>;
using tensor_type = BasicTensor<float, extents_type>;

AsyncTask row_scale_add(tensor_type const* lhs, tensor_type const* rhs, tensor_type* out, std::size_t row, float scale)
{
  auto lhs_view = lhs->mdspan();
  auto rhs_view = rhs->mdspan();
  auto out_view = out->mutable_mdspan();
  auto const cols = static_cast<std::size_t>(out_view.extents().extent(1));

  for (std::size_t col = 0; col < cols; ++col)
  {
    out_view[row, col] = lhs_view[row, col] * scale + rhs_view[row, col];
  }

  co_return;
}

AsyncTask row_sum_task(tensor_type const* tensor, WriteBuffer<float> out, std::size_t row)
{
  auto view = tensor->mdspan();
  auto const cols = static_cast<std::size_t>(view.extents().extent(1));
  float accum = 0.0F;

  for (std::size_t col = 0; col < cols; ++col)
    accum += view[row, col];

  co_await out = accum;
  co_return;
}

void initialize_tensor(tensor_type& tensor)
{
  auto view = tensor.mutable_mdspan();
  auto const rows = static_cast<std::size_t>(view.extents().extent(0));
  auto const cols = static_cast<std::size_t>(view.extents().extent(1));

  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c)
      view[r, c] = static_cast<float>((r + 1) * 0.5 + (c + 1) * 0.25);
}
} // namespace

static void TensorScaleAddTbb(benchmark::State& state)
{
  auto const threads = static_cast<int>(state.range(0));
  auto const rows = static_cast<std::size_t>(state.range(1));
  auto const cols = static_cast<std::size_t>(state.range(2));

  tensor_type lhs{extents_type{rows, cols}};
  tensor_type rhs{extents_type{rows, cols}};
  tensor_type out{extents_type{rows, cols}};
  initialize_tensor(lhs);
  initialize_tensor(rhs);

  TbbScheduler sched{threads};
  ScopedScheduler guard(&sched);

  for (auto _ : state)
  {
    for (std::size_t row = 0; row < rows; ++row)
      sched.schedule(row_scale_add(&lhs, &rhs, &out, row, 1.5F));

    sched.run_all();
    benchmark::DoNotOptimize(out);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(rows * cols));
}

BENCHMARK(TensorScaleAddTbb)
    ->Args({1, 256, 524288})
    ->Args({2, 256, 524288})
    ->Args({4, 256, 524288})
    ->Args({8, 256, 524288})
    ->ArgNames({"threads", "rows", "cols"});

static void TensorRowReductionTbb(benchmark::State& state)
{
  auto const threads = static_cast<int>(state.range(0));
  auto const rows = static_cast<std::size_t>(state.range(1));
  auto const cols = static_cast<std::size_t>(state.range(2));

  tensor_type tensor{extents_type{rows, cols}};
  initialize_tensor(tensor);

  TbbScheduler sched{threads};
  ScopedScheduler guard(&sched);

  for (auto _ : state)
  {
    std::vector<Async<float>> partials(rows);
    for (std::size_t row = 0; row < rows; ++row)
      sched.schedule(row_sum_task(&tensor, partials[row].write(), row));

    sched.run_all();

    float total = 0.0F;
    for (auto& partial : partials)
      total += partial.get_wait();

    benchmark::DoNotOptimize(total);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(rows * cols));
}

BENCHMARK(TensorRowReductionTbb)
    ->Args({1, 512, 128000})
    ->Args({2, 512, 128000})
    ->Args({4, 512, 128000})
    ->Args({8, 512, 128000})
    ->ArgNames({"threads", "rows", "cols"});
