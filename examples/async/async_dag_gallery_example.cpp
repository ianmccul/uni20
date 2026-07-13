#include <uni20/async/async.hpp>
#include <uni20/async/async_ops.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/task_registry.hpp>
#include <uni20/config.hpp>

#include <cmath>
#include <filesystem>
#include <fmt/core.h>
#include <string>
#include <utility>
#include <vector>

using namespace uni20::async;

namespace
{

std::filesystem::path dot_path(std::filesystem::path const& output_dir, char const* name)
{
  return output_dir / fmt::format("async-dag-gallery-{}.dot", name);
}

void write_dot(std::filesystem::path const& output_dir, char const* name)
{
  auto const path = dot_path(output_dir, name);
  auto const ok = uni20::TaskRegistry::dump_graphviz_file_best_effort(path.string());
  fmt::print("{} {}\n", ok ? "wrote" : "failed to write", path.string());
}

template <typename T> Async<T> branch_static(int mode, Async<T> const& a, Async<T> const& b, Async<T> const& c)
{
  return (mode == 1) ? a + b * c : (a + b) * c;
}

template <typename T>
Async<T> branch_dynamic(Async<int> const& mode, Async<T> const& a, Async<T> const& b, Async<T> const& c,
                        std::string const& task_label)
{
  Async<T> out;
  auto task = [](ReadBuffer<int> mode_in, ReadBuffer<T> a_in, ReadBuffer<T> b_in, ReadBuffer<T> c_in,
                 WriteBuffer<T> out_writer) static -> AsyncTask {
    auto [mode_value, av, bv, cv] = co_await all(mode_in, a_in, b_in, c_in);
    auto result = (mode_value == 1) ? av + bv * cv : (av + bv) * cv;
    mode_in.release();
    a_in.release();
    b_in.release();
    c_in.release();
    co_await out_writer = result;
    co_return;
  }(mode.read(), a.read(), b.read(), c.read(), out.write());
  task.debug_name(task_label);
  schedule(std::move(task));
  return out;
}

void branch_gallery(std::filesystem::path const& output_dir)
{
  fmt::print("\n=== Branching expression DAG ===\n");
  DebugScheduler scheduler;
  ScopedScheduler scoped(&scheduler);
  scheduler.pause();

  Async<int> a = 2;
  Async<int> b = a + 1;
  Async<int> c = b + 1;
  Async<int> mode1 = 1;
  Async<int> mode2 = mode1 + 1;
  a.debug_name("a");
  b.debug_name("b = a + 1");
  c.debug_name("c = b + 1");
  mode1.debug_name("mode1");
  mode2.debug_name("mode2 = mode1 + 1");

  auto r1 = branch_static(1, a, b, c);
  auto r2 = branch_static(2, a, b, c);
  auto r3 = branch_dynamic(mode1, a, b, c, "dynamic branch mode1");
  auto r4 = branch_dynamic(mode2, a, b, c, "dynamic branch mode2");
  r1.debug_name("r1 = a + b * c");
  r2.debug_name("r2 = (a + b) * c");
  r3.debug_name("r3 dynamic");
  r4.debug_name("r4 dynamic");

  write_dot(output_dir, "branch-01-constructed");
  scheduler.resume();
  fmt::print("static branch 1  = {}\n", r1.get_wait(scheduler));
  fmt::print("static branch 2  = {}\n", r2.get_wait(scheduler));
  fmt::print("dynamic branch 1 = {}\n", r3.get_wait(scheduler));
  fmt::print("dynamic branch 2 = {}\n", r4.get_wait(scheduler));
  write_dot(output_dir, "branch-02-complete");
}

double f(double x, double y) { return x + 2.0 * y; }
double g(double y) { return y * y - 1.0; }
double h(double u, double z) { return 0.5 * u + 0.25 * z; }

Async<double> f(Async<double> const& x, Async<double> const& y) { return x + 2.0 * y; }
Async<double> g(Async<double> const& y) { return y * y - 1.0; }
Async<double> h(Async<double> const& u, Async<double> const& z) { return 0.5 * u + 0.25 * z; }

Async<double> compact_expression_form(Async<double> const& x, Async<double> const& y)
{
  Async<double> z = f(x, y);
  Async<double> u = g(y);
  z += h(u, z);
  return z;
}

Async<double> explicit_kernel_form(Async<double> const& x, Async<double> const& y)
{
  Async<double> z;
  Async<double> u;

  auto z_task = [](ReadBuffer<double> x_in, ReadBuffer<double> y_in, WriteBuffer<double> z_out) static -> AsyncTask {
    co_await z_out = f(co_await x_in, co_await y_in);
    co_return;
  }(x.read(), y.read(), z.write());
  z_task.debug_name("explicit z = f(x, y)");
  schedule(std::move(z_task));

  auto u_task = [](ReadBuffer<double> y_in, WriteBuffer<double> u_out) static -> AsyncTask {
    co_await u_out = g(co_await y_in);
    co_return;
  }(y.read(), u.write());
  u_task.debug_name("explicit u = g(y)");
  schedule(std::move(u_task));

  auto update_task = [](ReadBuffer<double> u_in, WriteBuffer<double> z_io) static -> AsyncTask {
    // The writer provides both inspection and exclusive mutation of z.
    auto [uval, zval] = co_await all(u_in, z_io);
    zval += h(uval, zval);
    co_return;
  }(u.read(), z.write());
  update_task.debug_name("explicit z += h(u, z)");
  schedule(std::move(update_task));

  return z;
}

void kernel_shape_gallery(std::filesystem::path const& output_dir)
{
  fmt::print("\n=== Compact expression versus explicit kernels ===\n");
  DebugScheduler scheduler;
  ScopedScheduler scoped(&scheduler);
  scheduler.pause();

  Async<double> x = 2.0;
  Async<double> y = 3.0;
  x.debug_name("x");
  y.debug_name("y");
  auto compact = compact_expression_form(x, y);
  auto explicit_kernel = explicit_kernel_form(x, y);
  compact.debug_name("compact z");
  explicit_kernel.debug_name("explicit z");

  write_dot(output_dir, "kernel-shapes-01-constructed");
  scheduler.resume();
  auto const compact_result = compact.get_wait(scheduler);
  auto const explicit_result = explicit_kernel.get_wait(scheduler);
  write_dot(output_dir, "kernel-shapes-02-complete");

  fmt::print("compact result = {:.6f}\n", compact_result);
  fmt::print("explicit result = {:.6f}\n", explicit_result);
  fmt::print("absolute difference = {:.3e}\n", std::abs(compact_result - explicit_result));
}

AsyncTask square(ReadBuffer<int> in, WriteBuffer<int> out)
{
  int const value = co_await in;
  in.release();
  co_await out = value * value;
  co_return;
}

AsyncTask sum(ReadBuffer<int> lhs, ReadBuffer<int> rhs, WriteBuffer<int> out)
{
  int const left = co_await lhs;
  int const right = co_await rhs;
  lhs.release();
  rhs.release();
  co_await out = left + right;
  co_return;
}

void reduction_gallery(std::filesystem::path const& output_dir)
{
  fmt::print("\n=== Map/reduce DAG ===\n");
  DebugScheduler scheduler;
  scheduler.pause();

  constexpr int n = 8;
  std::vector<Async<int>> inputs;
  std::vector<Async<int>> current;
  inputs.reserve(n);
  current.resize(n);

  for (int i = 0; i < n; ++i)
  {
    inputs.emplace_back(i + 1);
    inputs.back().debug_name(fmt::format("input[{}]", i));
    current[i].debug_name(fmt::format("square[{}]", i));
    auto task = square(inputs[i].read(), current[i].write());
    task.debug_name(fmt::format("square task[{}]", i));
    scheduler.schedule(std::move(task));
  }

  std::size_t level = 0;
  while (current.size() > 1)
  {
    std::vector<Async<int>> next;
    next.reserve((current.size() + 1) / 2);
    for (std::size_t i = 0; i + 1 < current.size(); i += 2)
    {
      Async<int> partial;
      partial.debug_name(fmt::format("reduce L{}[{}]", level, i / 2));
      auto task = sum(current[i].read(), current[i + 1].read(), partial.write());
      task.debug_name(fmt::format("sum L{}[{}]", level, i / 2));
      scheduler.schedule(std::move(task));
      next.push_back(std::move(partial));
    }
    if (current.size() % 2 == 1) next.push_back(std::move(current.back()));
    current = std::move(next);
    ++level;
  }

  write_dot(output_dir, "reduction-01-constructed");
  scheduler.resume();
  scheduler.run();
  write_dot(output_dir, "reduction-02-after-one-run");
  auto const result = current[0].get_wait(scheduler);
  write_dot(output_dir, "reduction-03-complete");

  fmt::print("sum of squares 1..{} = {}\n", n, result);
}

void print_build_mode()
{
#if UNI20_DEBUG_DAG
  fmt::print("UNI20_DEBUG_DAG=ON: gallery DOT includes async value nodes and buffer dependency edges.\n");
#elif UNI20_DEBUG_ASYNC_TASKS
  fmt::print("UNI20_DEBUG_ASYNC_TASKS=ON, UNI20_DEBUG_DAG=OFF: gallery DOT contains task/epoch state only.\n");
  fmt::print("Rebuild with -DUNI20_DEBUG_DAG=ON to include async value nodes and dependency edges.\n");
#else
  fmt::print("UNI20_DEBUG_ASYNC_TASKS=OFF: registry is a dummy and Graphviz DOT output would be empty.\n");
#endif
}

bool task_registry_enabled() noexcept
{
#if UNI20_DEBUG_ASYNC_TASKS
  return true;
#else
  return false;
#endif
}

void print_rebuild_hint()
{
  fmt::print("No DOT files were written.\n");
  fmt::print("Configure and build a DAG-instrumented example with:\n\n");
  fmt::print("  cmake -S . -B ./build_codex/build_gcc13_debug_dag \\\n");
  fmt::print("    -DCMAKE_BUILD_TYPE=Debug \\\n");
  fmt::print("    -DUNI20_DEBUG_DAG=ON\n");
  fmt::print("  cmake --build ./build_codex/build_gcc13_debug_dag --target async_dag_gallery_example\n\n");
  fmt::print("Then run:\n");
  fmt::print("  ./build_codex/build_gcc13_debug_dag/examples/async_dag_gallery_example /tmp/uni20-dag-gallery\n");
}

void print_graph_legend(std::filesystem::path const& output_dir)
{
  fmt::print("\nHow to read the DAG snapshots:\n");
  fmt::print("  data_N  : Async<T> value node; labels show storage identity, state, and optional value.\n");
  fmt::print("            state=unconstructed means shared_storage has no current T object.\n");
  fmt::print("  task_N  : coroutine task; state is constructed/running/suspended/leaked.\n");
  fmt::print("  epoch_N : read/write ordering generation for one async value.\n");
  fmt::print("  arg read/write    : coarse constructor-time ReadBuffer/WriteBuffer dependency.\n");
  fmt::print("  co_await read/write: dependency observed when the coroutine actually awaited.\n");
  fmt::print("  await read/write  : task currently queued on an epoch.\n");
  fmt::print("  red or pink       : diagnostic highlight, such as missing writer or dependency cycle.\n");
  fmt::print("  later snapshots may be sparse after completed tasks and epochs are destroyed.\n");
  fmt::print("Start with: {}\n", dot_path(output_dir, "reduction-01-constructed").string());
  fmt::print("For more detail see docs/async/dag_debug_examples.md\n");
}

} // namespace

int main(int argc, char** argv)
{
  print_build_mode();

  if (!task_registry_enabled())
  {
    print_rebuild_hint();
    return 1;
  }

  auto output_dir = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("async-dag-gallery-output");
  std::filesystem::create_directories(output_dir);

  fmt::print("Graphviz DOT output directory: {}\n", output_dir.string());

  branch_gallery(output_dir);
  kernel_shape_gallery(output_dir);
  reduction_gallery(output_dir);

  fmt::print("\nRender a snapshot with: dot -Tsvg {} -o async-dag-gallery.svg\n",
             dot_path(output_dir, "reduction-01-constructed").string());
  print_graph_legend(output_dir);
  return 0;
}
