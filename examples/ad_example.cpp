#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>

using namespace uni20::async;

struct ADTaskRunner
{
    using GraphBuilder = std::function<void(DebugScheduler&)>;

    explicit ADTaskRunner(GraphBuilder builder) : builder_(std::move(builder)) {}

    void run_forward()
    {
      DebugScheduler sched;
      TRACE("Running forward pass...");
      builder_(sched);
      sched.run_all();
      TRACE("Forward pass complete.");
    }

  private:
    GraphBuilder builder_;
};

AsyncTask compute(ReadBuffer<double> x, ReadBuffer<double> y, WriteBuffer<double> z)
{
  double xval = co_await x;
  double yval = co_await y;
  x.release();
  y.release();
  double result = (xval + yval) * yval;
  TRACE("Computed z =", result);
  co_await z = result;
  co_return;
}

int main()
{
  // Persistent buffers
  Async<double> x = 0.0;
  Async<double> y = 0.0;
  Async<double> z = 0.0;

  // Builder function captures inputs and sets up the graph
  ADTaskRunner runner([&](DebugScheduler& sched) { sched.schedule(compute(x.read(), y.read(), z.write())); });

  for (int i = 0; i < 3; ++i)
  {
    double xi = 1.0 + i;
    double yi = 2.0 + i;
    x.unsafe_set(xi);
    y.unsafe_set(yi);

    TRACE("=== Run", i + 1, "x =", xi, "y =", yi, "===");
    runner.run_forward();
    double zi = z.unsafe_value();
    TRACE("z =", zi);
  }

  return 0;
}
