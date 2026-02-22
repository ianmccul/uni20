
#include "common/trace.hpp"
#include <thread>

// To customize output colors, you can set environment variables like:
// export UNI20_COLOR_TRACE_VALUE="fg:Green;Bold"
// export UNI20_TRACE_TIMESTAMP="yes"
// export UNI20_TRACE_THREAD_ID="yes"
// Then run this example again.

int add(int a, int b) { return a + b; }

int main()
{
  int foo = 42;
  std::string bar = "example";
  std::vector<int> vec(5, 0);
  std::vector<std::vector<int>> vec2{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  std::vector<std::vector<double>> vec3{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
  double Pi = M_PI;

  TRACE(vec2, foo);

  // A simple trace of variables.
  TRACE(foo, bar, vec);

  // A trace with a literal string as the first parameter.
  TRACE("Literal string", foo, bar);

  // A trace with expressions.
  TRACE(foo + 1, bar + "_suffix",
        "Template diamond brackets will garble the output. The solution is to put the template expression in brackets "
        "().",
        std::vector<int, std::allocator<int>>{1, 2, 3}, foo + 2);

  TRACE(foo<32, foo> 32);

  // A trace that includes a template instantiation and a string literal.
  TRACE(std::vector<int>(5), "Hello, world", foo + 2);

  // A trace with a function call and an expression involving a literal.
  TRACE(add(foo, 3), "Result: " + std::to_string(add(foo, 3)));

  TRACE("Multi-line output", vec2, foo, vec3, foo);

  // change the precision
  trace::get_formatting_options().fp_precision_float64 = 5;

  // Add timestamps and thread ID. NOTE: the preferred way to set this is via environment variables,
  // export UNI20_TRACE_TIMESTAMP=no
  // export UNI20_TRACE_THREAD_ID=no
  trace::get_formatting_options().timestamp = false;
  trace::get_formatting_options().threadId = trace::FormattingOptions::ThreadIdOptions::no;

  TRACE("Modified number of digits displayed; removed timestamp, thread ID:", vec2, foo, vec3, foo);

  DEBUG_TRACE("Modified number of digits displayed", vec2, foo, vec3, foo);

  TRACE_MODULE(TESTMODULE, foo, bar, Pi);

  // In a second thread
  std::thread([] {
    int x = 99;
    TRACE("From another thread", x);
  }).join();

  PANIC("Test panic; the program will abort now", foo);

  return 0;
}
