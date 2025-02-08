
#include "common/trace.hpp"

int add(int a, int b) { return a + b; }

int main()
{
  int foo = 42;
  std::string bar = "example";
  std::vector<int> vec(5, 0);
  std::vector<std::vector<int>> vec2{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  std::vector<std::vector<double>> vec3{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};

  // A simple trace of variables.
  TRACE(foo, bar, vec);

  // A trace with a literal string as the first parameter.
  TRACE("Literal string", foo, bar);

  // A trace with expressions.
  TRACE(foo + 1, bar + "_suffix", std::vector<int, std::allocator<int>>{1, 2, 3});

  // A trace that includes a template instantiation and a string literal.
  TRACE(std::vector<int>(5), "Hello, world", foo + 2);

  // A trace with a function call and an expression involving a literal.
  TRACE(add(foo, 3), "Result: " + std::to_string(add(foo, 3)));

  TRACE("Multi-line output", vec2, foo, vec3, foo);

  // change the precision
  trace::formatting_options.fp_precision<double> = 5;
  TRACE("Modified number of digits displayed", vec2, foo, vec3, foo);

  DEBUG_TRACE("Modified number of digits displayed", vec2, foo, vec3, foo);

  TRACE_MODULE(TESTMODULE, foo, bar);

  return 0;
}
