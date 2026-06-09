// example.cpp — minimal demonstration of the R/A/B/C term-trace reader.
//
// Usage: ./rabc_trace_example <term_trace.jsonl>
//
// Prints a per-record summary: the f-hypergraph size, the placement layout, and
// the total contraction weights (flops/bytes) derived purely from block
// dimensions. No matrix-element values are involved.

#include <cstdint>
#include <iostream>
#include <set>
#include <string>

#include "rabc_trace.hpp"

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cerr << "usage: " << argv[0] << " <term_trace.jsonl>\n";
    return 2;
  }

  const std::vector<rabc::Matvec> records = rabc::read_trace_file(argv[1]);
  std::cout << "records: " << records.size() << "\n";

  for (const rabc::Matvec& mv : records)
  {
    std::int64_t bc = 0, acc = 0, ibytes = 0;
    std::set<std::pair<int, int>> bc_groups;
    for (const rabc::Term& t : mv.terms)
    {
      bc += t.bc_flops;
      acc += t.accumulate_flops;
      ibytes += t.intermediate_bytes;
      bc_groups.insert({t.b, t.c});
    }

    std::cout << "\n[record " << mv.index << "] policy=" << mv.policy
              << " devices=" << mv.device_count << " blocks=" << mv.block_count
              << " terms=" << mv.terms.size() << " bc_groups=" << bc_groups.size()
              << "\n";
    std::cout << "  R shares B layout: " << (rabc::r_shares_b_layout(mv) ? "yes" : "NO")
              << " (placement domain = " << rabc::layout(mv).size() << " blocks)\n";
    std::cout << "  total bc_flops=" << bc << " accumulate_flops=" << acc
              << " intermediate_bytes=" << ibytes << "\n";

    if (!mv.terms.empty())
    {
      const rabc::Term& t = mv.terms.front();
      std::cout << "  term[0]: R_" << t.r << " += " << t.coefficient << " * A_" << t.a
                << " * (B_" << t.b << " * C_" << t.c << ")"
                << "  [A " << t.a_rows << "x" << t.a_cols << ", B " << t.b_rows << "x"
                << t.b_cols << ", C " << t.c_rows << "x" << t.c_cols << " -> R " << t.r_rows
                << "x" << t.r_cols << "]\n";
    }
  }
  return 0;
}
