// rabc_trace.hpp — minimal header-only reader for R/A/B/C term-trace JSONL.
//
// A term trace is a value-free description of a block-sparse Hamiltonian-apply
// matvec: the contraction f-hypergraph (which terms exist) plus the per-block
// dimensions. Matrix-element values are NOT present and are not needed to model
// contraction cost, data movement, or placement.
//
// One JSONL line == one matvec record (kind == "rabc_matvec"). See README.md for
// the full schema and the GEMM-shape / flop / byte conventions.
//
// Apply mode (key invariant): this is the eigensolver self-map apply. The output
// R and the input center vector B occupy the SAME block space, so R = H*B has B's
// structure and can be fed back as the next Lanczos/Davidson iterate. Hence `r`
// and `b` index the same `block_count` blocks, and the placement is a single
// layout over that block space: output_layout == input_layout (R is co-located
// with B per block). The `block_count` field plus helpers `layout()` /
// `r_shares_b_layout()` expose this.
//
// Dependency: nlohmann/json (https://github.com/nlohmann/json), header-only.
//   #include <nlohmann/json.hpp> must be resolvable on the include path.
//
// Usage:
//   auto records = rabc::read_trace_file("uni20_l40_m4608_term_trace.jsonl");
//   for (const rabc::Matvec& mv : records)
//     for (const rabc::Term& t : mv.terms) { ... }

#ifndef RABC_TRACE_HPP
#define RABC_TRACE_HPP

#include <cstdint>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace rabc {

// One contraction term: R_r += coefficient * A_a * (B_b * C_c).
// Right-first evaluation forms Y = B_b * C_c (b_cols == c_rows), then
// R_r += A_a * Y (a_cols == b_rows); R_r is r_rows x r_cols == a_rows x c_cols.
struct Term
{
  int r = 0, a = 0, b = 0, c = 0;          // block indices into the bond's sectors
  double coefficient = 1.0;                 // scalar prefactor
  int device = -1;                          // placement in the dumping run (run-specific)

  int r_rows = 0, r_cols = 0;
  int a_rows = 0, a_cols = 0;
  int b_rows = 0, b_cols = 0;
  int c_rows = 0, c_cols = 0;

  // Derived weights (functions of the dimensions; 8-byte real scalar):
  //   bc_flops         = 2 * b_rows * b_cols * c_cols
  //   accumulate_flops = 2 * a_rows * a_cols * c_cols
  //   intermediate_bytes = b_rows * c_cols * 8   (the temporary Y = B*C)
  std::int64_t bc_flops = 0;
  std::int64_t accumulate_flops = 0;
  std::int64_t intermediate_bytes = 0;
};

// Per-device aggregate for the dumping run's layout (run-specific; optional).
struct DeviceStats
{
  int device = 0;
  int input_blocks = 0, output_blocks = 0, terms = 0;
  int unique_bc = 0, unique_a = 0, unique_b = 0, unique_c = 0;
  std::int64_t bc_flops = 0, accumulate_flops = 0;
  std::int64_t b_local_bytes = 0, b_peer_bytes = 0;
  std::int64_t a_bytes = 0, c_bytes = 0, output_bytes = 0, intermediate_bytes = 0;
};

// Measured per-device GPU time for the dumping run (run-specific; optional).
struct DeviceTiming
{
  int device = 0;
  double gpu_s = 0.0;
};

// One matvec record.
struct Matvec
{
  std::string kind;                         // "rabc_matvec"
  int index = 0;
  std::string policy;                       // placement policy used by the dumping run

  int device_count = 0;
  int block_count = 0;                      // number of center-vector blocks (bond sectors)
  int term_count = 0;

  // Run-specific timings (seconds); zero if the trace omits them.
  double enqueue_s = 0.0, sync_s = 0.0, wall_s = 0.0, gpu_s = 0.0;

  // Block -> device maps used by the dumping run (run-specific).
  std::vector<int> input_layout;
  std::vector<int> output_layout;

  // The structure (the part to keep): the f-hypergraph with dimensions.
  std::vector<Term> terms;

  // Run-specific aggregates (optional; present in full traces).
  std::vector<DeviceStats> devices;
  std::vector<DeviceTiming> device_timings;
};

// --- nlohmann/json deserialization (ADL) -----------------------------------

inline void from_json(const nlohmann::json& j, Term& t)
{
  // Connectivity is required; everything else is read leniently.
  j.at("r").get_to(t.r);
  j.at("a").get_to(t.a);
  j.at("b").get_to(t.b);
  j.at("c").get_to(t.c);
  t.coefficient = j.value("coefficient", 1.0);
  t.device = j.value("device", -1);
  t.r_rows = j.value("r_rows", 0);
  t.r_cols = j.value("r_cols", 0);
  t.a_rows = j.value("a_rows", 0);
  t.a_cols = j.value("a_cols", 0);
  t.b_rows = j.value("b_rows", 0);
  t.b_cols = j.value("b_cols", 0);
  t.c_rows = j.value("c_rows", 0);
  t.c_cols = j.value("c_cols", 0);
  t.bc_flops = j.value("bc_flops", std::int64_t{0});
  t.accumulate_flops = j.value("accumulate_flops", std::int64_t{0});
  t.intermediate_bytes = j.value("intermediate_bytes", std::int64_t{0});
}

inline void from_json(const nlohmann::json& j, DeviceStats& d)
{
  d.device = j.value("device", 0);
  d.input_blocks = j.value("input_blocks", 0);
  d.output_blocks = j.value("output_blocks", 0);
  d.terms = j.value("terms", 0);
  d.unique_bc = j.value("unique_bc", 0);
  d.unique_a = j.value("unique_a", 0);
  d.unique_b = j.value("unique_b", 0);
  d.unique_c = j.value("unique_c", 0);
  d.bc_flops = j.value("bc_flops", std::int64_t{0});
  d.accumulate_flops = j.value("accumulate_flops", std::int64_t{0});
  d.b_local_bytes = j.value("b_local_bytes", std::int64_t{0});
  d.b_peer_bytes = j.value("b_peer_bytes", std::int64_t{0});
  d.a_bytes = j.value("a_bytes", std::int64_t{0});
  d.c_bytes = j.value("c_bytes", std::int64_t{0});
  d.output_bytes = j.value("output_bytes", std::int64_t{0});
  d.intermediate_bytes = j.value("intermediate_bytes", std::int64_t{0});
}

inline void from_json(const nlohmann::json& j, DeviceTiming& d)
{
  d.device = j.value("device", 0);
  d.gpu_s = j.value("gpu_s", 0.0);
}

inline void from_json(const nlohmann::json& j, Matvec& m)
{
  m.kind = j.value("kind", std::string{});
  m.index = j.value("index", 0);
  m.policy = j.value("policy", std::string{});
  m.device_count = j.value("device_count", 0);
  m.block_count = j.value("block_count", 0);
  m.term_count = j.value("term_count", 0);
  m.enqueue_s = j.value("enqueue_s", 0.0);
  m.sync_s = j.value("sync_s", 0.0);
  m.wall_s = j.value("wall_s", 0.0);
  m.gpu_s = j.value("gpu_s", 0.0);
  m.input_layout = j.value("input_layout", std::vector<int>{});
  m.output_layout = j.value("output_layout", std::vector<int>{});
  m.terms = j.value("terms", std::vector<Term>{});
  m.devices = j.value("devices", std::vector<DeviceStats>{});
  m.device_timings = j.value("device_timings", std::vector<DeviceTiming>{});
}

// --- reading ----------------------------------------------------------------

// Parse a JSONL stream, returning every "rabc_matvec" record (other kinds and
// blank lines are skipped).
inline std::vector<Matvec> read_trace(std::istream& in)
{
  std::vector<Matvec> records;
  std::string line;
  while (std::getline(in, line))
  {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos)
      continue;
    nlohmann::json j = nlohmann::json::parse(line);
    if (j.value("kind", std::string{}) != "rabc_matvec")
      continue;
    records.push_back(j.get<Matvec>());
  }
  return records;
}

inline std::vector<Matvec> read_trace_file(const std::string& path)
{
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("rabc::read_trace_file: cannot open " + path);
  return read_trace(in);
}

// --- self-map apply helpers -------------------------------------------------

// True if R and B share the same per-block placement (the expected invariant for
// the eigensolver self-map apply, so R can be fed back as the next B iterate).
inline bool r_shares_b_layout(const Matvec& m)
{
  return m.input_layout == m.output_layout;
}

// The single block -> device layout over the center-vector block space. In the
// self-map apply input_layout == output_layout; this returns that shared layout
// (input_layout), which is also the placement domain for partitioning.
inline const std::vector<int>& layout(const Matvec& m)
{
  return m.input_layout;
}

}  // namespace rabc

#endif  // RABC_TRACE_HPP
