# R/A/B/C term-trace reader

A minimal, header-only C++ reader for the R/A/B/C **term-trace** JSONL produced by
the uni20 TensorContraction bridge. The term trace is a *value-free* description of
a block-sparse Hamiltonian-apply matvec: the contraction **f-hypergraph** (which
terms exist) plus the **per-block dimensions**. Matrix-element values are not
present and are not needed to model contraction cost, data movement, or placement.

This is intended as a drop-in so external code can read the traces into its own data
structures; the reader deliberately does no analysis.

## Apply mode (key invariant)

The trace is the **eigensolver self-map apply**: the output `R` and the input
center vector `B` occupy the *same* block space, so `R = H·B` has `B`'s structure
and is fed back as the next Lanczos/Davidson iterate. Consequently:

- `r` and `b` both index the same `block_count` center-vector blocks.
- The placement is a **single layout** over that block space:
  `output_layout == input_layout` (R is co-located with B per block). This is the
  placement domain a partitioner operates on. `rabc::layout()` returns it and
  `rabc::r_shares_b_layout()` checks the invariant.

(`a` and `c` index the left/right environment block spaces.)

## Build / integrate

Header-only; the single dependency is [nlohmann/json](https://github.com/nlohmann/json).

Integrate into an existing CMake build:

```cmake
add_subdirectory(rabc_trace_reader)
target_link_libraries(your_target PRIVATE rabc_trace)   # brings in nlohmann/json
```

Or build the example standalone (fetches nlohmann/json):

```bash
cmake -S . -B build && cmake --build build -j
./build/rabc_trace_example uni20_l40_m4608_term_trace.jsonl
```

If you already have nlohmann/json, just put `rabc_trace.hpp` on your include path
and `#include <nlohmann/json.hpp>` resolves.

```cpp
auto records = rabc::read_trace_file("uni20_l40_m4608_term_trace.jsonl");
for (const rabc::Matvec& mv : records)
  for (const rabc::Term& t : mv.terms)
    /* R_t.r += t.coefficient * A_t.a * (B_t.b * C_t.c) */;
```

## Contraction semantics and shape conventions

Each term is `R_r += coefficient · A_a · (B_b · C_c)`, evaluated right-first:

1. `Y = B_b · C_c`  — contracts `b_cols == c_rows`; `Y` is `b_rows × c_cols`.
2. `R_r += A_a · Y` — contracts `a_cols == b_rows`; `R_r` is `a_rows × c_cols == r_rows × r_cols`.

Derived weights (real, 8-byte scalar):

| field | formula |
|---|---|
| `bc_flops` | `2 · b_rows · b_cols · c_cols` |
| `accumulate_flops` | `2 · a_rows · a_cols · c_cols` |
| `intermediate_bytes` | `b_rows · c_cols · 8` (the temporary `Y`) |

## JSONL schema

One line per matvec record. Records with `kind != "rabc_matvec"` (and blank lines)
should be skipped.

### Record (object)

| field | type | keep? | meaning |
|---|---|---|---|
| `kind` | string | — | always `"rabc_matvec"` |
| `index` | int | — | record index in the dump |
| `policy` | string | run | placement policy used by the dumping run |
| `device_count` | int | run | devices in the dumping run |
| `block_count` | int | **structure** | center-vector blocks; size of the placement domain (`r`,`b` range) |
| `term_count` | int | structure | number of terms |
| `enqueue_s`,`sync_s`,`wall_s`,`gpu_s` | double | run | timings (seconds); may be absent/zero |
| `input_layout` | int[`block_count`] | run | block → device for `B` |
| `output_layout` | int[`block_count`] | run | block → device for `R` (== `input_layout` in this mode) |
| `terms` | object[] | **structure** | the f-hypergraph (below) |
| `devices` | object[] | run | per-device aggregates for the dumping layout |
| `device_timings` | object[] | run | measured per-device GPU time |

### `terms[]` (object) — the structure to keep

| field | type | meaning |
|---|---|---|
| `r`,`a`,`b`,`c` | int | block indices; `r`,`b` in the center-vector space, `a`/`c` in the left/right environments |
| `coefficient` | double | scalar prefactor |
| `device` | int | device that ran this term in the dumping run (run-specific) |
| `r_rows`,`r_cols`,`a_rows`,`a_cols`,`b_rows`,`b_cols`,`c_rows`,`c_cols` | int | block dimensions |
| `bc_flops`,`accumulate_flops` | int64 | derived flop counts (see formulas) |
| `intermediate_bytes` | int64 | derived bytes of the `Y` temporary |

### `devices[]` (object) — run-specific aggregates

`device`, `input_blocks`, `output_blocks`, `terms`, `unique_bc`, `unique_a`,
`unique_b`, `unique_c` (int); `bc_flops`, `accumulate_flops`, `b_local_bytes`,
`b_peer_bytes`, `a_bytes`, `c_bytes`, `output_bytes`, `intermediate_bytes` (int64).

### `device_timings[]` (object) — run-specific

`device` (int), `gpu_s` (double).

## Notes for benchmark generation

For representative data, dump the **central-bond iteration** across a bond-dimension
ladder per model (`{U(1) Heisenberg, U(1)×U(1) Hubbard} × {m}`), tuning lattice size
and sweeps so the central bond reaches each `m`. The connectivity follows
analytically from the model and selection rule, but the **block dimensions are the
entanglement spectrum** and must be obtained by actually running the DMRG — they
cannot be freely synthesized. See `docs/tensorcontraction_integration_findings.md`.
