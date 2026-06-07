#!/usr/bin/env bash
set -euo pipefail

# Run one DMRG generator and dump the value-free central-bond term structure at
# every sweep, producing a bond-dimension ladder in a single run (the central
# bond grows each sweep). Output is JSONL readable by tools/rabc_trace_reader.
#
# Usage:
#   scripts/run-rabc-term-trace-ladder.sh --model {heisenberg|hubbard} --out PATH [options]
#
# Options:
#   --model NAME       heisenberg (U(1)) or hubbard (U(1)xU(1)). Required.
#   --out PATH         Output JSONL (central-bond structure per sweep). Required.
#   --exe PATH         Generator executable (default: derived from --model).
#   --length N         Chain length L (default 40). Central bond is left_site=L/2-1.
#   --max-rank M       SVD max rank / target bond dimension (default 8192).
#   --sweeps N         Number of sweeps (default 16; needs ~log2(M)+buffer to saturate).
#   --cuda-visible L   CUDA_VISIBLE_DEVICES (default 0; single GPU).
#   --timeout SECONDS  Per-run timeout (default 7200).

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${repo_root}/build_codex/tensorcontraction-polaron-release-fresh/examples"
model=""; out=""; exe=""; length=40; max_rank=8192; sweeps=16; cuda_visible=0; timeout_s=7200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model="$2"; shift 2;;
    --out) out="$2"; shift 2;;
    --exe) exe="$2"; shift 2;;
    --length) length="$2"; shift 2;;
    --max-rank) max_rank="$2"; shift 2;;
    --sweeps) sweeps="$2"; shift 2;;
    --cuda-visible) cuda_visible="$2"; shift 2;;
    --timeout) timeout_s="$2"; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

[[ -n "$model" && -n "$out" ]] || { echo "--model and --out are required" >&2; exit 2; }

case "$model" in
  heisenberg)
    exe="${exe:-${build}/spin_half_heisenberg_u1_dmrg}"
    env_args=(UNI20_HEISENBERG_LENGTH="$length" UNI20_HEISENBERG_MAX_RANK="$max_rank" UNI20_HEISENBERG_SWEEPS="$sweeps");;
  hubbard)
    exe="${exe:-${build}/fermi_hubbard_u1u1_dmrg}"
    env_args=(UNI20_HUBBARD_LENGTH="$length" UNI20_HUBBARD_MAX_RANK="$max_rank" UNI20_HUBBARD_SWEEPS="$sweeps");;
  *) echo "unknown model: $model" >&2; exit 2;;
esac

[[ -x "$exe" ]] || { echo "generator not executable: $exe" >&2; exit 2; }
left_site=$(( length / 2 - 1 ))
mkdir -p "$(dirname "$out")"
: > "$out"

echo "model=$model exe=$exe length=$length max_rank=$max_rank sweeps=$sweeps left_site=$left_site out=$out"
env HWLOC_HIDE_ERRORS=2 CUDA_VISIBLE_DEVICES="$cuda_visible" UNI20_TENSORCONTRACTION_DEVICES=1 \
    "${env_args[@]}" \
    UNI20_RABC_TERM_TRACE_PATH="$out" UNI20_RABC_TERM_TRACE_LEFT_SITE="$left_site" \
    /usr/bin/timeout "$timeout_s" "$exe" > "${out%.jsonl}.log" 2>&1

echo "records=$(grep -c '"kind":"rabc_matvec"' "$out" 2>/dev/null || echo 0)"
grep -i RABC_TERM_TRACE "${out%.jsonl}.log" | tail -5 || true
