#!/usr/bin/env bash
set -euo pipefail

build_dir=${UNI20_BENCH_BUILD_DIR:-build_codex/tensorcontraction-polaron}
example=${UNI20_BENCH_EXAMPLE:-"${build_dir}/examples/spin_half_heisenberg_dmrg"}
sweeps=${UNI20_BENCH_SWEEPS:-3}
out_dir=${UNI20_BENCH_OUT_DIR:-"/tmp/uni20_l20_rank_bench_$(date +%Y%m%d_%H%M%S)"}

if [[ $# -gt 0 ]]; then
  ranks=("$@")
else
  ranks=(16 32 64)
fi

mkdir -p "${out_dir}"

printf 'output_dir=%s\n' "${out_dir}"
printf 'example=%s\n' "${example}"
printf 'sweeps=%s\n' "${sweeps}"
printf 'ranks=%s\n' "${ranks[*]}"

for rank in "${ranks[@]}"; do
  bench="${out_dir}/m${rank}.bench"
  stdout="${out_dir}/m${rank}.out"
  stderr="${out_dir}/m${rank}.err"
  printf '\n[run] m=%s\n' "${rank}"
  /usr/bin/time -f 'WALL %e' \
    env HWLOC_HIDE_ERRORS=2 \
        UNI20_HEISENBERG_SWEEPS="${sweeps}" \
        UNI20_HEISENBERG_MAX_RANK="${rank}" \
        MP_BENCHFILE="${bench}" \
        "${example}" >"${stdout}" 2>"${stderr}"
done

python3 - "$out_dir" "${ranks[@]}" <<'PY'
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
ranks = sys.argv[2:]

print("\nsummary")
print("rank wall_s bench_s rows max_states final_energy env_sum_s solve_sum_s")
for rank in ranks:
    bench = out_dir / f"m{rank}.bench"
    stderr = out_dir / f"m{rank}.err"
    stdout = out_dir / f"m{rank}.out"
    rows = []
    for line in bench.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        rows.append(
            {
                "t": float(parts[0]),
                "states": int(parts[3]),
                "energy": float(parts[4]),
                "solve": float(parts[9]),
                "env": float(parts[12]),
            }
        )

    wall = "?"
    for line in stderr.read_text().splitlines():
        if line.startswith("WALL "):
            wall = line.split()[1]

    final = ""
    for line in stdout.read_text().splitlines():
        if line.startswith("sweep "):
            final = line

    if not rows:
        print(f"{rank} {wall} ? 0 0 ? 0 0")
        continue

    print(
        f"{rank} {wall} {rows[-1]['t']:.6f} {len(rows)} "
        f"{max(row['states'] for row in rows)} {rows[-1]['energy']:.15g} "
        f"{sum(row['env'] for row in rows):.6f} {sum(row['solve'] for row in rows):.6f}"
    )
    if final:
        print(f"# m={rank} {final}")
PY
