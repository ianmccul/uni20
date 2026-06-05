#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run-rabc-layout-sweep.sh --fixture PATH [options]

Run repeated no-trace TensorContraction R/A/B/C replay benchmarks for a list of
manual center-vector layouts.  The script writes one stdout, stderr, and
MP_BENCHFILE table per layout, then converts the bench tables into one JSONL
benchmark dataset.

Options:
  --fixture PATH          R/A/B/C fixture to replay. Required.
  --exe PATH              Benchmark executable.
  --output-dir PATH       Output directory. Default: /tmp/uni20_rabc_sweep_<timestamp>.
  --labels LIST           Comma-separated labels. Default: representative contiguous cuts.
                          Labels are looked up with rabc-trace-model.py layouts.
  --layout NAME=LIST      Add one custom layout. May be repeated. The custom
                          layout name is appended after --labels entries.
  --block-count N         Number of center-vector blocks. Default: 42.
  --device-count N        Number of local CUDA devices. Default: 2.
  --cuda-visible LIST     CUDA_VISIBLE_DEVICES value. Default: 0,1.
  --repeats N             Measured repeats per layout. Default: 3.
  --iters N               Lanczos max/min iterations. Default: 24.
  --timeout SECONDS       Per-layout timeout. Default: 240.
  --no-warmup             Disable one warmup replay before measured repeats.
  --resume                Reuse nonempty existing .bench files in the output directory.
  --trace-path PATH       Append R/A/B/C JSONL trace records to PATH. Use "auto"
                          for <output-dir>/trace.jsonl.
  --trace-terms           Include term metadata in trace records. Requires --trace-path.
  --placement-log         Print TensorContraction placement diagnostics to stderr.
  --help                  Show this help.

Outputs:
  <output-dir>/<label>.bench
  <output-dir>/<label>.out
  <output-dir>/<label>.err
  <output-dir>/benchmarks.jsonl
  <output-dir>/layouts.txt
  <output-dir>/trace.jsonl, if --trace-path auto is used
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture=""
exe="${repo_root}/build_codex/tensorcontraction-polaron-release-fresh/examples/tensorcontraction_rabc_lanczos_benchmark"
output_dir="/tmp/uni20_rabc_sweep_$(date +%Y%m%d_%H%M%S)"
labels_csv="cut3,cut4,cut5,cut6,cut7,cut8,cut9,cut10,cut12,cut14,cut16,cut21,cut36,alternating"
block_count=42
device_count=2
cuda_visible="0,1"
repeats=3
iters=24
timeout_seconds=240
warmup=1
resume=0
trace_path=""
trace_terms=0
placement_log=0
declare -A custom_layouts=()
custom_layout_names=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fixture)
      fixture="$2"
      shift 2
      ;;
    --exe)
      exe="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --labels)
      labels_csv="$2"
      shift 2
      ;;
    --layout)
      layout_name="${2%%=*}"
      layout_value="${2#*=}"
      if [[ -z "${layout_name}" || "${layout_name}" == "${2}" || -z "${layout_value}" ]]; then
        echo "--layout expects NAME=LIST" >&2
        exit 2
      fi
      custom_layouts["${layout_name}"]="${layout_value}"
      custom_layout_names+=("${layout_name}")
      shift 2
      ;;
    --block-count)
      block_count="$2"
      shift 2
      ;;
    --device-count)
      device_count="$2"
      shift 2
      ;;
    --cuda-visible)
      cuda_visible="$2"
      shift 2
      ;;
    --repeats)
      repeats="$2"
      shift 2
      ;;
    --iters)
      iters="$2"
      shift 2
      ;;
    --timeout)
      timeout_seconds="$2"
      shift 2
      ;;
    --no-warmup)
      warmup=0
      shift
      ;;
    --resume)
      resume=1
      shift
      ;;
    --trace-path)
      trace_path="$2"
      shift 2
      ;;
    --trace-terms)
      trace_terms=1
      shift
      ;;
    --placement-log)
      placement_log=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${fixture}" ]]; then
  echo "--fixture is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "${fixture}" ]]; then
  echo "fixture does not exist: ${fixture}" >&2
  exit 2
fi
if [[ ! -x "${exe}" ]]; then
  echo "benchmark executable is not executable: ${exe}" >&2
  exit 2
fi
if [[ "${trace_terms}" -eq 1 && -z "${trace_path}" ]]; then
  echo "--trace-terms requires --trace-path" >&2
  exit 2
fi

mkdir -p "${output_dir}"
if [[ "${trace_path}" == "auto" ]]; then
  trace_path="${output_dir}/trace.jsonl"
fi
if [[ -n "${trace_path}" ]]; then
  mkdir -p "$(dirname "${trace_path}")"
  if [[ "${resume}" -eq 0 ]]; then
    : > "${trace_path}"
  fi
fi
layouts_file="${output_dir}/layouts.txt"
jsonl="${output_dir}/benchmarks.jsonl"
"${repo_root}/scripts/rabc-trace-model.py" layouts --block-count "${block_count}" --device-count "${device_count}" \
  --contiguous-cuts > "${layouts_file}"
: > "${jsonl}"

labels=()
if [[ -n "${labels_csv}" ]]; then
  IFS=',' read -r -a raw_labels <<< "${labels_csv}"
  for raw_label in "${raw_labels[@]}"; do
    if [[ -n "${raw_label}" ]]; then
      labels+=("${raw_label}")
    fi
  done
fi
labels+=("${custom_layout_names[@]}")
if [[ "${#labels[@]}" -eq 0 ]]; then
  echo "no labels requested" >&2
  exit 2
fi

has_bench_rows() {
  local bench_file="$1"
  [[ -s "${bench_file}" ]] && awk 'NF > 0 && $1 !~ /^#/ { found = 1 } END { exit found ? 0 : 1 }' "${bench_file}"
}

safe_label() {
  local raw="$1"
  printf '%s' "${raw}" | sed 's/[^A-Za-z0-9._-]/_/g'
}

lookup_layout() {
  local item="$1"
  if [[ -v "custom_layouts[${item}]" ]]; then
    printf '%s' "${custom_layouts[${item}]}"
    return 0
  fi
  awk -v name="${item}" '$1 == name { print $2 }' "${layouts_file}"
}

record_bench() {
  local bench_file="$1"
  local name="$2"
  local layout="$3"
  local append_flag=()
  if [[ -s "${jsonl}" ]]; then
    append_flag=(--append)
  fi
  "${repo_root}/scripts/rabc-trace-model.py" bench-record "${bench_file}" \
    --name "${name}" \
    --layout "${layout}" \
    --input-format benchfile \
    --output "${jsonl}" \
    "${append_flag[@]}"
}

echo "fixture=${fixture}"
echo "exe=${exe}"
echo "output_dir=${output_dir}"
echo "labels=${labels_csv}"
echo "repeats=${repeats} iters=${iters} warmup=${warmup} timeout=${timeout_seconds}"
if [[ -n "${trace_path}" ]]; then
  echo "trace_path=${trace_path} trace_terms=${trace_terms}"
fi

for item in "${labels[@]}"; do
  label="${item%%=*}"
  layout="$(lookup_layout "${item}")"
  if [[ -z "${layout}" ]]; then
    echo "missing layout for label: ${item}" >&2
    exit 2
  fi

  file_label="$(safe_label "${label}")"
  bench_file="${output_dir}/${file_label}.bench"
  stdout_file="${output_dir}/${file_label}.out"
  stderr_file="${output_dir}/${file_label}.err"

  echo "=== ${label} ==="
  echo "layout=${layout}"
  if [[ "${resume}" -eq 1 ]] && has_bench_rows "${bench_file}"; then
    echo "resume=skip existing ${bench_file}"
  else
    env_args=(
      "HWLOC_HIDE_ERRORS=2"
      "CUDA_VISIBLE_DEVICES=${cuda_visible}"
      "UNI20_TENSORCONTRACTION_DEVICES=${device_count}"
      "UNI20_TENSORCONTRACTION_RABC_PLACEMENT=manual"
      "UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT=${layout}"
      "UNI20_RABC_LANCZOS_ITERS=${iters}"
      "UNI20_RABC_LANCZOS_MIN_ITERS=${iters}"
      "UNI20_RABC_REPEATS=${repeats}"
      "UNI20_RABC_WARMUP=${warmup}"
      "MP_BENCHFILE=${bench_file}"
    )
    if [[ -n "${trace_path}" ]]; then
      env_args+=("UNI20_TENSORCONTRACTION_RABC_TRACE_PATH=${trace_path}")
    fi
    if [[ "${trace_terms}" -eq 1 ]]; then
      env_args+=("UNI20_TENSORCONTRACTION_RABC_TRACE_TERMS=1")
    fi
    if [[ "${placement_log}" -eq 1 ]]; then
      env_args+=("UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LOG=1")
    fi
    env "${env_args[@]}" /usr/bin/timeout "${timeout_seconds}" "${exe}" "${fixture}" > "${stdout_file}" 2> "${stderr_file}"
  fi

  record_bench "${bench_file}" "${label}" "${layout}"
  tail -n "${repeats}" "${stdout_file}" || true
done

echo "=== summary ==="
"${repo_root}/scripts/rabc-trace-model.py" bench-summary "${jsonl}"
echo "benchmark_jsonl=${jsonl}"
if [[ -n "${trace_path}" ]]; then
  echo "trace_jsonl=${trace_path}"
fi
