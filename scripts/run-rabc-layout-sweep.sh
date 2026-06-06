#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run-rabc-layout-sweep.sh --fixture PATH [options]

Run repeated no-trace TensorContraction R/A/B/C replay benchmarks for manual
center-vector layouts or automatic placement policies.  The script writes one
stdout, stderr, and MP_BENCHFILE table per run.  Runs with known or inferred
layouts are also converted into one JSONL benchmark dataset.

Options:
  --fixture PATH          R/A/B/C fixture to replay. Required.
  --exe PATH              Benchmark executable.
  --output-dir PATH       Output directory. Default: /tmp/uni20_rabc_sweep_<timestamp>.
  --policy POLICY         R/A/B/C placement policy. Default: manual.
                          Non-manual policies use labels as run names only.
  --labels LIST           Comma-separated labels. Default: representative contiguous cuts.
                          Labels are looked up with rabc-trace-model.py layouts.
                          For non-manual policies, labels are run names only.
  --layout NAME=LIST      Add one custom layout. May be repeated. The custom
                          layout name is appended after --labels entries.
  --layout-file NAME=PATH Add one custom layout read from PATH. The file may
                          contain either a raw comma-separated layout or a
                          bench-suggest output line beginning with "layout=".
  --segmented-cuts        Include bounded segmented layout labels in the generated
                          layout table for lookup by --labels.
  --max-segments N        Maximum segments for --segmented-cuts. Default: 2.
  --segment-cut-stride N  Only generate segmented cuts at multiples of this
                          block stride. Default: 1.
  --max-segment-layouts N Reject segmented generation above this layout count.
                          Default: 20000.
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
  --empirical-coefficients-file PATH
                          Pass a fitted empirical-contiguous coefficient file
                          to the benchmark runtime.
  --placement-log         Print TensorContraction placement diagnostics to stderr.
                          Enabled automatically for non-manual policies so the
                          selected layout can be recorded when possible.
  --show-layouts          Print full manual placement lists to stdout. By default
                          large layouts are summarized compactly.
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
policy="manual"
default_labels_csv="cut3,cut4,cut5,cut6,cut7,cut8,cut9,cut10,cut12,cut14,cut16,cut21,cut36,alternating"
labels_csv="${default_labels_csv}"
labels_was_set=0
segmented_cuts=0
max_segments=2
segment_cut_stride=1
max_segment_layouts=20000
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
empirical_coefficients_file=""
placement_log=0
show_layouts=0
declare -A custom_layouts=()
custom_layout_names=()

read_layout_file() {
  local layout_path="$1"
  local layout_value=""
  layout_value="$(
    awk '
      /^layout=/ {
        value = substr($0, 8)
        gsub(/[[:space:]]/, "", value)
        print value
        found = 1
        exit
      }
      END {
        exit found ? 0 : 1
      }
    ' "${layout_path}" || true
  )"
  if [[ -z "${layout_value}" ]]; then
    layout_value="$(tr -d '[:space:]' < "${layout_path}")"
  fi
  printf '%s' "${layout_value}"
}

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
    --policy)
      policy="$2"
      shift 2
      ;;
    --labels)
      labels_csv="$2"
      labels_was_set=1
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
    --layout-file)
      layout_name="${2%%=*}"
      layout_path="${2#*=}"
      if [[ -z "${layout_name}" || "${layout_name}" == "${2}" || -z "${layout_path}" ]]; then
        echo "--layout-file expects NAME=PATH" >&2
        exit 2
      fi
      if [[ ! -f "${layout_path}" ]]; then
        echo "layout file does not exist: ${layout_path}" >&2
        exit 2
      fi
      layout_value="$(read_layout_file "${layout_path}")"
      if [[ -z "${layout_value}" ]]; then
        echo "layout file is empty: ${layout_path}" >&2
        exit 2
      fi
      custom_layouts["${layout_name}"]="${layout_value}"
      custom_layout_names+=("${layout_name}")
      shift 2
      ;;
    --segmented-cuts)
      segmented_cuts=1
      shift
      ;;
    --max-segments)
      max_segments="$2"
      shift 2
      ;;
    --segment-cut-stride)
      segment_cut_stride="$2"
      shift 2
      ;;
    --max-segment-layouts)
      max_segment_layouts="$2"
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
    --empirical-coefficients-file)
      empirical_coefficients_file="$2"
      shift 2
      ;;
    --placement-log)
      placement_log=1
      shift
      ;;
    --show-layouts)
      show_layouts=1
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
if [[ -n "${empirical_coefficients_file}" && ! -f "${empirical_coefficients_file}" ]]; then
  echo "empirical coefficient file does not exist: ${empirical_coefficients_file}" >&2
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
layout_args=(--block-count "${block_count}" --device-count "${device_count}" --contiguous-cuts)
if [[ "${segmented_cuts}" -eq 1 ]]; then
  layout_args+=(
    --segmented-cuts
    --max-segments "${max_segments}"
    --segment-cut-stride "${segment_cut_stride}"
    --max-segment-layouts "${max_segment_layouts}"
  )
fi
"${repo_root}/scripts/rabc-trace-model.py" layouts "${layout_args[@]}" > "${layouts_file}"
: > "${jsonl}"

manual_policy=0
if [[ "${policy}" == "manual" || "${policy}" == "layout" ]]; then
  manual_policy=1
fi
if [[ "${manual_policy}" -eq 0 && "${labels_was_set}" -eq 0 && "${#custom_layout_names[@]}" -eq 0 ]]; then
  labels_csv="${policy}"
fi

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

layout_from_cut() {
  local cut="$1"
  local block
  for ((block = 0; block < block_count; ++block)); do
    if [[ "${block}" -gt 0 ]]; then
      printf ','
    fi
    if [[ "${block}" -lt "${cut}" ]]; then
      printf '0'
    else
      printf '1'
    fi
  done
  printf '\n'
}

layout_from_ranges_line() {
  local line="$1"
  local remaining="${line}"
  local pattern='device([0-9]+)=\{blocks=\[([0-9]+),([0-9]+)\)'
  local -a inferred=()
  local block
  for ((block = 0; block < block_count; ++block)); do
    inferred["${block}"]=""
  done

  while [[ "${remaining}" =~ ${pattern} ]]; do
    local device="${BASH_REMATCH[1]}"
    local begin="${BASH_REMATCH[2]}"
    local end="${BASH_REMATCH[3]}"
    if [[ "${device}" -ge "${device_count}" || "${begin}" -gt "${end}" || "${end}" -gt "${block_count}" ]]; then
      return 1
    fi
    for ((block = begin; block < end; ++block)); do
      inferred["${block}"]="${device}"
    done
    remaining="${remaining#*"${BASH_REMATCH[0]}"}"
  done

  for ((block = 0; block < block_count; ++block)); do
    if [[ -z "${inferred[${block}]}" ]]; then
      return 1
    fi
    if [[ "${block}" -gt 0 ]]; then
      printf ','
    fi
    printf '%s' "${inferred[${block}]}"
  done
  printf '\n'
}

layout_from_policy_log() {
  local stderr_file="$1"
  local policy_name="$2"
  local line=""

  case "${policy_name}" in
    stripe|striped|round-robin|alternating)
      local block
      for ((block = 0; block < block_count; ++block)); do
        if [[ "${block}" -gt 0 ]]; then
          printf ','
        fi
        printf '%d' $((block % device_count))
      done
      printf '\n'
      return 0
      ;;
  esac

  if [[ ! -s "${stderr_file}" ]]; then
    return 1
  fi

  line="$(grep -E '\[TENSORCONTRACTION\]\[RABC_PLACEMENT\].*(cut=|blocks=\[)' "${stderr_file}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    return 1
  fi

  if [[ "${line}" =~ (^|[[:space:]])cut=([0-9]+) ]]; then
    layout_from_cut "${BASH_REMATCH[2]}"
    return 0
  fi
  if [[ "${line}" =~ fallback=default-byte-balanced && "${line}" =~ default_cut=([0-9]+) ]]; then
    layout_from_cut "${BASH_REMATCH[1]}"
    return 0
  fi
  layout_from_ranges_line "${line}"
}

layout_summary_from_layout() {
  local layout_value="$1"
  awk -F',' -v device_count="${device_count}" '{
    for (device = 0; device < device_count; ++device) {
      counts[device] = 0
    }
    previous = ""
    segments = 0
    for (field = 1; field <= NF; ++field) {
      ++counts[$field]
      if (field == 1 || $field != previous) {
        ++segments
        previous = $field
      }
    }
    printf "blocks=%d;counts=", NF
    for (device = 0; device < device_count; ++device) {
      if (device > 0) {
        printf ","
      }
      printf "%d:%d", device, counts[device]
    }
    printf ";segments=%d", segments
    if (segments > 1) {
      printf ";transitions=%d", segments - 1
    }
    printf "\n"
  }' <<< "${layout_value}"
}

echo "fixture=${fixture}"
echo "exe=${exe}"
echo "output_dir=${output_dir}"
echo "policy=${policy}"
echo "labels=${labels_csv}"
echo "repeats=${repeats} iters=${iters} warmup=${warmup} timeout=${timeout_seconds}"
if [[ -n "${trace_path}" ]]; then
  echo "trace_path=${trace_path} trace_terms=${trace_terms}"
fi

for item in "${labels[@]}"; do
  label="${item}"
  layout=""
  if [[ "${manual_policy}" -eq 1 ]]; then
    layout="$(lookup_layout "${item}")"
    if [[ -z "${layout}" ]]; then
      echo "missing layout for label: ${item}" >&2
      exit 2
    fi
  fi

  file_label="$(safe_label "${label}")"
  bench_file="${output_dir}/${file_label}.bench"
  stdout_file="${output_dir}/${file_label}.out"
  stderr_file="${output_dir}/${file_label}.err"

  echo "=== ${label} ==="
  if [[ "${manual_policy}" -eq 1 ]]; then
    if [[ "${show_layouts}" -eq 1 ]]; then
      echo "layout=${layout}"
    else
      counts="$(
        awk -F',' -v device_count="${device_count}" '{
          for (device = 0; device < device_count; ++device) {
            counts[device] = 0
          }
          for (field = 1; field <= NF; ++field) {
            ++counts[$field]
          }
          for (device = 0; device < device_count; ++device) {
            if (device > 0) {
              printf ","
            }
            printf "%d:%d", device, counts[device]
          }
          printf "\n"
        }' <<< "${layout}"
      )"
      echo "layout_summary=blocks=${block_count};counts=${counts};stored_in=${layouts_file}"
    fi
  fi
  if [[ "${resume}" -eq 1 ]] && has_bench_rows "${bench_file}"; then
    echo "resume=skip existing ${bench_file}"
  else
    env_args=(
      "HWLOC_HIDE_ERRORS=2"
      "CUDA_VISIBLE_DEVICES=${cuda_visible}"
      "UNI20_TENSORCONTRACTION_DEVICES=${device_count}"
      "UNI20_TENSORCONTRACTION_RABC_PLACEMENT=${policy}"
      "UNI20_RABC_LANCZOS_ITERS=${iters}"
      "UNI20_RABC_LANCZOS_MIN_ITERS=${iters}"
      "UNI20_RABC_REPEATS=${repeats}"
      "UNI20_RABC_WARMUP=${warmup}"
      "MP_BENCHFILE=${bench_file}"
    )
    if [[ "${manual_policy}" -eq 1 ]]; then
      env_args+=("UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT=${layout}")
    fi
    if [[ -n "${empirical_coefficients_file}" ]]; then
      env_args+=("UNI20_TENSORCONTRACTION_RABC_EMPIRICAL_COEFFICIENTS_FILE=${empirical_coefficients_file}")
    fi
    if [[ -n "${trace_path}" ]]; then
      env_args+=("UNI20_TENSORCONTRACTION_RABC_TRACE_PATH=${trace_path}")
    fi
    if [[ "${trace_terms}" -eq 1 ]]; then
      env_args+=("UNI20_TENSORCONTRACTION_RABC_TRACE_TERMS=1")
    fi
    if [[ "${placement_log}" -eq 1 || "${manual_policy}" -eq 0 ]]; then
      env_args+=("UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LOG=1")
    fi
    env "${env_args[@]}" /usr/bin/timeout "${timeout_seconds}" "${exe}" "${fixture}" > "${stdout_file}" 2> "${stderr_file}"
  fi

  if [[ "${manual_policy}" -eq 1 ]]; then
    record_bench "${bench_file}" "${label}" "${layout}"
  else
    inferred_layout=""
    if inferred_layout="$(layout_from_policy_log "${stderr_file}" "${policy}")" && [[ -n "${inferred_layout}" ]]; then
      record_bench "${bench_file}" "${label}" "${inferred_layout}"
      if [[ "${show_layouts}" -eq 1 ]]; then
        echo "inferred_layout=${inferred_layout}"
      else
        echo "inferred_layout_summary=$(layout_summary_from_layout "${inferred_layout}")"
      fi
    else
      echo "inferred_layout=unavailable"
    fi
  fi
  tail -n "${repeats}" "${stdout_file}" || true
done

echo "=== summary ==="
if [[ -s "${jsonl}" ]]; then
  summary_args=()
  if [[ "${show_layouts}" -eq 0 ]]; then
    summary_args+=(--compact-layouts)
  fi
  "${repo_root}/scripts/rabc-trace-model.py" bench-summary "${jsonl}" "${summary_args[@]}"
  echo "benchmark_jsonl=${jsonl}"
else
  echo "benchmark_jsonl=not_written_for_policy_${policy}"
fi
if [[ -n "${trace_path}" ]]; then
  echo "trace_jsonl=${trace_path}"
fi
