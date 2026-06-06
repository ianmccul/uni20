#!/usr/bin/env python3
"""Fit and query empirical R/A/B/C layout cost models from JSONL traces."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


BASE_FEATURE_NAMES = [
    "intercept",
    "bc_flops",
    "accumulate_flops",
    "temporary_accumulate_flops",
    "b_local_bytes",
    "b_peer_bytes",
    "temporary_peer_request_bytes",
    "temporary_peer_bytes",
    "a_bytes",
    "c_bytes",
    "output_bytes",
    "intermediate_bytes",
    "terms",
    "unique_bc",
    "unique_a",
    "unique_b",
    "unique_c",
    "bc_gemms",
    "final_gemms",
    "direct_final_gemms",
    "accumulation_groups",
    "accumulation_terms",
    "source_accumulation_groups",
    "source_accumulation_terms",
    "output_accumulation_groups",
    "output_accumulation_terms",
    "source_axpys",
    "output_axpys",
    "zero_fills",
    "intermediate_matrices",
    "temporary_matrices",
    "temporary_peer_requests",
    "temporary_peer_copies",
]

GRAPH_FEATURE_NAMES = [
    "b_cut_terms",
    "b_peer_blocks",
    "right_duplicate_groups",
]

BENCHMARK_LAYOUT_FEATURE_NAMES = [
    "layout_transitions",
    "layout_segments",
    "active_devices",
    "max_output_block_fraction",
    "max_output_byte_fraction",
]

ILP_LAYOUT_FEATURE_NAMES = [
    "layout_transitions",
    "layout_segments",
]

DEVICE_BENCHMARK_FEATURE_NAMES = [
    "right_flops",
    "b_peer_bytes",
    "terms",
    "unique_bc",
    "output_bytes",
]

DEVICE_BENCHMARK_GRAPH_FEATURE_NAMES = [
    "b_cut_terms",
    "b_peer_blocks",
    "right_duplicate_groups",
    "mixed_duplicate_groups",
    "mixed_left_groups",
    "mixed_right_groups",
]

CALIBRATION_SEED_FEATURE_NAMES = [
    "intercept",
    "bc_flops",
    "accumulate_flops",
    "temporary_accumulate_flops",
    "bc_gemms",
    "final_gemms",
    "source_axpys",
    "output_axpys",
    "zero_fills",
    "temporary_peer_requests",
    "temporary_peer_bytes",
]

STRUCTURE_SCORE_FEATURE_NAMES = [
    "right_max_gflop",
    "mixed_max_gflop",
    "b_peer_mb",
    "b_peer_blocks",
    "b_cut_terms",
    "max_terms",
    "max_unique_bc",
    "max_output_mb",
    "segments",
    "transitions",
    "max_output_byte_fraction",
    "right_duplicate_groups",
    "mixed_duplicate_groups",
    "mixed_left_groups",
    "mixed_right_groups",
]

STRUCTURE_SCORE_FEATURE_SETS = {
    "all": STRUCTURE_SCORE_FEATURE_NAMES,
    "execution-pressure": [
        "mixed_max_gflop",
        "max_terms",
        "max_unique_bc",
        "b_peer_mb",
        "b_peer_blocks",
        "mixed_duplicate_groups",
        "segments",
        "transitions",
    ],
    "launch-pressure": [
        "max_terms",
        "max_unique_bc",
        "mixed_left_groups",
        "mixed_right_groups",
        "mixed_duplicate_groups",
        "b_peer_blocks",
        "b_cut_terms",
        "segments",
        "transitions",
    ],
    "no-output": [
        "right_max_gflop",
        "mixed_max_gflop",
        "b_peer_mb",
        "b_peer_blocks",
        "b_cut_terms",
        "max_terms",
        "max_unique_bc",
        "segments",
        "transitions",
        "right_duplicate_groups",
        "mixed_duplicate_groups",
        "mixed_left_groups",
        "mixed_right_groups",
    ],
    "typed-hypergraph": [
        "mixed_max_gflop",
        "right_reuse_split_first_gflop",
        "left_reuse_split_first_gflop",
        "b_fanout_split_mb",
        "b_fanout_split_blocks",
        "right_reuse_split_groups",
        "left_reuse_split_groups",
        "rb_cut_terms",
        "rb_cut_edges",
        "max_terms",
        "max_unique_bc",
        "segments",
        "transitions",
    ],
}

BENCHMARK_REPEAT_RE = re.compile(
    r"repeat=(?P<repeat>\d+)\s+wall=(?P<wall>[0-9.eE+-]+)s\b.*\bmatvec=(?P<matvec>[0-9.eE+-]+)s\b"
)


def feature_names(include_graph_features: bool) -> list[str]:
    """Return the active model feature names."""
    if include_graph_features:
        return BASE_FEATURE_NAMES + GRAPH_FEATURE_NAMES
    return BASE_FEATURE_NAMES


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Read R/A/B/C JSONL trace records."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") != "rabc_matvec":
            continue
        record["_line"] = line_number
        records.append(record)
    return records


def read_benchmark_records(path: Path) -> list[dict[str, Any]]:
    """Read no-trace R/A/B/C replay benchmark JSONL records."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") != "rabc_replay_benchmark":
            continue
        record["_line"] = line_number
        records.append(record)
    return records


def read_calibration_records(path: Path) -> list[dict[str, Any]]:
    """Read synthetic R/A/B/C calibration JSONL records."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") != "rabc_calibration":
            continue
        record["_line"] = line_number
        records.append(record)
    return records


def parse_benchmark_stdout(path: Path, layout: str, name: str) -> list[dict[str, Any]]:
    """Parse replay benchmark stdout into benchmark records."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        match = BENCHMARK_REPEAT_RE.search(line)
        if match is None:
            continue
        records.append(
            {
                "kind": "rabc_replay_benchmark",
                "source": str(path),
                "line": line_number,
                "name": name,
                "layout": layout,
                "repeat": int(match.group("repeat")),
                "wall_s": float(match.group("wall")),
                "matvec_s": float(match.group("matvec")),
            }
        )
    if not records:
        raise ValueError(f"{path} contains no R/A/B/C replay benchmark repeat rows")
    return records


def parse_benchmark_benchfile(path: Path, layout: str, name: str) -> list[dict[str, Any]]:
    """Parse an `MP_BENCHFILE` table into benchmark records."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 8:
            continue
        try:
            repeat = int(fields[0])
            wall_s = float(fields[1])
            energy = float(fields[2])
            residual = float(fields[3])
            iterations = int(fields[4])
            stop_reason = fields[5]
            matvec_s = float(fields[6])
            matvec_count = int(fields[7])
        except ValueError:
            continue
        records.append(
            {
                "kind": "rabc_replay_benchmark",
                "source": str(path),
                "line": line_number,
                "name": name,
                "layout": layout,
                "repeat": repeat,
                "wall_s": wall_s,
                "energy": energy,
                "residual": residual,
                "iterations": iterations,
                "stop_reason": stop_reason,
                "matvec_s": matvec_s,
                "matvec_count": matvec_count,
                "matvec_per_apply_s": matvec_s / matvec_count if matvec_count != 0 else math.nan,
            }
        )
    if not records:
        raise ValueError(f"{path} contains no R/A/B/C replay benchmark table rows")
    return records


def benchmark_matvec_seconds(record: dict[str, Any]) -> float:
    """Return normalized replay matvec time for one record."""
    if "matvec_per_apply_s" in record:
        return float(record["matvec_per_apply_s"])
    if "matvec_count" in record:
        count = int(record["matvec_count"])
        return float(record["matvec_s"]) / count if count != 0 else math.nan
    return float(record["matvec_s"])


def parse_benchmark_file(path: Path, layout: str, name: str, input_format: str) -> list[dict[str, Any]]:
    """Parse replay benchmark rows from stdout or `MP_BENCHFILE` format."""
    if input_format == "stdout":
        return parse_benchmark_stdout(path, layout, name)
    if input_format == "benchfile":
        return parse_benchmark_benchfile(path, layout, name)
    if input_format == "auto":
        try:
            return parse_benchmark_stdout(path, layout, name)
        except ValueError:
            return parse_benchmark_benchfile(path, layout, name)
    raise ValueError(f"unsupported benchmark input format: {input_format}")


def fnv1a_update_u64(hash_value: int, value: int) -> int:
    """Update an FNV-1a hash with one unsigned 64-bit little-endian value."""
    for byte in value.to_bytes(8, "little", signed=False):
        hash_value ^= byte
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return hash_value


def output_shape_signature(output_shapes: list[tuple[int, int]]) -> str:
    """Return the runtime output-shape signature for `R` block dimensions."""
    hash_value = 14695981039346656037
    hash_value = fnv1a_update_u64(hash_value, len(output_shapes))
    for rows, cols in output_shapes:
        hash_value = fnv1a_update_u64(hash_value, int(rows))
        hash_value = fnv1a_update_u64(hash_value, int(cols))
    return f"fnv1a64:{hash_value:016x}"


def parse_dmrg_benchfile(path: Path) -> list[dict[str, Any]]:
    """Parse a live DMRG `MP_BENCHFILE` table."""
    header: list[str] = []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#Time"):
            header = [field[1:] if field.startswith("#") else field for field in stripped.split()]
            continue
        if stripped.startswith("#"):
            continue
        if not header:
            continue
        fields = stripped.split()
        if len(fields) < len(header):
            continue
        row: dict[str, Any] = {"source": str(path), "line": line_number}
        for name, value in zip(header, fields):
            row[name] = value
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no live DMRG benchmark table rows")
    return rows


def dmrg_row_float(row: dict[str, Any], key: str) -> float:
    """Return a DMRG benchmark row value as `float`."""
    try:
        return float(row[key])
    except KeyError as exc:
        raise ValueError(f"DMRG benchmark row is missing required column {key}") from exc


def dmrg_row_int(row: dict[str, Any], key: str) -> int:
    """Return a DMRG benchmark row value as `int`."""
    return int(dmrg_row_float(row, key))


def summarize_dmrg_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize a set of live DMRG benchmark rows."""
    solve_sum = sum(dmrg_row_float(row, "SolveS") for row in rows)
    split_sum = sum(dmrg_row_float(row, "SplitS") for row in rows)
    env_sum = sum(dmrg_row_float(row, "EnvS") for row in rows)
    summary: dict[str, float | int] = {
        "rows": len(rows),
        "solve_sum": solve_sum,
        "solve_mean": solve_sum / len(rows),
        "split_sum": split_sum,
        "split_mean": split_sum / len(rows),
        "env_sum": env_sum,
        "env_mean": env_sum / len(rows),
    }
    if "LanczosMatvecS" in rows[0]:
        matvec_sum = sum(dmrg_row_float(row, "LanczosMatvecS") for row in rows)
        matvec_count = sum(dmrg_row_int(row, "LanczosMatvecN") for row in rows)
        summary.update(
            {
                "matvec_sum": matvec_sum,
                "matvec_mean": matvec_sum / len(rows),
                "matvec_per_apply": matvec_sum / matvec_count if matvec_count != 0 else math.nan,
                "matvec_count": matvec_count,
            }
        )
    return summary


def print_dmrg_summary_row(name: str, threshold: int, summary: dict[str, float | int], half_sweep: str = "-") -> None:
    """Print one live DMRG summary row."""
    matvec_sum = summary.get("matvec_sum", math.nan)
    matvec_mean = summary.get("matvec_mean", math.nan)
    matvec_per_apply = summary.get("matvec_per_apply", math.nan)
    matvec_count = summary.get("matvec_count", 0)
    print(
        f"{name} {threshold} {half_sweep} {int(summary['rows'])} "
        f"{float(summary['solve_sum']):.9g} {float(summary['solve_mean']):.9g} "
        f"{float(summary['split_sum']):.9g} {float(summary['split_mean']):.9g} "
        f"{float(summary['env_sum']):.9g} {float(summary['env_mean']):.9g} "
        f"{float(matvec_sum):.9g} {float(matvec_mean):.9g} {float(matvec_per_apply):.9g} {int(matvec_count)}"
    )


def print_dmrg_shape_summary_row(
    name: str, threshold: int, output_blocks: str, output_shape: str, summary: dict[str, float | int]
) -> None:
    """Print one live DMRG summary row grouped by R/A/B/C output shape."""
    matvec_sum = summary.get("matvec_sum", math.nan)
    matvec_mean = summary.get("matvec_mean", math.nan)
    matvec_per_apply = summary.get("matvec_per_apply", math.nan)
    matvec_count = summary.get("matvec_count", 0)
    print(
        f"{name} {threshold} {output_blocks} {output_shape} {int(summary['rows'])} "
        f"{float(summary['solve_sum']):.9g} {float(summary['solve_mean']):.9g} "
        f"{float(summary['split_sum']):.9g} {float(summary['split_mean']):.9g} "
        f"{float(summary['env_sum']):.9g} {float(summary['env_mean']):.9g} "
        f"{float(matvec_sum):.9g} {float(matvec_mean):.9g} {float(matvec_per_apply):.9g} {int(matvec_count)}"
    )


def layout_key(record: dict[str, Any]) -> tuple[int, ...]:
    """Return the output-layout key for grouping repeated measurements."""
    return tuple(int(item) for item in record.get("output_layout", []))


def drop_initial_per_layout(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Drop the first measured rows for each output layout."""
    if count <= 0:
        return records
    seen: dict[tuple[int, ...], int] = {}
    filtered: list[dict[str, Any]] = []
    for record in records:
        key = layout_key(record)
        observed = seen.get(key, 0)
        seen[key] = observed + 1
        if observed >= count:
            filtered.append(record)
    return filtered


def trace_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load trace records and apply command-line filters."""
    records = read_trace(args.trace)
    records = drop_initial_per_layout(records, getattr(args, "drop_first_per_layout", 0))
    if not records:
        raise ValueError("trace filters removed all R/A/B/C records")
    return records


def term_records_for_args(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return records that can supply term-level problem metadata."""
    term_trace = getattr(args, "term_trace", None)
    if term_trace is None:
        return records
    term_records = read_trace(term_trace)
    if not term_records:
        raise ValueError(f"{term_trace} contains no R/A/B/C trace records")
    return term_records


def trace_problem(records: list[dict[str, Any]], term_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Find the static term problem in either timing records or a companion term trace."""
    for record in term_records if term_records is not None else records:
        if "terms" in record:
            return term_problem(record)
    raise ValueError("trace contains no term metadata; rerun with RABC_TRACE_TERMS=1 or pass --term-trace")


def group_by_layout(records: list[dict[str, Any]]) -> dict[tuple[int, ...], list[dict[str, Any]]]:
    """Group trace records by output layout."""
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(layout_key(record), []).append(record)
    return grouped


def layout_string(layout: list[int]) -> str:
    """Format a layout as the manual placement environment value."""
    return ",".join(str(device) for device in layout)


def canonicalize_device_labels(layout: list[int]) -> list[int]:
    """Relabel devices by first occurrence to pick one representative of global permutations."""
    mapping: dict[int, int] = {}
    next_device = 0
    canonical: list[int] = []
    for device in layout:
        if device not in mapping:
            mapping[device] = next_device
            next_device += 1
        canonical.append(mapping[device])
    return canonical


def compact_layout_string(layout: list[int]) -> str:
    """Format a layout for human-readable summaries without printing every block."""
    if not layout:
        return "blocks=0;counts=;segments=0"

    counts: dict[int, int] = {}
    segments: list[tuple[int, int, int]] = []
    start = 0
    previous = layout[0]
    for index, device in enumerate(layout):
        counts[device] = counts.get(device, 0) + 1
        if device != previous:
            segments.append((previous, start, index))
            start = index
            previous = device
    segments.append((previous, start, len(layout)))

    counts_text = ",".join(f"{device}:{counts[device]}" for device in sorted(counts))
    prefix = f"blocks={len(layout)};counts={counts_text};segments={len(segments)}"
    if len(segments) == 1:
        return f"all{segments[0][0]};{prefix}"
    if len(segments) == 2 and segments[0][0] == 0 and segments[1][0] == 1:
        return f"cut{segments[0][2]};{prefix}"
    return f"{prefix};transitions={len(segments) - 1}"


def maybe_compact_layout(layout: list[int], compact: bool) -> str:
    """Return either the full manual layout or its compact summary."""
    if compact:
        return compact_layout_string(layout)
    return layout_string(layout)


def layout_segments(layout: list[int]) -> list[tuple[int, int, int]]:
    """Return contiguous `(device, begin, end)` segments for a layout."""
    if not layout:
        return []
    segments: list[tuple[int, int, int]] = []
    begin = 0
    previous = layout[0]
    for index, device in enumerate(layout):
        if device != previous:
            segments.append((previous, begin, index))
            begin = index
            previous = device
    segments.append((previous, begin, len(layout)))
    return segments


def is_ordered_contiguous_layout(layout: list[int]) -> bool:
    """Return whether a layout is made from ordered contiguous device ranges."""
    segments = layout_segments(layout)
    devices = [device for device, _, _ in segments]
    return devices == sorted(set(devices))


def layout_shape_features(problem: dict[str, Any], layout: list[int]) -> dict[str, float]:
    """Return global layout features that are not tied to an anonymous critical path."""
    validate_layout(problem, layout)
    segments = layout_segments(layout)
    device_count = int(problem["device_count"])
    block_count = int(problem["block_count"])
    output_bytes = [int(item) for item in problem["output_bytes"]]
    blocks_by_device = [0] * device_count
    bytes_by_device = [0] * device_count
    for block, device in enumerate(layout):
        blocks_by_device[device] += 1
        bytes_by_device[device] += output_bytes[block]

    total_bytes = sum(bytes_by_device)
    return {
        "layout_transitions": float(max(0, len(segments) - 1)),
        "layout_segments": float(len(segments)),
        "active_devices": float(sum(1 for count in blocks_by_device if count != 0)),
        "max_output_block_fraction": (max(blocks_by_device) / block_count) if block_count != 0 else 0.0,
        "max_output_byte_fraction": (max(bytes_by_device) / total_bytes) if total_bytes != 0 else 0.0,
    }


def observed_layout_shape_support(records: list[dict[str, Any]], problem: dict[str, Any]) -> dict[str, Any]:
    """Return discrete and range support for shapes seen in benchmark rows."""
    grouped = group_benchmarks_by_layout(records, problem)
    if not grouped:
        raise ValueError("benchmark rows contain no layouts for shape support")

    features = [layout_shape_features(problem, list(layout)) for layout in grouped]
    return {
        "layout_transitions": {int(row["layout_transitions"]) for row in features},
        "layout_segments": {int(row["layout_segments"]) for row in features},
        "active_devices": {int(row["active_devices"]) for row in features},
        "max_output_block_fraction": (
            min(row["max_output_block_fraction"] for row in features),
            max(row["max_output_block_fraction"] for row in features),
        ),
        "max_output_byte_fraction": (
            min(row["max_output_byte_fraction"] for row in features),
            max(row["max_output_byte_fraction"] for row in features),
        ),
    }


def layout_shape_is_supported(problem: dict[str, Any], layout: list[int], support: dict[str, Any]) -> bool:
    """Return whether a candidate stays inside observed benchmark shape support."""
    features = layout_shape_features(problem, layout)
    for name in ("layout_transitions", "layout_segments", "active_devices"):
        if int(features[name]) not in support[name]:
            return False
    for name in ("max_output_block_fraction", "max_output_byte_fraction"):
        low, high = support[name]
        if not (low <= features[name] <= high):
            return False
    return True


def observed_layout_shape_support_by_segment_count(
    records: list[dict[str, Any]], problem: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    """Return shape-support envelopes keyed by observed layout segment count."""
    grouped = group_benchmarks_by_layout(records, problem)
    records_by_segment_count: dict[int, list[dict[str, Any]]] = {}
    for layout, layout_records in grouped.items():
        shape = layout_shape_features(problem, list(layout))
        records_by_segment_count.setdefault(int(shape["layout_segments"]), []).extend(layout_records)
    return {
        segment_count: observed_layout_shape_support(segment_records, problem)
        for segment_count, segment_records in records_by_segment_count.items()
    }


def layout_shape_is_supported_by_segment_count(
    problem: dict[str, Any], layout: list[int], support_by_segment_count: dict[int, dict[str, Any]]
) -> bool:
    """Return whether a layout is supported by rows with the same segment count."""
    shape = layout_shape_features(problem, layout)
    support = support_by_segment_count.get(int(shape["layout_segments"]))
    if support is None:
        return False
    return layout_shape_is_supported(problem, layout, support)


def benchmark_records_matching_layouts(
    records: list[dict[str, Any]], problem: dict[str, Any], layouts: list[list[int]]
) -> list[dict[str, Any]]:
    """Return benchmark rows whose layouts are in the requested candidate set."""
    candidate_keys = {tuple(layout) for layout in layouts}
    return [record for record in records if tuple(benchmark_layout(record, problem)) in candidate_keys]


def benchmark_feature_names(problem: dict[str, Any], include_graph_features: bool, model: str) -> list[str]:
    """Return the active whole-layout benchmark model feature names."""
    if model == "critical":
        return feature_names(include_graph_features) + BENCHMARK_LAYOUT_FEATURE_NAMES
    if model == "device":
        names = ["intercept"]
        for device in range(int(problem["device_count"])):
            names.extend(f"d{device}_{name}" for name in DEVICE_BENCHMARK_FEATURE_NAMES)
            if include_graph_features:
                names.extend(f"d{device}_{name}" for name in DEVICE_BENCHMARK_GRAPH_FEATURE_NAMES)
        names.extend(BENCHMARK_LAYOUT_FEATURE_NAMES)
        return names
    raise ValueError(f"unsupported benchmark model: {model}")


def ilp_feature_names() -> list[str]:
    """Return feature names expressible by the grouped RABC MILP objective."""
    return feature_names(include_graph_features=False) + ILP_LAYOUT_FEATURE_NAMES


def include_environment_bytes(args: argparse.Namespace) -> bool:
    """Return whether the selected timing objective includes environment staging."""
    return getattr(args, "timing_objective", "steady-state") == "cold-start"


def include_environment_bytes_for_objective(timing_objective: str) -> bool:
    """Return whether a serialized timing objective includes environment staging."""
    if timing_objective == "steady-state":
        return False
    if timing_objective == "cold-start":
        return True
    raise ValueError(f"unsupported timing objective in model file: {timing_objective}")


def feature_vector(features: dict[str, Any], include_env_bytes: bool, include_graph_features: bool) -> list[float]:
    """Convert one per-device feature dictionary into the model vector."""
    values = [1.0]
    for name in feature_names(include_graph_features)[1:]:
        if not include_env_bytes and name in ("a_bytes", "c_bytes"):
            values.append(0.0)
        else:
            values.append(float(features.get(name, 0.0)))
    return values


def gaussian_solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a dense linear system by partial-pivot Gaussian elimination."""
    n = len(rhs)
    a = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1.0e-30:
            continue
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        for item in range(col, n + 1):
            a[col][item] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor == 0.0:
                continue
            for item in range(col, n + 1):
                a[row][item] -= factor * a[col][item]
    return [a[row][n] for row in range(n)]


def fit_linear_samples(
    samples: list[tuple[list[float], float]],
    names: list[str],
    ridge: float,
    empty_message: str,
    prior: dict[str, float] | None = None,
    prior_weight: float = 0.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit a ridge-regularized linear model to already-materialized samples."""
    if not samples:
        raise ValueError(empty_message)

    scales = [1.0] * len(names)
    for column in range(1, len(names)):
        scales[column] = max(abs(vector[column]) for vector, _ in samples) or 1.0

    n = len(names)
    ata = [[0.0 for _ in range(n)] for _ in range(n)]
    aty = [0.0 for _ in range(n)]
    for raw_vector, target in samples:
        vector = [value / scale for value, scale in zip(raw_vector, scales)]
        for row in range(n):
            aty[row] += vector[row] * target
            for col in range(n):
                ata[row][col] += vector[row] * vector[col]
    for diagonal in range(n):
        ata[diagonal][diagonal] += ridge
    if prior is not None and prior_weight > 0.0:
        for diagonal, name in enumerate(names):
            if name not in prior:
                continue
            ata[diagonal][diagonal] += prior_weight
            aty[diagonal] += prior_weight * prior[name] * scales[diagonal]

    scaled_coefficients = gaussian_solve(ata, aty)
    coefficients = {name: scaled_coefficients[index] / scales[index] for index, name in enumerate(names)}

    predictions = [sum(coefficients[name] * raw[index] for index, name in enumerate(names)) for raw, _ in samples]
    targets = [target for _, target in samples]
    mean_target = sum(targets) / len(targets)
    residual_sum = sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions))
    total_sum = sum((target - mean_target) ** 2 for target in targets)
    stats = {
        "samples": float(len(samples)),
        "rmse": math.sqrt(residual_sum / len(samples)),
        "r2": 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0,
    }
    return coefficients, stats


def fit_coefficients(
    records: list[dict[str, Any]],
    ridge: float,
    include_env_bytes: bool,
    include_graph_features: bool,
    problem: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit per-device elapsed time as a linear function of aggregate features."""
    samples: list[tuple[list[float], float]] = []
    for record in records:
        timings = {int(item["device"]): float(item["gpu_s"]) for item in record.get("device_timings", [])}
        for features in device_features_for_record(record, include_graph_features, problem):
            device = int(features["device"])
            if device not in timings:
                continue
            samples.append((feature_vector(features, include_env_bytes, include_graph_features), timings[device]))

    return fit_linear_samples(
        samples, feature_names(include_graph_features), ridge, "trace contains no per-device timing samples"
    )


def summarize(records: list[dict[str, Any]]) -> None:
    """Print a compact summary of observed trace rows."""
    print("rows block_count term_count policy gpu_s layout")
    best: dict[str, Any] | None = None
    for record in records:
        layout = layout_string([int(item) for item in record.get("output_layout", [])])
        gpu_s = float(record.get("gpu_s", float("nan")))
        print(
            f"{record['_line']} {record.get('block_count')} {record.get('term_count')} "
            f"{record.get('policy', '')} {gpu_s:.9g} {layout}"
        )
        if best is None or gpu_s < float(best.get("gpu_s", float("inf"))):
            best = record
    if best is not None:
        print(f"best_observed_line={best['_line']} gpu_s={float(best['gpu_s']):.9g}")


def summarize_grouped(records: list[dict[str, Any]]) -> None:
    """Print per-layout timing aggregates."""
    grouped = group_by_layout(records)

    print("count mean_gpu_s min_gpu_s max_gpu_s first_line layout")
    best_key: tuple[int, ...] | None = None
    best_mean = float("inf")
    for key, rows in grouped.items():
        timings = [float(record.get("gpu_s", float("nan"))) for record in rows]
        mean = sum(timings) / len(timings)
        minimum = min(timings)
        maximum = max(timings)
        if mean < best_mean:
            best_mean = mean
            best_key = key
        print(
            f"{len(rows)} {mean:.9g} {minimum:.9g} {maximum:.9g} "
            f"{rows[0]['_line']} {layout_string(list(key))}"
        )
    if best_key is not None:
        print(f"best_mean_gpu_s={best_mean:.9g} layout={layout_string(list(best_key))}")


def summarize_benchmark_records(records: list[dict[str, Any]], compact_layouts: bool) -> None:
    """Print no-trace replay benchmark timing aggregates."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record.get("name", "")), str(record.get("layout", ""))), []).append(record)

    layout_column = "layout_summary" if compact_layouts else "layout"
    print(f"count mean_matvec_per_apply_s min_matvec_per_apply_s max_matvec_per_apply_s mean_wall_s name {layout_column}")
    best_key: tuple[str, str] | None = None
    best_mean = float("inf")
    for key, rows in grouped.items():
        matvec = [benchmark_matvec_seconds(record) for record in rows]
        wall = [float(record["wall_s"]) for record in rows]
        mean_matvec = sum(matvec) / len(matvec)
        mean_wall = sum(wall) / len(wall)
        if mean_matvec < best_mean:
            best_mean = mean_matvec
            best_key = key
        layout = [int(item) for item in key[1].split(",") if item]
        print(
            f"{len(rows)} {mean_matvec:.9g} {min(matvec):.9g} {max(matvec):.9g} "
            f"{mean_wall:.9g} {key[0]} {maybe_compact_layout(layout, compact_layouts)}"
        )
    if best_key is not None:
        layout = [int(item) for item in best_key[1].split(",") if item]
        print(
            f"best_mean_matvec_per_apply_s={best_mean:.9g} name={best_key[0]} "
            f"{layout_column}={maybe_compact_layout(layout, compact_layouts)}"
        )


def summarize_calibration_records(records: list[dict[str, Any]], compact_layouts: bool) -> None:
    """Print synthetic calibration timing aggregates."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("family", "")), str(record.get("name", "")), str(record.get("layout", "")))
        grouped.setdefault(key, []).append(record)

    layout_column = "layout_summary" if compact_layouts else "layout"
    print(
        "count mean_apply_s min_apply_s max_apply_s active_devices used_devices block_count term_count "
        f"inner_iterations center_bytes family name {layout_column}"
    )
    for key, rows in sorted(grouped.items()):
        mean_apply = [float(record["mean_apply_s"]) for record in rows]
        min_apply = [float(record["min_apply_s"]) for record in rows]
        max_apply = [float(record["max_apply_s"]) for record in rows]
        first = rows[0]
        layout = [int(item) for item in key[2].split(",") if item]
        print(
            f"{len(rows)} {sum(mean_apply) / len(mean_apply):.9g} {min(min_apply):.9g} "
            f"{max(max_apply):.9g} {int(first['active_devices'])} {int(first.get('used_devices', 0))} "
            f"{int(first['block_count'])} "
            f"{int(first['term_count'])} {int(first['inner_iterations'])} "
            f"{int(first.get('center_bytes', 0))} {key[0]} {key[1]} "
            f"{maybe_compact_layout(layout, compact_layouts)}"
        )


def benchmark_layout_rank_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return benchmark timing aggregates grouped by actual layout."""
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(tuple(raw_layout_values(record)), []).append(record)

    rows: list[dict[str, Any]] = []
    for layout, layout_records in grouped.items():
        matvec = [benchmark_matvec_seconds(record) for record in layout_records]
        wall = [float(record["wall_s"]) for record in layout_records]
        names = ",".join(
            sorted({str(record.get("name", "")) for record in layout_records if str(record.get("name", ""))})
        )
        rows.append(
            {
                "layout": layout,
                "count": len(layout_records),
                "mean_matvec_s": sum(matvec) / len(matvec),
                "min_matvec_s": min(matvec),
                "max_matvec_s": max(matvec),
                "mean_wall_s": sum(wall) / len(wall),
                "names": names,
            }
        )
    rows.sort(key=lambda row: (float(row["mean_matvec_s"]), str(row["names"]), tuple(row["layout"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def print_benchmark_layout_rank(
    records: list[dict[str, Any]], compact_layouts: bool, selected_names: list[str]
) -> None:
    """Print replay benchmark timing aggregates grouped by actual layout."""
    rows = benchmark_layout_rank_rows(records)
    layout_column = "layout_summary" if compact_layouts else "layout"
    print(f"rank count mean_matvec_per_apply_s min_matvec_per_apply_s max_matvec_per_apply_s mean_wall_s names {layout_column}")
    for row in rows:
        layout = list(row["layout"])
        print(
            f"{row['rank']} {row['count']} {row['mean_matvec_s']:.9g} {row['min_matvec_s']:.9g} "
            f"{row['max_matvec_s']:.9g} {row['mean_wall_s']:.9g} {row['names']} "
            f"{maybe_compact_layout(layout, compact_layouts)}"
        )

    if rows:
        best = rows[0]
        print(
            f"best_mean_matvec_per_apply_s={best['mean_matvec_s']:.9g} rank=1 names={best['names']} "
            f"{layout_column}={maybe_compact_layout(list(best['layout']), compact_layouts)}"
        )

    if not selected_names:
        return

    rows_by_layout = {tuple(row["layout"]): row for row in rows}
    layouts_by_name: dict[str, set[tuple[int, ...]]] = {name: set() for name in selected_names}
    for record in records:
        name = str(record.get("name", ""))
        if name in layouts_by_name:
            layouts_by_name[name].add(tuple(raw_layout_values(record)))

    best_mean = float(rows[0]["mean_matvec_s"]) if rows else float("nan")
    for name in selected_names:
        layouts = sorted(layouts_by_name[name])
        if not layouts:
            print(f"selected_name={name} found=false")
            continue
        for layout in layouts:
            row = rows_by_layout[layout]
            delta = float(row["mean_matvec_s"]) - best_mean
            ratio = float(row["mean_matvec_s"]) / best_mean if best_mean > 0.0 else float("nan")
            print(
                f"selected_name={name} found=true rank={row['rank']} "
                f"mean_matvec_per_apply_s={row['mean_matvec_s']:.9g} delta_vs_best_s={delta:.9g} "
                f"ratio_vs_best={ratio:.9g} names={row['names']} "
                f"{layout_column}={maybe_compact_layout(list(layout), compact_layouts)}"
            )


def summarize_benchmark_structure(
    records: list[dict[str, Any]], problem: dict[str, Any], sort_key: str, compact_layouts: bool
) -> None:
    """Print no-trace timings with raw structural layout features."""
    grouped = group_benchmarks_by_layout(records, problem)
    rows: list[dict[str, Any]] = []
    for layout_key, layout_records in grouped.items():
        layout = list(layout_key)
        matvec = [benchmark_matvec_seconds(record) for record in layout_records]
        graph = graph_metrics_for_layout(problem, layout)
        device_features = features_for_layout(problem, layout, include_env_bytes=False, include_graph_features=True)
        shape = layout_shape_features(problem, layout)
        names = ",".join(
            sorted({str(record.get("name", "")) for record in layout_records if str(record.get("name", ""))})
        )
        rows.append(
            {
                "count": len(layout_records),
                "mean_matvec_s": sum(matvec) / len(matvec),
                "right_max_gflop": float(graph["right_max_device_flops"]) / 1.0e9,
                "mixed_max_gflop": float(graph["mixed_max_device_flops"]) / 1.0e9,
                "b_peer_mb": int(graph["b_peer_bytes"]) / 1.0e6,
                "b_peer_blocks": int(graph["b_peer_blocks"]),
                "b_cut_terms": int(graph["b_cut_terms"]),
                "max_terms": max(int(row["terms"]) for row in device_features),
                "max_unique_bc": max(int(row["unique_bc"]) for row in device_features),
                "max_output_mb": max(int(row["output_bytes"]) for row in device_features) / 1.0e6,
                "segments": int(shape["layout_segments"]),
                "transitions": int(shape["layout_transitions"]),
                "active_devices": int(shape["active_devices"]),
                "max_output_byte_fraction": float(shape["max_output_byte_fraction"]),
                "right_duplicate_groups": int(graph["right_first"]["duplicate_groups"]),
                "mixed_duplicate_groups": int(graph["mixed"]["duplicate_groups"]),
                "mixed_left_groups": int(graph["mixed_left_groups"]),
                "mixed_right_groups": int(graph["mixed_right_groups"]),
                "name": names,
                "layout": maybe_compact_layout(layout, compact_layouts),
            }
        )

    sort_columns = {
        "observed": "mean_matvec_s",
        "right-flops": "right_max_gflop",
        "mixed-flops": "mixed_max_gflop",
        "peer-bytes": "b_peer_mb",
        "peer-blocks": "b_peer_blocks",
        "segments": "segments",
        "transitions": "transitions",
        "right-duplicates": "right_duplicate_groups",
        "mixed-duplicates": "mixed_duplicate_groups",
        "terms": "max_terms",
        "name": "name",
    }
    column = sort_columns[sort_key]
    rows.sort(key=lambda row: row[column])
    print(
        "count mean_matvec_per_apply_s right_max_gflop mixed_max_gflop b_peer_mb "
        "b_peer_blocks b_cut_terms max_terms max_unique_bc max_output_mb "
        "segments transitions active_devices max_output_byte_fraction "
        "right_duplicate_groups mixed_duplicate_groups mixed_left_groups mixed_right_groups name layout"
    )
    for row in rows:
        print(
            f"{row['count']} {row['mean_matvec_s']:.9g} {row['right_max_gflop']:.9g} "
            f"{row['mixed_max_gflop']:.9g} {row['b_peer_mb']:.9g} {row['b_peer_blocks']} "
            f"{row['b_cut_terms']} {row['max_terms']} {row['max_unique_bc']} {row['max_output_mb']:.9g} "
            f"{row['segments']} {row['transitions']} {row['active_devices']} "
            f"{row['max_output_byte_fraction']:.9g} {row['right_duplicate_groups']} "
            f"{row['mixed_duplicate_groups']} {row['mixed_left_groups']} {row['mixed_right_groups']} "
            f"{row['name']} {row['layout']}"
        )
    if rows:
        best = min(rows, key=lambda row: row["mean_matvec_s"])
        print(
            f"best_mean_matvec_per_apply_s={best['mean_matvec_s']:.9g} name={best['name']} "
            f"right_max_gflop={best['right_max_gflop']:.9g} b_peer_mb={best['b_peer_mb']:.9g}"
        )


def typed_hypergraph_metrics_for_layout(problem: dict[str, Any], layout: list[int]) -> dict[str, Any]:
    """Return weighted split metrics over the typed sparse `f` hyperedges."""
    validate_layout(problem, layout)
    b_fanout: dict[int, dict[str, Any]] = {}
    rb_edges: dict[tuple[int, int], dict[str, Any]] = {}
    right_reuse: dict[tuple[int, int], dict[str, Any]] = {}
    left_reuse: dict[tuple[int, int], dict[str, Any]] = {}

    for term in problem["terms"]:
        r = int(term["r"])
        a = int(term["a"])
        b = int(term["b"])
        c = int(term["c"])
        output_device = int(layout[r])
        input_device = int(layout[b])

        b_row = b_fanout.setdefault(
            b,
            {
                "devices": set(),
                "terms": 0,
                "bytes": int(problem["input_bytes"][b]),
            },
        )
        b_row["devices"].add(output_device)
        b_row["terms"] += 1

        rb_row = rb_edges.setdefault(
            (r, b),
            {
                "cut": False,
                "terms": 0,
                "bytes": int(problem["input_bytes"][b]),
            },
        )
        rb_row["cut"] = bool(rb_row["cut"]) or output_device != input_device
        rb_row["terms"] += 1

        right_key = (b, c)
        right_row = right_reuse.setdefault(
            right_key,
            {
                "devices": set(),
                "terms": 0,
                "first_flops": float(term["bc_flops"]),
                "second_flops": 0.0,
            },
        )
        right_row["devices"].add(output_device)
        right_row["terms"] += 1
        right_row["second_flops"] += float(term["accumulate_flops"])

        left_key = (a, b)
        left_first, left_second = term_left_first_flops(term)
        left_row = left_reuse.setdefault(
            left_key,
            {
                "devices": set(),
                "terms": 0,
                "first_flops": left_first,
                "second_flops": 0.0,
            },
        )
        left_row["devices"].add(output_device)
        left_row["terms"] += 1
        left_row["second_flops"] += left_second

    def split_group_stats(groups: Iterable[dict[str, Any]]) -> dict[str, float]:
        split_groups = 0
        split_terms = 0
        split_first_flops = 0.0
        split_second_flops = 0.0
        for row in groups:
            extra_devices = len(row["devices"]) - 1
            if extra_devices <= 0:
                continue
            split_groups += 1
            split_terms += int(row["terms"])
            split_first_flops += extra_devices * float(row["first_flops"])
            split_second_flops += float(row["second_flops"])
        return {
            "groups": float(split_groups),
            "terms": float(split_terms),
            "first_gflop": split_first_flops / 1.0e9,
            "second_gflop": split_second_flops / 1.0e9,
        }

    b_split_blocks = 0
    b_split_terms = 0
    b_split_bytes = 0
    for row in b_fanout.values():
        extra_devices = len(row["devices"]) - 1
        if extra_devices <= 0:
            continue
        b_split_blocks += 1
        b_split_terms += int(row["terms"])
        b_split_bytes += extra_devices * int(row["bytes"])

    rb_cut_edges = 0
    rb_cut_terms = 0
    rb_cut_bytes = 0
    for row in rb_edges.values():
        if not bool(row["cut"]):
            continue
        rb_cut_edges += 1
        rb_cut_terms += int(row["terms"])
        rb_cut_bytes += int(row["bytes"])

    right = split_group_stats(right_reuse.values())
    left = split_group_stats(left_reuse.values())
    return {
        "b_fanout_split_blocks": b_split_blocks,
        "b_fanout_split_terms": b_split_terms,
        "b_fanout_split_mb": b_split_bytes / 1.0e6,
        "rb_cut_edges": rb_cut_edges,
        "rb_cut_terms": rb_cut_terms,
        "rb_cut_mb": rb_cut_bytes / 1.0e6,
        "right_reuse_split_groups": int(right["groups"]),
        "right_reuse_split_terms": int(right["terms"]),
        "right_reuse_split_first_gflop": right["first_gflop"],
        "right_reuse_split_second_gflop": right["second_gflop"],
        "left_reuse_split_groups": int(left["groups"]),
        "left_reuse_split_terms": int(left["terms"]),
        "left_reuse_split_first_gflop": left["first_gflop"],
        "left_reuse_split_second_gflop": left["second_gflop"],
    }


def layout_structure_row(problem: dict[str, Any], layout: list[int]) -> dict[str, Any]:
    """Return graph-derived structural counters for one layout."""
    graph = graph_metrics_for_layout(problem, layout)
    typed = typed_hypergraph_metrics_for_layout(problem, layout)
    device_features = features_for_layout(problem, layout, include_env_bytes=False, include_graph_features=True)
    shape = layout_shape_features(problem, layout)
    return {
        "right_max_gflop": float(graph["right_max_device_flops"]) / 1.0e9,
        "mixed_max_gflop": float(graph["mixed_max_device_flops"]) / 1.0e9,
        "b_peer_mb": int(graph["b_peer_bytes"]) / 1.0e6,
        "b_peer_blocks": int(graph["b_peer_blocks"]),
        "b_cut_terms": int(graph["b_cut_terms"]),
        "max_terms": max(int(row["terms"]) for row in device_features),
        "max_unique_bc": max(int(row["unique_bc"]) for row in device_features),
        "max_output_mb": max(int(row["output_bytes"]) for row in device_features) / 1.0e6,
        "segments": int(shape["layout_segments"]),
        "transitions": int(shape["layout_transitions"]),
        "max_output_byte_fraction": float(shape["max_output_byte_fraction"]),
        "right_duplicate_groups": int(graph["right_first"]["duplicate_groups"]),
        "mixed_duplicate_groups": int(graph["mixed"]["duplicate_groups"]),
        "mixed_left_groups": int(graph["mixed_left_groups"]),
        "mixed_right_groups": int(graph["mixed_right_groups"]),
        **typed,
    }


def format_layout_structure_columns(row: dict[str, Any]) -> str:
    """Return graph-derived structural counters as table columns."""
    return (
        f"{row['right_max_gflop']:.9g} {row['mixed_max_gflop']:.9g} {row['b_peer_mb']:.9g} "
        f"{row['b_peer_blocks']} {row['b_cut_terms']} {row['max_terms']} {row['max_unique_bc']} "
        f"{row['max_output_mb']:.9g} {row['segments']} {row['transitions']} "
        f"{row['max_output_byte_fraction']:.9g} {row['right_duplicate_groups']} "
        f"{row['mixed_duplicate_groups']} {row['mixed_left_groups']} {row['mixed_right_groups']}"
    )


def structure_feature_names(feature_set: str) -> list[str]:
    """Return the ordered structural feature list for a named score model."""
    try:
        return list(STRUCTURE_SCORE_FEATURE_SETS[feature_set])
    except KeyError as error:
        choices = ",".join(sorted(STRUCTURE_SCORE_FEATURE_SETS))
        raise ValueError(f"unknown structure feature set {feature_set!r}; expected one of {choices}") from error


def layout_structure_vector(problem: dict[str, Any], layout: list[int], feature_names: list[str]) -> list[float]:
    """Return monotonic structural score features for one layout."""
    row = layout_structure_row(problem, layout)
    return [float(row[name]) for name in feature_names]


def structure_row_cache_for_layouts(
    problem: dict[str, Any], layouts: Iterable[tuple[int, ...]]
) -> dict[tuple[int, ...], dict[str, Any]]:
    """Return graph-derived structural counters keyed by layout."""
    return {layout: layout_structure_row(problem, list(layout)) for layout in layouts}


def structure_vector_from_row(row: dict[str, Any], feature_names: list[str]) -> list[float]:
    """Return selected structural score features from a cached row."""
    return [float(row[name]) for name in feature_names]


def fit_monotonic_structure_coefficients(
    records: list[dict[str, Any]],
    problem: dict[str, Any],
    ridge: float,
    iterations: int,
    feature_names: list[str],
    structure_rows: dict[tuple[int, ...], dict[str, Any]] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fit non-negative structural coefficients to benchmark layout means."""
    grouped = group_benchmarks_by_layout(records, problem)
    if not grouped:
        raise ValueError("benchmark data contains no replay matvec samples")

    if structure_rows is None:
        structure_rows = structure_row_cache_for_layouts(problem, grouped)

    raw_x = [structure_vector_from_row(structure_rows[layout], feature_names) for layout in grouped]
    y = [benchmark_layout_mean_matvec_seconds(layout_records) for layout_records in grouped.values()]
    baseline = min(y)
    shifted_y = [max(0.0, value - baseline) for value in y]
    offsets = [min(row[column] for row in raw_x) for column in range(len(feature_names))]
    centered_x = [
        [max(0.0, value - offset) for value, offset in zip(row, offsets)]
        for row in raw_x
    ]
    scales = [
        max(max(row[column] for row in centered_x), 1.0)
        for column in range(len(feature_names))
    ]
    x = [[value / scale for value, scale in zip(row, scales)] for row in centered_x]
    weights = [0.0] * len(feature_names)
    iterations = max(1, iterations)

    # Coordinate descent solves a small non-negative ridge problem without adding
    # a scipy dependency to this analysis script.
    for _ in range(iterations):
        predictions = [sum(weight * value for weight, value in zip(weights, row)) for row in x]
        max_delta = 0.0
        for column in range(len(weights)):
            numerator = 0.0
            denominator = ridge
            for row, target, prediction in zip(x, shifted_y, predictions):
                value = row[column]
                residual_without_column = target - prediction + weights[column] * value
                numerator += value * residual_without_column
                denominator += value * value
            next_weight = max(0.0, numerator / denominator) if denominator > 0.0 else 0.0
            delta = next_weight - weights[column]
            if delta != 0.0:
                for index, row in enumerate(x):
                    predictions[index] += delta * row[column]
                weights[column] = next_weight
                max_delta = max(max_delta, abs(delta))
        if max_delta < 1.0e-12:
            break

    coefficients = {
        name: weight / scale for name, weight, scale in zip(feature_names, weights, scales)
    }
    predicted = [
        baseline + sum(coefficients[name] * value for name, value in zip(feature_names, row))
        for row in centered_x
    ]
    errors = [estimate - observed for estimate, observed in zip(predicted, y)]
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean_y = sum(y) / len(y)
    total = sum((value - mean_y) ** 2 for value in y)
    residual = sum(error * error for error in errors)
    stats = {
        "samples": float(len(y)),
        "baseline": baseline,
        "rmse": rmse,
        "r2": 1.0 - residual / total if total != 0.0 else 1.0,
        "offsets": {name: offset for name, offset in zip(feature_names, offsets)},
    }
    return coefficients, stats


def score_monotonic_structure_layout(
    problem: dict[str, Any],
    layout: list[int],
    baseline: float,
    coefficients: dict[str, float],
    offsets: dict[str, float] | None = None,
    feature_names: list[str] | None = None,
    structure_row: dict[str, Any] | None = None,
) -> float:
    """Return a non-negative structural score in seconds for one layout."""
    feature_names = feature_names if feature_names is not None else list(coefficients)
    structure_row = structure_row if structure_row is not None else layout_structure_row(problem, layout)
    vector = structure_vector_from_row(structure_row, feature_names)
    offsets = offsets if offsets is not None else {}
    return baseline + sum(
        coefficients[name] * max(0.0, value - offsets.get(name, 0.0))
        for name, value in zip(feature_names, vector)
    )


def layout_mean_gpu_seconds(records: list[dict[str, Any]]) -> float:
    """Return the mean max-device GPU time for one layout."""
    timings = [float(record.get("gpu_s", float("nan"))) for record in records]
    return sum(timings) / len(timings)


def benchmark_records_from_paths(paths: list[Path]) -> list[dict[str, Any]]:
    """Read no-trace benchmark records from one or more JSONL files."""
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_benchmark_records(path))
    if not records:
        raise ValueError("benchmark files contained no R/A/B/C replay benchmark records")
    return records


def calibration_records_from_paths(paths: list[Path]) -> list[dict[str, Any]]:
    """Read synthetic calibration records from one or more JSONL files."""
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_calibration_records(path))
    if not records:
        raise ValueError("calibration files contained no R/A/B/C calibration records")
    return records


def calibration_dimension(record: dict[str, Any]) -> int:
    """Infer the square matrix dimension in a synthetic calibration row."""
    block_count = int(record["block_count"])
    center_values = int(record["center_values"])
    if block_count <= 0 or center_values % block_count != 0:
        raise ValueError(f"invalid calibration center size in {record.get('family', '')}/{record.get('name', '')}")
    values_per_block = center_values // block_count
    dimension = math.isqrt(values_per_block)
    if dimension * dimension != values_per_block:
        raise ValueError(
            f"calibration row {record.get('family', '')}/{record.get('name', '')} is not square-block shaped"
        )
    return dimension


def square_term(r: int, a: int, b: int, c: int, coefficient: float, dimension: int) -> dict[str, Any]:
    """Construct one square-block synthetic term with right-first GEMM metadata."""
    flops = 2.0 * float(dimension) * float(dimension) * float(dimension)
    bytes_per_block = dimension * dimension * 8
    return {
        "r": r,
        "a": a,
        "b": b,
        "c": c,
        "coefficient": coefficient,
        "r_rows": dimension,
        "r_cols": dimension,
        "a_rows": dimension,
        "a_cols": dimension,
        "b_rows": dimension,
        "b_cols": dimension,
        "c_rows": dimension,
        "c_cols": dimension,
        "bc_flops": flops,
        "accumulate_flops": flops,
        "intermediate_bytes": bytes_per_block,
    }


def calibration_problem(record: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the synthetic R/A/B/C problem represented by one calibration row."""
    family = str(record["family"])
    block_count = int(record["block_count"])
    dimension = calibration_dimension(record)
    terms: list[dict[str, Any]] = []

    if family in ("gemm_launch", "gemm_throughput"):
        for block in range(block_count):
            terms.append(square_term(block, block, block, block, 1.0, dimension))
    elif family in ("peer_latency", "peer_bandwidth"):
        if block_count % 2 != 0:
            raise ValueError("peer calibration rows must have an even block count")
        pair_count = block_count // 2
        for pair in range(pair_count):
            terms.append(square_term(pair + pair_count, 0, pair, 0, 1.0, dimension))
            terms.append(square_term(pair, 0, pair + pair_count, 0, 1.0, dimension))
    elif family == "source_accumulation":
        for r in range(block_count):
            for b in range(block_count):
                terms.append(square_term(r, 0, b, 0, 1.0, dimension))
    elif family == "output_accumulation":
        if block_count % 2 != 0:
            raise ValueError("output accumulation calibration rows must have an even block count")
        group_count = block_count // 2
        for r in range(block_count):
            local_b = r
            remote_b = r + group_count if r < group_count else r - group_count
            terms.append(square_term(r, 0, local_b, 0, 1.0, dimension))
            terms.append(square_term(r, 0, remote_b, 0, -1.0, dimension))
    else:
        raise ValueError(f"unsupported calibration family: {family}")

    bytes_per_block = dimension * dimension * 8
    layout = raw_layout_values(record)
    device_count = max(int(record.get("active_devices", 1)), max(layout, default=0) + 1)
    return {
        "block_count": block_count,
        "device_count": device_count,
        "terms": terms,
        "input_bytes": [bytes_per_block] * block_count,
        "output_bytes": [bytes_per_block] * block_count,
        "output_shapes": [(dimension, dimension)] * block_count,
        "output_shape_signature": f"square{dimension}x{dimension}x{block_count}",
    }


def calibration_feature_vector(record: dict[str, Any]) -> list[float]:
    """Return the MILP-compatible critical-path feature vector for one calibration row."""
    problem = calibration_problem(record)
    layout = parse_layout(str(record["layout"]), int(problem["block_count"]), int(problem["device_count"]))
    return layout_ilp_feature_vector(problem, layout, include_env_bytes=False)


def compressed_calibration_vector(vector: list[float]) -> list[float]:
    """Project the full ILP feature vector onto identifiable calibration seed features."""
    full_names = feature_names(include_graph_features=False)
    by_name = {name: vector[index] for index, name in enumerate(full_names)}
    return [by_name.get(name, 0.0) for name in CALIBRATION_SEED_FEATURE_NAMES]


def expand_calibration_coefficients(coefficients: dict[str, float]) -> dict[str, float]:
    """Expand sparse calibration coefficients to the full R/A/B/C feature space."""
    full = {name: 0.0 for name in ilp_feature_names()}
    for name, value in coefficients.items():
        full[name] = value
    return full


def calibration_prior_coefficients(coefficients: dict[str, float]) -> dict[str, float]:
    """Return only coefficients directly constrained by synthetic calibration."""
    return {name: coefficients[name] for name in CALIBRATION_SEED_FEATURE_NAMES if name in coefficients}


def fit_calibration_coefficients(
    records: list[dict[str, Any]], ridge: float
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit operation-cost seed coefficients from synthetic calibration rows."""
    samples = [
        (compressed_calibration_vector(calibration_feature_vector(record)), float(record["mean_apply_s"]))
        for record in records
    ]
    coefficients, stats = fit_linear_samples(
        samples,
        CALIBRATION_SEED_FEATURE_NAMES,
        ridge,
        "calibration data contains no synthetic R/A/B/C rows",
    )
    return expand_calibration_coefficients(coefficients), stats


def raw_layout_values(record: dict[str, Any]) -> list[int]:
    """Parse a benchmark layout string without validating the device count."""
    layout: list[int] = []
    for item in str(record.get("layout", "")).split(","):
        stripped = item.strip()
        if stripped:
            layout.append(int(stripped))
    return layout


def benchmark_problem(records: list[dict[str, Any]], term_trace: Path) -> dict[str, Any]:
    """Return a static term problem with device count inferred from benchmarks."""
    problem = trace_problem([], read_trace(term_trace))
    block_count = int(problem["block_count"])
    device_count = int(problem["device_count"])
    for record in records:
        layout = raw_layout_values(record)
        if len(layout) != block_count:
            raise ValueError(f"layout has {len(layout)} blocks, expected {block_count}")
        if layout:
            device_count = max(device_count, max(layout) + 1)

    problem = dict(problem)
    problem["device_count"] = device_count
    return problem


def benchmark_layout(record: dict[str, Any], problem: dict[str, Any]) -> list[int]:
    """Parse the benchmark record's manual placement layout."""
    return parse_layout(str(record.get("layout", "")), int(problem["block_count"]), int(problem["device_count"]))


def benchmark_run_key(record: dict[str, Any], problem: dict[str, Any]) -> tuple[tuple[int, ...], str, str]:
    """Return the measurement-run key used for dropping cold benchmark rows."""
    layout = tuple(benchmark_layout(record, problem))
    source = str(record.get("source", ""))
    name = str(record.get("name", ""))
    return (layout, source, name)


def drop_initial_per_benchmark_run(
    records: list[dict[str, Any]], problem: dict[str, Any], count: int
) -> list[dict[str, Any]]:
    """Drop the first measured rows for each benchmark source and layout.

    Multiple benchmark JSONL files can contain repeated measurements of the
    same logical layout.  Each source file has its own cold first row, so the
    filter must key by both layout and source rather than by layout alone.
    """
    if count <= 0:
        return records
    seen: dict[tuple[tuple[int, ...], str, str], int] = {}
    filtered: list[dict[str, Any]] = []
    for record in records:
        key = benchmark_run_key(record, problem)
        observed = seen.get(key, 0)
        seen[key] = observed + 1
        if observed >= count:
            filtered.append(record)
    if not filtered:
        raise ValueError("benchmark cold-row filter removed all replay benchmark records")
    return filtered


def group_benchmarks_by_layout(
    records: list[dict[str, Any]], problem: dict[str, Any]
) -> dict[tuple[int, ...], list[dict[str, Any]]]:
    """Group no-trace benchmark records by parsed output layout."""
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(tuple(benchmark_layout(record, problem)), []).append(record)
    return grouped


def observed_benchmark_layouts(records: list[dict[str, Any]], problem: dict[str, Any]) -> list[list[int]]:
    """Return unique layouts measured in no-trace benchmark rows."""
    layouts: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for record in records:
        layout = tuple(benchmark_layout(record, problem))
        if layout not in seen:
            seen.add(layout)
            layouts.append(list(layout))
    return layouts


def filter_benchmark_records_by_layout(
    records: list[dict[str, Any]], problem: dict[str, Any], layout_filter: str
) -> list[dict[str, Any]]:
    """Return benchmark records matching the requested layout class."""
    if layout_filter == "all":
        return records
    if layout_filter != "contiguous":
        raise ValueError(f"unsupported benchmark layout filter: {layout_filter}")

    filtered = [
        record for record in records if is_ordered_contiguous_layout(benchmark_layout(record, problem))
    ]
    if not filtered:
        raise ValueError("layout-filter=contiguous removed all benchmark records")
    return filtered


def benchmark_layout_mean_matvec_seconds(records: list[dict[str, Any]]) -> float:
    """Return the mean resident matvec time for one no-trace benchmark layout."""
    timings = [benchmark_matvec_seconds(record) for record in records]
    return sum(timings) / len(timings)


def layout_critical_path_vector(
    problem: dict[str, Any],
    layout: list[int],
    include_env_bytes: bool,
    include_graph_features: bool,
) -> list[float]:
    """Reduce per-device layout features to one critical-path benchmark vector."""
    device_vectors = [
        feature_vector(features, include_env_bytes, include_graph_features)
        for features in features_for_layout(problem, layout, include_env_bytes, include_graph_features)
    ]
    if not device_vectors:
        raise ValueError("layout produced no device feature vectors")
    values = [1.0]
    for column in range(1, len(device_vectors[0])):
        values.append(max(vector[column] for vector in device_vectors))
    shape = layout_shape_features(problem, layout)
    values.extend(shape[name] for name in BENCHMARK_LAYOUT_FEATURE_NAMES)
    return values


def layout_device_aware_vector(
    problem: dict[str, Any],
    layout: list[int],
    include_env_bytes: bool,
    include_graph_features: bool,
) -> list[float]:
    """Return a device-identity-aware whole-layout feature vector."""
    if include_env_bytes:
        raise ValueError("device-aware benchmark model currently supports steady-state timing only")
    validate_layout(problem, layout)
    device_count = int(problem["device_count"])
    features_by_device = {
        int(row["device"]): row for row in features_for_layout(problem, layout, False, False)
    }
    graph_by_device = {int(row["device"]): row for row in graph_metrics_for_layout(problem, layout)["devices"]}

    values = [1.0]
    for device in range(device_count):
        features = features_by_device[device]
        graph = graph_by_device[device]
        values.extend(
            [
                float(graph["right_first_flops"]) + float(graph["right_second_flops"]),
                float(features["b_peer_bytes"]),
                float(features["terms"]),
                float(features["unique_bc"]),
                float(features["output_bytes"]),
            ]
        )
        if include_graph_features:
            values.extend(float(graph[name]) for name in DEVICE_BENCHMARK_GRAPH_FEATURE_NAMES)
    shape = layout_shape_features(problem, layout)
    values.extend(shape[name] for name in BENCHMARK_LAYOUT_FEATURE_NAMES)
    return values


def benchmark_layout_vector(
    problem: dict[str, Any],
    layout: list[int],
    include_env_bytes: bool,
    include_graph_features: bool,
    model: str,
) -> list[float]:
    """Return the feature vector for a whole-layout replay benchmark model."""
    cache = problem.setdefault("_benchmark_layout_vector_cache", {})
    key = (tuple(layout), bool(include_env_bytes), bool(include_graph_features), model)
    if key in cache:
        return list(cache[key])

    if model == "critical":
        vector = layout_critical_path_vector(problem, layout, include_env_bytes, include_graph_features)
    elif model == "device":
        vector = layout_device_aware_vector(problem, layout, include_env_bytes, include_graph_features)
    else:
        raise ValueError(f"unsupported benchmark model: {model}")
    cache[key] = tuple(vector)
    return vector


def fit_benchmark_coefficients(
    records: list[dict[str, Any]],
    problem: dict[str, Any],
    ridge: float,
    include_env_bytes: bool,
    include_graph_features: bool,
    model: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit whole-layout replay matvec time from grouped layout features."""
    grouped = group_benchmarks_by_layout(records, problem)
    samples = [
        (
            benchmark_layout_vector(problem, list(layout), include_env_bytes, include_graph_features, model),
            benchmark_layout_mean_matvec_seconds(layout_records),
        )
        for layout, layout_records in grouped.items()
    ]
    return fit_linear_samples(
        samples,
        benchmark_feature_names(problem, include_graph_features, model),
        ridge,
        "benchmark data contains no replay matvec samples",
    )


def layout_ilp_feature_vector(problem: dict[str, Any], layout: list[int], include_env_bytes: bool) -> list[float]:
    """Return the critical-path feature vector expressible by the grouped RABC MILP."""
    device_vectors = [
        feature_vector(features, include_env_bytes, include_graph_features=False)
        for features in features_for_layout(problem, layout, include_env_bytes, include_graph_features=False)
    ]
    if not device_vectors:
        raise ValueError("layout produced no device feature vectors")
    values = [1.0]
    for column in range(1, len(device_vectors[0])):
        values.append(max(vector[column] for vector in device_vectors))
    shape = layout_shape_features(problem, layout)
    values.extend(shape[name] for name in ILP_LAYOUT_FEATURE_NAMES)
    return values


def fit_ilp_benchmark_coefficients(
    records: list[dict[str, Any]],
    problem: dict[str, Any],
    ridge: float,
    include_env_bytes: bool,
    prior: dict[str, float] | None = None,
    prior_weight: float = 0.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit MILP-compatible operation costs from no-trace replay matvec timing."""
    grouped = group_benchmarks_by_layout(records, problem)
    samples = [
        (
            layout_ilp_feature_vector(problem, list(layout), include_env_bytes),
            benchmark_layout_mean_matvec_seconds(layout_records),
        )
        for layout, layout_records in grouped.items()
    ]
    return fit_linear_samples(
        samples,
        ilp_feature_names(),
        ridge,
        "benchmark data contains no replay matvec samples",
        prior,
        prior_weight,
    )


def score_ilp_layout(
    problem: dict[str, Any], layout: list[int], coefficients: dict[str, float], include_env_bytes: bool
) -> tuple[float, list[float]]:
    """Score a layout with the MILP-compatible critical-path feature vector."""
    _, device_scores = score_layout(
        problem, layout, coefficients, include_env_bytes, include_graph_features=False
    )
    shape = layout_shape_features(problem, layout)
    layout_penalty = sum(coefficients.get(name, 0.0) * shape[name] for name in ILP_LAYOUT_FEATURE_NAMES)
    scores = [value + layout_penalty for value in device_scores]
    return max(scores), scores


def score_benchmark_layout(
    problem: dict[str, Any],
    layout: list[int],
    coefficients: dict[str, float],
    include_env_bytes: bool,
    include_graph_features: bool,
    model: str,
) -> tuple[float, list[float]]:
    """Predict no-trace resident matvec time for one candidate layout."""
    vector = benchmark_layout_vector(problem, layout, include_env_bytes, include_graph_features, model)
    names = benchmark_feature_names(problem, include_graph_features, model)
    return sum(coefficients[name] * value for name, value in zip(names, vector)), vector


def candidate_benchmark_layouts(problem: dict[str, Any], records: list[dict[str, Any]], random_count: int) -> list[list[int]]:
    """Generate seed layouts for benchmark-targeted local search."""
    block_count = int(problem["block_count"])
    device_count = int(problem["device_count"])
    layouts = observed_benchmark_layouts(records, problem)
    layouts.append(byte_balanced_layout(problem))
    layouts.append([block % device_count for block in range(block_count)])
    layouts.append([(block + 1) % device_count for block in range(block_count)])
    for device in range(device_count):
        layouts.append([device] * block_count)
    rng = random.Random(1)
    for _ in range(random_count):
        layouts.append([rng.randrange(device_count) for _ in range(block_count)])

    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for layout in layouts:
        key = tuple(layout)
        if key not in seen:
            seen.add(key)
            unique.append(layout)
    return unique


def benchmark_local_search(
    problem: dict[str, Any],
    start: list[int],
    passes: int,
    score_layout: Callable[[list[int]], float],
) -> list[int]:
    """Improve a layout using a caller-supplied benchmark-targeted score."""
    device_count = int(problem["device_count"])
    layout = start[:]
    best_score = score_layout(layout)
    for _ in range(passes):
        improved = False
        for block in range(len(layout)):
            original = layout[block]
            best_device = original
            for device in range(device_count):
                if device == original:
                    continue
                trial = layout[:]
                trial[block] = device
                trial_score = score_layout(trial)
                if trial_score < best_score:
                    best_score = trial_score
                    best_device = device
            if best_device != original:
                layout[block] = best_device
                improved = True
        if not improved:
            break
    return layout


def leave_one_benchmark_layout_out(
    records: list[dict[str, Any]],
    problem: dict[str, Any],
    ridge: float,
    clamp_negative: bool,
    include_env_bytes: bool,
    include_graph_features: bool,
    model: str,
) -> list[dict[str, Any]]:
    """Predict each benchmarked layout from a fit over all other layouts."""
    grouped = group_benchmarks_by_layout(records, problem)
    if len(grouped) < 2:
        raise ValueError("benchmark validation requires at least two distinct layouts")

    rows: list[dict[str, Any]] = []
    for key, held_out in grouped.items():
        train = [record for other_key, group in grouped.items() if other_key != key for record in group]
        coefficients, _ = fit_benchmark_coefficients(
            train, problem, ridge, include_env_bytes, include_graph_features, model
        )
        if clamp_negative:
            coefficients = clamp_negative_coefficients(coefficients)
        predicted, _ = score_benchmark_layout(
            problem, list(key), coefficients, include_env_bytes, include_graph_features, model
        )
        observed = benchmark_layout_mean_matvec_seconds(held_out)
        names = sorted({str(record.get("name", "")) for record in held_out if str(record.get("name", ""))})
        rows.append(
            {
                "layout": key,
                "first_line": held_out[0]["_line"],
                "count": len(held_out),
                "observed": observed,
                "predicted": predicted,
                "error": predicted - observed,
                "names": ",".join(names),
            }
        )
    rows.sort(key=lambda row: row["observed"])
    return rows


def leave_one_monotonic_structure_layout_out(
    records: list[dict[str, Any]],
    problem: dict[str, Any],
    ridge: float,
    iterations: int,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Predict each benchmarked layout from monotonic structural counters."""
    grouped = group_benchmarks_by_layout(records, problem)
    if len(grouped) < 2:
        raise ValueError("benchmark validation requires at least two distinct layouts")

    structure_rows = structure_row_cache_for_layouts(problem, grouped)
    rows: list[dict[str, Any]] = []
    for key, held_out in grouped.items():
        train = [record for other_key, group in grouped.items() if other_key != key for record in group]
        coefficients, stats = fit_monotonic_structure_coefficients(
            train, problem, ridge, iterations, feature_names, structure_rows
        )
        predicted = score_monotonic_structure_layout(
            problem,
            list(key),
            stats["baseline"],
            coefficients,
            stats["offsets"],
            feature_names,
            structure_rows[key],
        )
        observed = benchmark_layout_mean_matvec_seconds(held_out)
        names = sorted({str(record.get("name", "")) for record in held_out if str(record.get("name", ""))})
        rows.append(
            {
                "layout": key,
                "first_line": held_out[0]["_line"],
                "count": len(held_out),
                "observed": observed,
                "predicted": predicted,
                "error": predicted - observed,
                "names": ",".join(names),
            }
        )
    rows.sort(key=lambda row: row["observed"])
    return rows


def print_benchmark_validation(rows: list[dict[str, Any]], compact_layouts: bool) -> None:
    """Print leave-one-layout-out validation for no-trace benchmark timing."""
    stats = validation_stats(rows)
    print(
        f"layouts={int(stats['layouts'])} mae_s={stats['mae']:.9g} rmse_s={stats['rmse']:.9g} "
        f"r2={stats['r2']:.9g} best_observed_line={int(stats['best_observed_line'])} "
        f"best_predicted_line={int(stats['best_predicted_line'])} "
        f"top1_match={str(bool(stats['top1_match'])).lower()}"
    )
    layout_column = "layout_summary" if compact_layouts else "layout"
    print(f"observed_matvec_per_apply_s predicted_matvec_per_apply_s error_s count first_line names {layout_column}")
    for row in rows:
        layout = list(row["layout"])
        print(
            f"{row['observed']:.9g} {row['predicted']:.9g} {row['error']:.9g} "
            f"{row['count']} {row['first_line']} {row['names']} "
            f"{maybe_compact_layout(layout, compact_layouts)}"
        )


def print_tune_best_summaries(candidates: list[dict[str, Any]]) -> None:
    """Print model-selection summaries from ranked validation candidates."""
    if not candidates:
        return

    def print_candidate(label: str, item: dict[str, Any]) -> None:
        stats = item["stats"]
        print(
            f"{label} ridge={item['ridge']:.9g} clamp_negative={str(item['clamp_negative']).lower()} "
            f"r2={stats['r2']:.9g} rmse_s={stats['rmse']:.9g} "
            f"top1_match={str(bool(stats['top1_match'])).lower()}"
        )

    print_candidate("best_top1_first", candidates[0])
    print_candidate("best_rmse", min(candidates, key=lambda item: item["stats"]["rmse"]))
    print_candidate("best_r2", max(candidates, key=lambda item: item["stats"]["r2"]))


def term_left_first_flops(term: dict[str, Any]) -> tuple[float, float]:
    """Return first and second GEMM flops for the `A * B` order."""
    first = 2.0 * float(term["a_rows"]) * float(term["a_cols"]) * float(term["b_cols"])
    second = 2.0 * float(term["a_rows"]) * float(term["b_cols"]) * float(term["c_cols"])
    return first, second


def term_left_first_intermediate_bytes(term: dict[str, Any]) -> int:
    """Return intermediate bytes for the `A * B` order."""
    return int(term["a_rows"]) * int(term["b_cols"]) * 8


def order_stats_for_record(record: dict[str, Any]) -> dict[str, Any]:
    """Compute flop-only left/right order diagnostics for one traced layout."""
    if "terms" not in record:
        raise ValueError("order-summary requires trace rows captured with UNI20_TENSORCONTRACTION_RABC_TRACE_TERMS=1")

    layout = [int(item) for item in record.get("output_layout", [])]
    return order_stats_for_layout(term_problem(record), layout)


def order_stats_for_layout(problem: dict[str, Any], layout: list[int]) -> dict[str, Any]:
    """Compute flop-only left/right order diagnostics for one candidate layout."""
    validate_layout(problem, layout)
    device_count = int(problem["device_count"])
    devices: list[dict[str, Any]] = []
    staged_right: list[set[tuple[int, int]]] = [set() for _ in range(device_count)]
    staged_left: list[set[tuple[int, int]]] = [set() for _ in range(device_count)]
    for device in range(device_count):
        devices.append(
            {
                "device": device,
                "terms": 0,
                "right_first_flops": 0.0,
                "right_second_flops": 0.0,
                "right_intermediate_bytes": 0,
                "left_first_flops": 0.0,
                "left_second_flops": 0.0,
                "left_intermediate_bytes": 0,
                "term_pref_left": 0,
                "term_pref_right": 0,
                "term_pref_equal": 0,
                "b_long_dim_pref_left": 0,
                "b_long_dim_pref_right": 0,
                "b_square": 0,
            }
        )

    for term in problem["terms"]:
        r = int(term["r"])
        device = layout[r]
        row = devices[device]
        row["terms"] += 1

        right_key = (int(term["b"]), int(term["c"]))
        if right_key not in staged_right[device]:
            staged_right[device].add(right_key)
            row["right_first_flops"] += float(term["bc_flops"])
            row["right_intermediate_bytes"] += int(term["intermediate_bytes"])
        row["right_second_flops"] += float(term["accumulate_flops"])

        left_key = (int(term["a"]), int(term["b"]))
        left_first, left_second = term_left_first_flops(term)
        if left_key not in staged_left[device]:
            staged_left[device].add(left_key)
            row["left_first_flops"] += left_first
            row["left_intermediate_bytes"] += term_left_first_intermediate_bytes(term)
        row["left_second_flops"] += left_second

        right_term_flops = float(term["bc_flops"]) + float(term["accumulate_flops"])
        left_term_flops = left_first + left_second
        if left_term_flops < right_term_flops:
            row["term_pref_left"] += 1
        elif right_term_flops < left_term_flops:
            row["term_pref_right"] += 1
        else:
            row["term_pref_equal"] += 1

        b_rows = int(term["b_rows"])
        b_cols = int(term["b_cols"])
        if b_rows > b_cols:
            row["b_long_dim_pref_left"] += 1
        elif b_cols > b_rows:
            row["b_long_dim_pref_right"] += 1
        else:
            row["b_square"] += 1

    for row in devices:
        row["right_unique_first"] = len(staged_right[row["device"]])
        row["left_unique_first"] = len(staged_left[row["device"]])
        row["right_total_flops"] = row["right_first_flops"] + row["right_second_flops"]
        row["left_total_flops"] = row["left_first_flops"] + row["left_second_flops"]

    return {
        "layout": layout,
        "devices": devices,
        "right_total_flops": sum(float(row["right_total_flops"]) for row in devices),
        "left_total_flops": sum(float(row["left_total_flops"]) for row in devices),
        "right_max_device_flops": max(float(row["right_total_flops"]) for row in devices),
        "left_max_device_flops": max(float(row["left_total_flops"]) for row in devices),
    }


def block_order_preferences(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize left/right order preference for each `(device, B block)` group."""
    if "terms" not in record:
        raise ValueError("block order preferences require trace rows captured with term details")

    layout = [int(item) for item in record.get("output_layout", [])]
    return block_order_preferences_for_layout(term_problem(record), layout)


def block_order_preferences_for_layout(problem: dict[str, Any], layout: list[int]) -> list[dict[str, Any]]:
    """Summarize left/right order preference for each `(device, B block)` group."""
    validate_layout(problem, layout)
    groups: dict[tuple[int, int], dict[str, Any]] = {}
    staged_right: dict[tuple[int, int], set[int]] = {}
    staged_left: dict[tuple[int, int], set[int]] = {}

    for term in problem["terms"]:
        r = int(term["r"])
        b = int(term["b"])
        device = layout[r]
        key = (device, b)
        row = groups.setdefault(
            key,
            {
                "device": device,
                "b": b,
                "b_rows": int(term["b_rows"]),
                "b_cols": int(term["b_cols"]),
                "terms": 0,
                "right_first_flops": 0.0,
                "right_second_flops": 0.0,
                "left_first_flops": 0.0,
                "left_second_flops": 0.0,
            },
        )
        row["terms"] += 1

        c = int(term["c"])
        if c not in staged_right.setdefault(key, set()):
            staged_right[key].add(c)
            row["right_first_flops"] += float(term["bc_flops"])
        row["right_second_flops"] += float(term["accumulate_flops"])

        a = int(term["a"])
        left_first, left_second = term_left_first_flops(term)
        if a not in staged_left.setdefault(key, set()):
            staged_left[key].add(a)
            row["left_first_flops"] += left_first
        row["left_second_flops"] += left_second

    rows = list(groups.values())
    for row in rows:
        row["right_total_flops"] = row["right_first_flops"] + row["right_second_flops"]
        row["left_total_flops"] = row["left_first_flops"] + row["left_second_flops"]
        if row["left_total_flops"] < row["right_total_flops"]:
            row["preference"] = "left"
        elif row["right_total_flops"] < row["left_total_flops"]:
            row["preference"] = "right"
        else:
            row["preference"] = "equal"
        if row["b_rows"] > row["b_cols"]:
            row["long_dim_heuristic"] = "left"
        elif row["b_cols"] > row["b_rows"]:
            row["long_dim_heuristic"] = "right"
        else:
            row["long_dim_heuristic"] = "equal"
        row["abs_delta_flops"] = abs(row["right_total_flops"] - row["left_total_flops"])
    rows.sort(key=lambda row: (row["device"], -row["abs_delta_flops"], row["b"]))
    return rows


def duplicate_stats(groups_by_device: list[set[tuple[Any, ...]]]) -> dict[str, int]:
    """Summarize duplicated logical first-stage groups across devices."""
    owners: dict[tuple[Any, ...], set[int]] = {}
    for device, groups in enumerate(groups_by_device):
        for group in groups:
            owners.setdefault(group, set()).add(device)
    duplicate_groups = 0
    duplicate_extra = 0
    for devices in owners.values():
        if len(devices) > 1:
            duplicate_groups += 1
            duplicate_extra += len(devices) - 1
    return {
        "unique_global": len(owners),
        "uses": sum(len(groups) for groups in groups_by_device),
        "duplicate_groups": duplicate_groups,
        "duplicate_extra": duplicate_extra,
    }


def duplicate_counts_by_device(groups_by_device: list[set[tuple[Any, ...]]]) -> list[int]:
    """Return per-device counts of groups also used by at least one peer device."""
    owners: dict[tuple[Any, ...], set[int]] = {}
    for device, groups in enumerate(groups_by_device):
        for group in groups:
            owners.setdefault(group, set()).add(device)

    counts = [0] * len(groups_by_device)
    for devices in owners.values():
        if len(devices) <= 1:
            continue
        for device in devices:
            counts[device] += 1
    return counts


def empty_graph_device(device: int) -> dict[str, Any]:
    """Create a per-device graph partition summary row."""
    return {
        "device": device,
        "terms": 0,
        "b_cut_terms": 0,
        "b_peer_blocks": 0,
        "b_peer_bytes": 0,
        "right_first_flops": 0.0,
        "right_second_flops": 0.0,
        "right_intermediate_bytes": 0,
        "left_first_flops": 0.0,
        "left_second_flops": 0.0,
        "left_intermediate_bytes": 0,
        "mixed_first_flops": 0.0,
        "mixed_second_flops": 0.0,
        "mixed_intermediate_bytes": 0,
        "mixed_left_groups": 0,
        "mixed_right_groups": 0,
        "right_duplicate_groups": 0,
        "left_duplicate_groups": 0,
        "mixed_duplicate_groups": 0,
    }


def graph_metrics_for_layout(problem: dict[str, Any], layout: list[int]) -> dict[str, Any]:
    """Compute graph cut and first-stage reuse metrics for one center-block layout."""
    validate_layout(problem, layout)
    device_count = int(problem["device_count"])
    devices = [empty_graph_device(device) for device in range(device_count)]
    b_peer_blocks: list[set[int]] = [set() for _ in range(device_count)]
    right_first_groups: list[set[tuple[Any, ...]]] = [set() for _ in range(device_count)]
    left_first_groups: list[set[tuple[Any, ...]]] = [set() for _ in range(device_count)]
    mixed_first_groups: list[set[tuple[Any, ...]]] = [set() for _ in range(device_count)]

    order_rows = block_order_preferences_for_layout(problem, layout)
    mixed_order = {
        (int(row["device"]), int(row["b"])): "left" if row["preference"] == "left" else "right"
        for row in order_rows
    }

    for row in order_rows:
        if row["preference"] == "left":
            devices[int(row["device"])]["mixed_left_groups"] += 1
        else:
            devices[int(row["device"])]["mixed_right_groups"] += 1

    for term in problem["terms"]:
        r = int(term["r"])
        a = int(term["a"])
        b = int(term["b"])
        c = int(term["c"])
        device = layout[r]
        row = devices[device]
        row["terms"] += 1
        if layout[b] != device:
            row["b_cut_terms"] += 1
            b_peer_blocks[device].add(b)

        right_key = ("right", b, c)
        if right_key not in right_first_groups[device]:
            right_first_groups[device].add(right_key)
            row["right_first_flops"] += float(term["bc_flops"])
            row["right_intermediate_bytes"] += int(term["intermediate_bytes"])
        row["right_second_flops"] += float(term["accumulate_flops"])

        left_key = ("left", a, b)
        left_first, left_second = term_left_first_flops(term)
        if left_key not in left_first_groups[device]:
            left_first_groups[device].add(left_key)
            row["left_first_flops"] += left_first
            row["left_intermediate_bytes"] += term_left_first_intermediate_bytes(term)
        row["left_second_flops"] += left_second

        order = mixed_order[(device, b)]
        if order == "left":
            if left_key not in mixed_first_groups[device]:
                mixed_first_groups[device].add(left_key)
                row["mixed_first_flops"] += left_first
                row["mixed_intermediate_bytes"] += term_left_first_intermediate_bytes(term)
            row["mixed_second_flops"] += left_second
        else:
            if right_key not in mixed_first_groups[device]:
                mixed_first_groups[device].add(right_key)
                row["mixed_first_flops"] += float(term["bc_flops"])
                row["mixed_intermediate_bytes"] += int(term["intermediate_bytes"])
            row["mixed_second_flops"] += float(term["accumulate_flops"])

    for device, blocks in enumerate(b_peer_blocks):
        devices[device]["b_peer_blocks"] = len(blocks)
        devices[device]["b_peer_bytes"] = sum(int(problem["input_bytes"][block]) for block in blocks)

    right_duplicate_groups = duplicate_counts_by_device(right_first_groups)
    left_duplicate_groups = duplicate_counts_by_device(left_first_groups)
    mixed_duplicate_groups = duplicate_counts_by_device(mixed_first_groups)

    for row in devices:
        device = int(row["device"])
        row["right_first_groups"] = len(right_first_groups[device])
        row["left_first_groups"] = len(left_first_groups[device])
        row["mixed_first_groups"] = len(mixed_first_groups[device])
        row["right_duplicate_groups"] = right_duplicate_groups[device]
        row["left_duplicate_groups"] = left_duplicate_groups[device]
        row["mixed_duplicate_groups"] = mixed_duplicate_groups[device]
        row["right_total_flops"] = row["right_first_flops"] + row["right_second_flops"]
        row["left_total_flops"] = row["left_first_flops"] + row["left_second_flops"]
        row["mixed_total_flops"] = row["mixed_first_flops"] + row["mixed_second_flops"]

    return {
        "layout": layout,
        "devices": devices,
        "b_cut_terms": sum(int(row["b_cut_terms"]) for row in devices),
        "b_peer_blocks": sum(int(row["b_peer_blocks"]) for row in devices),
        "b_peer_bytes": sum(int(row["b_peer_bytes"]) for row in devices),
        "right_first": duplicate_stats(right_first_groups),
        "left_first": duplicate_stats(left_first_groups),
        "mixed": duplicate_stats(mixed_first_groups),
        "right_max_device_flops": max(float(row["right_total_flops"]) for row in devices),
        "left_max_device_flops": max(float(row["left_total_flops"]) for row in devices),
        "mixed_max_device_flops": max(float(row["mixed_total_flops"]) for row in devices),
        "mixed_left_groups": sum(int(row["mixed_left_groups"]) for row in devices),
        "mixed_right_groups": sum(int(row["mixed_right_groups"]) for row in devices),
    }


def sorted_pin_string(pins: set[int]) -> str:
    """Format a set of center-block pins for compact reports."""
    return ",".join(str(pin) for pin in sorted(pins))


def rabc_hypergraph_summary(problem: dict[str, Any]) -> dict[str, Any]:
    """Summarize layout-relevant hyperedges induced by the sparse `f` tensor."""
    b_fanout: dict[int, dict[str, Any]] = {}
    rb_edges: dict[tuple[int, int], dict[str, Any]] = {}
    right_reuse: dict[tuple[int, int], dict[str, Any]] = {}
    left_reuse: dict[tuple[int, int], dict[str, Any]] = {}

    for term in problem["terms"]:
        r = int(term["r"])
        a = int(term["a"])
        b = int(term["b"])
        c = int(term["c"])

        b_row = b_fanout.setdefault(
            b,
            {
                "b": b,
                "b_bytes": int(problem["input_bytes"][b]),
                "outputs": set(),
                "terms": 0,
            },
        )
        b_row["outputs"].add(r)
        b_row["terms"] += 1

        rb_row = rb_edges.setdefault(
            (r, b),
            {
                "r": r,
                "b": b,
                "b_bytes": int(problem["input_bytes"][b]),
                "terms": 0,
            },
        )
        rb_row["terms"] += 1

        right_key = (b, c)
        right_row = right_reuse.setdefault(
            right_key,
            {
                "b": b,
                "c": c,
                "outputs": set(),
                "terms": 0,
                "first_flops": float(term["bc_flops"]),
                "second_flops": 0.0,
                "intermediate_bytes": int(term["intermediate_bytes"]),
            },
        )
        right_row["outputs"].add(r)
        right_row["terms"] += 1
        right_row["second_flops"] += float(term["accumulate_flops"])

        left_key = (a, b)
        left_first, left_second = term_left_first_flops(term)
        left_row = left_reuse.setdefault(
            left_key,
            {
                "a": a,
                "b": b,
                "outputs": set(),
                "terms": 0,
                "first_flops": left_first,
                "second_flops": 0.0,
                "intermediate_bytes": term_left_first_intermediate_bytes(term),
            },
        )
        left_row["outputs"].add(r)
        left_row["terms"] += 1
        left_row["second_flops"] += left_second

    return {
        "block_count": int(problem["block_count"]),
        "term_count": len(problem["terms"]),
        "b_fanout": list(b_fanout.values()),
        "rb_edges": list(rb_edges.values()),
        "right_reuse": list(right_reuse.values()),
        "left_reuse": list(left_reuse.values()),
    }


def print_hypergraph_summary(summary: dict[str, Any], top: int) -> None:
    """Print a compact text summary of `f`-tensor hypergraph connectivity."""
    b_fanout = summary["b_fanout"]
    rb_edges = summary["rb_edges"]
    right_reuse = summary["right_reuse"]
    left_reuse = summary["left_reuse"]
    print(
        "blocks terms b_fanout_edges rb_edges right_reuse_edges left_reuse_edges "
        "max_b_fanout max_right_fanout max_left_fanout"
    )
    print(
        f"{summary['block_count']} {summary['term_count']} {len(b_fanout)} {len(rb_edges)} "
        f"{len(right_reuse)} {len(left_reuse)} "
        f"{max((len(row['outputs']) for row in b_fanout), default=0)} "
        f"{max((len(row['outputs']) for row in right_reuse), default=0)} "
        f"{max((len(row['outputs']) for row in left_reuse), default=0)}"
    )

    def total_flops(row: dict[str, Any]) -> float:
        return float(row.get("first_flops", 0.0)) + float(row.get("second_flops", 0.0))

    print("top_b_fanout b b_bytes terms fanout outputs")
    for row in sorted(b_fanout, key=lambda item: (len(item["outputs"]), item["terms"], item["b_bytes"]), reverse=True)[
        :top
    ]:
        print(
            f"  {row['b']} {row['b_bytes']} {row['terms']} {len(row['outputs'])} "
            f"{sorted_pin_string(row['outputs'])}"
        )

    print("top_right_reuse b c terms fanout first_gflop total_gflop intermediate_bytes outputs")
    for row in sorted(right_reuse, key=lambda item: (total_flops(item), len(item["outputs"])), reverse=True)[:top]:
        print(
            f"  {row['b']} {row['c']} {row['terms']} {len(row['outputs'])} "
            f"{float(row['first_flops']) / 1.0e9:.9g} {total_flops(row) / 1.0e9:.9g} "
            f"{row['intermediate_bytes']} {sorted_pin_string(row['outputs'])}"
        )

    print("top_left_reuse a b terms fanout first_gflop total_gflop intermediate_bytes outputs")
    for row in sorted(left_reuse, key=lambda item: (total_flops(item), len(item["outputs"])), reverse=True)[:top]:
        print(
            f"  {row['a']} {row['b']} {row['terms']} {len(row['outputs'])} "
            f"{float(row['first_flops']) / 1.0e9:.9g} {total_flops(row) / 1.0e9:.9g} "
            f"{row['intermediate_bytes']} {sorted_pin_string(row['outputs'])}"
        )


def term_problem(record: dict[str, Any]) -> dict[str, Any]:
    """Extract the shape-rich term problem needed to score candidate layouts."""
    terms = record.get("terms", [])
    required = {"r_rows", "r_cols", "a_rows", "a_cols", "b_rows", "b_cols", "c_rows", "c_cols"}
    if not terms or any(not required.issubset(term) for term in terms):
        raise ValueError("suggest requires trace rows captured with UNI20_TENSORCONTRACTION_RABC_TRACE_TERMS=1")

    block_count = int(record["block_count"])
    device_count = int(record["device_count"])
    input_bytes = [0] * block_count
    output_bytes = [0] * block_count
    output_shapes = [(0, 0)] * block_count
    for term in terms:
        b = int(term["b"])
        r = int(term["r"])
        if not 0 <= b < block_count or not 0 <= r < block_count:
            raise ValueError("trace term references a center block outside block_count")
        input_bytes[b] = int(term["b_rows"]) * int(term["b_cols"]) * 8
        output_bytes[r] = int(term["r_rows"]) * int(term["r_cols"]) * 8
        output_shapes[r] = (int(term["r_rows"]), int(term["r_cols"]))
    signature = str(record.get("output_shape_signature", ""))
    if not signature:
        signature = output_shape_signature(output_shapes)
    return {
        "block_count": block_count,
        "device_count": device_count,
        "terms": terms,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "output_shapes": output_shapes,
        "output_shape_signature": signature,
    }


def device_features_for_record(
    record: dict[str, Any], include_graph_features: bool, problem: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Return per-device model features, optionally augmented from term graph metrics."""
    devices = [dict(features) for features in record.get("devices", [])]
    if not include_graph_features:
        return devices

    if problem is None:
        problem = term_problem(record)
    layout = [int(item) for item in record.get("output_layout", [])]
    graph_by_device = {int(row["device"]): row for row in graph_metrics_for_layout(problem, layout)["devices"]}
    for features in devices:
        graph = graph_by_device.get(int(features["device"]), {})
        for name in GRAPH_FEATURE_NAMES:
            features[name] = graph.get(name, 0)
    return devices


def validate_layout(problem: dict[str, Any], layout: list[int]) -> None:
    """Validate that a candidate layout matches the traced problem."""
    block_count = int(problem["block_count"])
    device_count = int(problem["device_count"])
    if len(layout) != block_count:
        raise ValueError(f"layout has {len(layout)} blocks, expected {block_count}")
    for device in layout:
        if not 0 <= device < device_count:
            raise ValueError(f"layout references device {device}, expected [0,{device_count})")


def empty_device_features(device: int) -> dict[str, Any]:
    """Create one zeroed per-device feature record."""
    return {
        "device": device,
        "input_blocks": 0,
        "output_blocks": 0,
        "terms": 0,
        "unique_bc": 0,
        "unique_a": 0,
        "unique_b": 0,
        "unique_c": 0,
        "bc_gemms": 0,
        "final_gemms": 0,
        "direct_final_gemms": 0,
        "accumulation_groups": 0,
        "accumulation_terms": 0,
        "source_accumulation_groups": 0,
        "source_accumulation_terms": 0,
        "output_accumulation_groups": 0,
        "output_accumulation_terms": 0,
        "source_axpys": 0,
        "output_axpys": 0,
        "zero_fills": 0,
        "intermediate_matrices": 0,
        "temporary_matrices": 0,
        "temporary_peer_requests": 0,
        "temporary_peer_copies": 0,
        "bc_flops": 0.0,
        "accumulate_flops": 0.0,
        "temporary_accumulate_flops": 0.0,
        "b_local_bytes": 0,
        "b_peer_bytes": 0,
        "temporary_peer_request_bytes": 0,
        "temporary_peer_bytes": 0,
        "a_bytes": 0,
        "c_bytes": 0,
        "output_bytes": 0,
        "intermediate_bytes": 0,
    }


def features_for_layout(
    problem: dict[str, Any], layout: list[int], include_env_bytes: bool, include_graph_features: bool
) -> list[dict[str, Any]]:
    """Compute input-anchored grouped right-first features for a candidate layout."""
    validate_layout(problem, layout)
    device_count = int(problem["device_count"])
    devices = [empty_device_features(device) for device in range(device_count)]
    staged_bc: list[set[tuple[int, int]]] = [set() for _ in range(device_count)]
    staged_a: list[set[int]] = [set() for _ in range(device_count)]
    staged_b: list[set[int]] = [set() for _ in range(device_count)]
    staged_c: list[set[int]] = [set() for _ in range(device_count)]
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    intermediate_bytes: dict[tuple[int, int, int], int] = {}
    migrated_temporaries: set[tuple[str, int, int, int, int]] = set()

    for block, device in enumerate(layout):
        devices[device]["input_blocks"] += 1
        devices[device]["output_blocks"] += 1
        devices[device]["output_bytes"] += problem["output_bytes"][block]

    for term in problem["terms"]:
        r = int(term["r"])
        a = int(term["a"])
        b = int(term["b"])
        c = int(term["c"])
        first_device = layout[b]
        output_device = layout[r]
        first_row = devices[first_device]
        output_row = devices[output_device]
        output_row["terms"] += 1
        groups.setdefault((r, a), []).append(term)

        bc_key = (b, c)
        intermediate_key = (first_device, b, c)
        intermediate_bytes[intermediate_key] = int(term["intermediate_bytes"])
        if bc_key not in staged_bc[first_device]:
            staged_bc[first_device].add(bc_key)
            first_row["bc_gemms"] += 1
            first_row["intermediate_matrices"] += 1
            first_row["bc_flops"] += float(term["bc_flops"])
            first_row["intermediate_bytes"] += int(term["intermediate_bytes"])

        if b not in staged_b[first_device]:
            staged_b[first_device].add(b)
            first_row["b_local_bytes"] += problem["input_bytes"][b]

        if c not in staged_c[first_device]:
            staged_c[first_device].add(c)
            first_row["c_bytes"] += int(term["c_rows"]) * int(term["c_cols"]) * 8

        if a not in staged_a[output_device]:
            staged_a[output_device].add(a)
            output_row["a_bytes"] += int(term["a_rows"]) * int(term["a_cols"]) * 8

    for (r, a), group_terms in groups.items():
        output_device = layout[r]
        output_row = devices[output_device]
        combined: dict[tuple[int, int, int], float] = {}
        for term in group_terms:
            b = int(term["b"])
            c = int(term["c"])
            key = (layout[b], b, c)
            combined[key] = combined.get(key, 0.0) + float(term["coefficient"])
        active_inputs = [(key, coefficient) for key, coefficient in combined.items() if coefficient != 0.0]
        if not active_inputs:
            continue

        output_row["final_gemms"] += 1
        output_row["accumulate_flops"] += float(group_terms[0]["accumulate_flops"])

        source_groups: dict[int, list[tuple[tuple[int, int, int], float]]] = {}
        for key, coefficient in active_inputs:
            source_groups.setdefault(key[0], []).append((key, coefficient))

        partials: list[dict[str, Any]] = []
        for source_device, source_inputs in source_groups.items():
            if len(source_inputs) == 1:
                key, _ = source_inputs[0]
                _, b, c = key
                partials.append(
                    {
                        "device": source_device,
                        "bytes": intermediate_bytes[key],
                        "migration": ("Y", source_device, output_device, b, c),
                    }
                )
                continue

            source_row = devices[source_device]
            source_row["accumulation_groups"] += 1
            source_row["accumulation_terms"] += len(source_inputs)
            source_row["source_accumulation_groups"] += 1
            source_row["source_accumulation_terms"] += len(source_inputs)
            source_row["source_axpys"] += len(source_inputs)
            source_row["zero_fills"] += 1
            source_row["temporary_matrices"] += 1
            source_bytes = 0
            for key, _ in source_inputs:
                source_bytes = intermediate_bytes[key]
                source_row["temporary_accumulate_flops"] += 2.0 * float(source_bytes // 8)
            partials.append(
                {
                    "device": source_device,
                    "bytes": source_bytes,
                    "migration": ("Q", source_device, output_device, r, a),
                }
            )

        if len(partials) == 1:
            output_row["direct_final_gemms"] += 1
        else:
            output_row["accumulation_groups"] += 1
            output_row["accumulation_terms"] += len(partials)
            output_row["output_accumulation_groups"] += 1
            output_row["output_accumulation_terms"] += len(partials)
            output_row["output_axpys"] += len(partials)
            output_row["zero_fills"] += 1
            output_row["temporary_matrices"] += 1
            output_row["temporary_accumulate_flops"] += sum(2.0 * float(partial["bytes"] // 8) for partial in partials)

        for partial in partials:
            if int(partial["device"]) == output_device:
                continue
            output_row["temporary_peer_requests"] += 1
            output_row["temporary_peer_request_bytes"] += int(partial["bytes"])
            migration_key = partial["migration"]
            if migration_key not in migrated_temporaries:
                migrated_temporaries.add(migration_key)
                output_row["temporary_peer_copies"] += 1
                output_row["temporary_peer_bytes"] += int(partial["bytes"])

    if not include_env_bytes:
        for row in devices:
            row["a_bytes"] = 0
            row["c_bytes"] = 0

    for device in range(device_count):
        devices[device]["unique_bc"] = len(staged_bc[device])
        devices[device]["unique_a"] = len(staged_a[device])
        devices[device]["unique_b"] = len(staged_b[device])
        devices[device]["unique_c"] = len(staged_c[device])

    if include_graph_features:
        graph_by_device = {int(row["device"]): row for row in graph_metrics_for_layout(problem, layout)["devices"]}
        for row in devices:
            graph = graph_by_device[int(row["device"])]
            for name in GRAPH_FEATURE_NAMES:
                row[name] = graph[name]
    return devices


def predict_device_seconds(
    features: dict[str, Any], coefficients: dict[str, float], include_env_bytes: bool, include_graph_features: bool
) -> float:
    """Predict elapsed seconds for one device."""
    return sum(
        coefficients[name] * value
        for name, value in zip(
            feature_names(include_graph_features),
            feature_vector(features, include_env_bytes, include_graph_features),
        )
    )


def score_layout(
    problem: dict[str, Any],
    layout: list[int],
    coefficients: dict[str, float],
    include_env_bytes: bool,
    include_graph_features: bool,
) -> tuple[float, list[float]]:
    """Score a layout by the predicted maximum per-device elapsed time."""
    predictions = [
        predict_device_seconds(features, coefficients, include_env_bytes, include_graph_features)
        for features in features_for_layout(problem, layout, include_env_bytes, include_graph_features)
    ]
    return max(predictions), predictions


def byte_balanced_layout(problem: dict[str, Any]) -> list[int]:
    """Construct a contiguous byte-balanced output layout."""
    block_count = int(problem["block_count"])
    device_count = int(problem["device_count"])
    output_bytes = problem["output_bytes"]
    layout = [0] * block_count
    total_bytes = sum(output_bytes)
    begin = 0
    for device in range(device_count):
        if device == device_count - 1:
            end = block_count
        else:
            target = total_bytes * (device + 1) // device_count
            prefix = sum(output_bytes[:begin])
            end = begin
            while end < block_count and (end == begin or prefix + output_bytes[end] <= target):
                prefix += output_bytes[end]
                end += 1
            remaining_blocks = block_count - end
            remaining_devices = device_count - device - 1
            if remaining_blocks < remaining_devices:
                end -= remaining_devices - remaining_blocks
        for block in range(begin, end):
            layout[block] = device
        begin = end
    return layout


def contiguous_range_layouts(block_count: int, device_count: int) -> list[tuple[str, list[int]]]:
    """Generate layouts that assign contiguous block-index ranges to ordered devices."""
    if device_count <= 0:
        raise ValueError("device_count must be positive")
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    if block_count < device_count:
        raise ValueError("block_count must be at least device_count for nonempty contiguous ranges")

    layouts: list[tuple[str, list[int]]] = []

    def visit(device: int, begin: int, cuts: list[int], layout: list[int]) -> None:
        if device == device_count - 1:
            final = layout + [device] * (block_count - begin)
            if device_count == 2:
                name = f"cut{cuts[0]}"
            else:
                name = "cuts_" + "_".join(str(cut) for cut in cuts)
            layouts.append((name, final))
            return

        remaining_devices = device_count - device - 1
        max_end = block_count - remaining_devices
        for end in range(begin + 1, max_end + 1):
            visit(device + 1, end, cuts + [end], layout + [device] * (end - begin))

    visit(0, 0, [], [])
    return layouts


def segmented_alternating_layouts(
    block_count: int, device_count: int, max_segments: int, cut_stride: int, max_layouts: int
) -> list[tuple[str, list[int]]]:
    """Generate alternating two-device segmented layouts with capped enumeration."""
    if device_count != 2:
        raise ValueError("segmented layout search currently supports exactly two devices")
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    if max_segments <= 0:
        raise ValueError("max_segments must be positive")
    if cut_stride <= 0:
        raise ValueError("segment_cut_stride must be positive")
    if max_layouts <= 0:
        raise ValueError("max_segment_layouts must be positive")

    cut_points = list(range(cut_stride, block_count, cut_stride))
    total = 2
    for segment_count in range(2, max_segments + 1):
        total += 2 * math.comb(len(cut_points), segment_count - 1)
    if total > max_layouts:
        raise ValueError(
            "segmented layout search would generate "
            f"{total} layouts; increase --max-segment-layouts or increase --segment-cut-stride"
        )

    layouts: list[tuple[str, list[int]]] = []
    for start_device in range(2):
        layouts.append((f"seg1_start{start_device}", [start_device] * block_count))

    for segment_count in range(2, max_segments + 1):
        for cuts in itertools.combinations(cut_points, segment_count - 1):
            bounds = (0, *cuts, block_count)
            for start_device in range(2):
                layout: list[int] = []
                for segment in range(segment_count):
                    device = (start_device + segment) % 2
                    layout.extend([device] * (bounds[segment + 1] - bounds[segment]))
                name = f"seg{segment_count}_start{start_device}_cuts" + "_".join(str(cut) for cut in cuts)
                layouts.append((name, layout))
    return layouts


def observed_layouts(records: list[dict[str, Any]]) -> list[list[int]]:
    """Return unique output layouts observed in the trace."""
    layouts: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for record in records:
        layout = layout_key(record)
        if layout and layout not in seen:
            seen.add(layout)
            layouts.append(list(layout))
    return layouts


def candidate_layouts(problem: dict[str, Any], records: list[dict[str, Any]], random_count: int) -> list[list[int]]:
    """Generate seed layouts for local search."""
    block_count = int(problem["block_count"])
    device_count = int(problem["device_count"])
    layouts = observed_layouts(records)
    layouts.append(byte_balanced_layout(problem))
    layouts.append([block % device_count for block in range(block_count)])
    layouts.append([(block + 1) % device_count for block in range(block_count)])
    for device in range(device_count):
        layouts.append([device] * block_count)
    rng = random.Random(1)
    for _ in range(random_count):
        layouts.append([rng.randrange(device_count) for _ in range(block_count)])

    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for layout in layouts:
        key = tuple(layout)
        if key not in seen:
            seen.add(key)
            unique.append(layout)
    return unique


def local_search(
    problem: dict[str, Any],
    start: list[int],
    coefficients: dict[str, float],
    passes: int,
    include_env_bytes: bool,
    include_graph_features: bool,
) -> list[int]:
    """Improve a layout with deterministic single-block moves."""
    device_count = int(problem["device_count"])
    layout = start[:]
    best_score, _ = score_layout(problem, layout, coefficients, include_env_bytes, include_graph_features)
    for _ in range(passes):
        improved = False
        for block in range(len(layout)):
            original = layout[block]
            best_device = original
            for device in range(device_count):
                if device == original:
                    continue
                trial = layout[:]
                trial[block] = device
                trial_score, _ = score_layout(problem, trial, coefficients, include_env_bytes, include_graph_features)
                if trial_score < best_score:
                    best_score = trial_score
                    best_device = device
            if best_device != original:
                layout[block] = best_device
                improved = True
        if not improved:
            break
    return layout


class LinearExpr:
    """Small affine expression helper for the R/A/B/C MILP builder."""

    def __init__(self, constant: float = 0.0) -> None:
        self.constant = float(constant)
        self.terms: dict[int, float] = {}

    def add_var(self, index: int, coefficient: float = 1.0) -> None:
        if coefficient == 0.0:
            return
        self.terms[index] = self.terms.get(index, 0.0) + coefficient
        if self.terms[index] == 0.0:
            del self.terms[index]

    def add_expr(self, other: "LinearExpr", scale: float = 1.0) -> None:
        if scale == 0.0:
            return
        self.constant += scale * other.constant
        for index, coefficient in other.terms.items():
            self.add_var(index, scale * coefficient)


def solve_grouped_right_first_ilp(
    problem: dict[str, Any],
    coefficients: dict[str, float],
    include_env_bytes: bool,
    time_limit: float | None,
    mip_rel_gap: float | None,
    max_layout_segments: int | None = None,
    symmetry_break: bool = True,
) -> dict[str, Any]:
    """Solve the grouped input-anchored right-first placement MILP."""
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise RuntimeError("ilp-suggest requires scipy with scipy.optimize.milp") from exc

    block_count = int(problem["block_count"])
    device_count = int(problem["device_count"])
    if block_count <= 0 or device_count <= 0:
        raise ValueError("ILP requires positive block and device counts")
    if max_layout_segments is not None and max_layout_segments <= 0:
        raise ValueError("max_layout_segments must be positive")

    var_names: list[str] = []
    objective: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    integrality: list[int] = []
    constraints: list[tuple[LinearExpr, float, float]] = []

    def add_var(name: str, binary: bool = True, lower: float = 0.0, upper: float = 1.0) -> int:
        index = len(var_names)
        var_names.append(name)
        objective.append(0.0)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        integrality.append(1 if binary else 0)
        return index

    def add_constraint(expr: LinearExpr, lower: float = -math.inf, upper: float = math.inf) -> None:
        constraints.append((expr, lower - expr.constant, upper - expr.constant))

    def var_expr(index: int, coefficient: float = 1.0) -> LinearExpr:
        expr = LinearExpr()
        expr.add_var(index, coefficient)
        return expr

    def add_or(output: int, inputs: list[int]) -> None:
        if not inputs:
            add_constraint(var_expr(output), 0.0, 0.0)
            return
        for item in inputs:
            expr = LinearExpr()
            expr.add_var(item, 1.0)
            expr.add_var(output, -1.0)
            add_constraint(expr, upper=0.0)
        expr = var_expr(output)
        for item in inputs:
            expr.add_var(item, -1.0)
        add_constraint(expr, upper=0.0)

    def add_and2(output: int, left: int, right: int) -> None:
        expr = LinearExpr()
        expr.add_var(output, 1.0)
        expr.add_var(left, -1.0)
        add_constraint(expr, upper=0.0)
        expr = LinearExpr()
        expr.add_var(output, 1.0)
        expr.add_var(right, -1.0)
        add_constraint(expr, upper=0.0)
        expr = LinearExpr(1.0)
        expr.add_var(output, 1.0)
        expr.add_var(left, -1.0)
        expr.add_var(right, -1.0)
        add_constraint(expr, lower=0.0)

    def coefficient(name: str) -> float:
        if not include_env_bytes and name in ("a_bytes", "c_bytes"):
            return 0.0
        return float(coefficients.get(name, 0.0))

    costs = [LinearExpr(coefficient("intercept")) for _ in range(device_count)]

    def add_cost_var(device: int, feature: str, variable: int, amount: float = 1.0) -> None:
        value = coefficient(feature) * amount
        if value != 0.0:
            costs[device].add_var(variable, value)

    x: dict[tuple[int, int], int] = {}
    for block in range(block_count):
        row = LinearExpr()
        for device in range(device_count):
            variable = add_var(f"x[{block},{device}]")
            x[(block, device)] = variable
            row.add_var(variable, 1.0)
        add_constraint(row, 1.0, 1.0)

    layout_transition_cost = coefficient("layout_transitions") + coefficient("layout_segments")
    transition_vars: list[int] = []
    if coefficient("layout_segments") != 0.0:
        for cost in costs:
            cost.constant += coefficient("layout_segments")
    if layout_transition_cost != 0.0 or max_layout_segments is not None:
        for block in range(block_count - 1):
            transition = add_var(f"layout_transition[{block}]")
            transition_vars.append(transition)
            same_device_vars: list[int] = []
            for device in range(device_count):
                same_device = add_var(f"layout_same[{block},{device}]")
                add_and2(same_device, x[(block, device)], x[(block + 1, device)])
                same_device_vars.append(same_device)
            exact_transition = var_expr(transition)
            for same_device in same_device_vars:
                exact_transition.add_var(same_device, 1.0)
            add_constraint(exact_transition, 1.0, 1.0)
            for cost in costs:
                cost.add_var(transition, layout_transition_cost)
    if max_layout_segments is not None:
        transition_sum = LinearExpr()
        for transition in transition_vars:
            transition_sum.add_var(transition, 1.0)
        add_constraint(transition_sum, upper=float(max_layout_segments - 1))

    terms = list(problem["terms"])
    used_b = sorted({int(term["b"]) for term in terms})
    for block in used_b:
        for device in range(device_count):
            owner = x[(block, device)]
            add_cost_var(device, "b_local_bytes", owner, float(problem["input_bytes"][block]))
            add_cost_var(device, "unique_b", owner)

    for block in range(block_count):
        for device in range(device_count):
            owner = x[(block, device)]
            add_cost_var(device, "output_bytes", owner, float(problem["output_bytes"][block]))

    unique_bc: dict[tuple[int, int], dict[str, float]] = {}
    unique_c_by_device: dict[tuple[int, int], set[int]] = {}
    groups: dict[tuple[int, int], dict[str, Any]] = {}
    for term in terms:
        r = int(term["r"])
        a = int(term["a"])
        b = int(term["b"])
        c = int(term["c"])
        bc_key = (b, c)
        unique_bc.setdefault(
            bc_key,
            {
                "bc_flops": float(term["bc_flops"]),
                "intermediate_bytes": float(term["intermediate_bytes"]),
            },
        )
        for device in range(device_count):
            unique_c_by_device.setdefault((device, c), set()).add(x[(b, device)])
            add_cost_var(device, "terms", x[(r, device)])

        group = groups.setdefault(
            (r, a),
            {
                "r": r,
                "a": a,
                "inputs_by_bc": {},
                "accumulate_flops": float(term["accumulate_flops"]),
                "intermediate_bytes": float(term["intermediate_bytes"]),
            },
        )
        group["inputs_by_bc"][bc_key] = group["inputs_by_bc"].get(bc_key, 0.0) + float(term["coefficient"])

    for (b, c), values in unique_bc.items():
        for device in range(device_count):
            owner = x[(b, device)]
            add_cost_var(device, "unique_bc", owner)
            add_cost_var(device, "bc_gemms", owner)
            add_cost_var(device, "intermediate_matrices", owner)
            add_cost_var(device, "bc_flops", owner, values["bc_flops"])
            add_cost_var(device, "intermediate_bytes", owner, values["intermediate_bytes"])

    for (device, c), owners in unique_c_by_device.items():
        variable = add_var(f"c_use[{device},{c}]")
        add_or(variable, sorted(owners))
        add_cost_var(device, "unique_c", variable)
        c_bytes = 0.0
        for term in terms:
            if int(term["c"]) == c:
                c_bytes = float(int(term["c_rows"]) * int(term["c_cols"]) * 8)
                break
        add_cost_var(device, "c_bytes", variable, c_bytes)

    unique_a_by_device: dict[tuple[int, int], set[int]] = {}
    for r, a in groups:
        for device in range(device_count):
            unique_a_by_device.setdefault((device, a), set()).add(x[(r, device)])
    for (device, a), owners in unique_a_by_device.items():
        variable = add_var(f"a_use[{device},{a}]")
        add_or(variable, sorted(owners))
        add_cost_var(device, "unique_a", variable)
        a_bytes = 0.0
        for term in terms:
            if int(term["a"]) == a:
                a_bytes = float(int(term["a_rows"]) * int(term["a_cols"]) * 8)
                break
        add_cost_var(device, "a_bytes", variable, a_bytes)

    raw_migration_terms: dict[tuple[int, int, int, int], list[int]] = {}
    for group_index, group in enumerate(groups.values()):
        r = int(group["r"])
        active_inputs = [
            (b, c) for (b, c), value in sorted(group["inputs_by_bc"].items()) if value != 0.0
        ]
        if not active_inputs:
            continue
        input_count = len(active_inputs)
        bytes_per_partial = float(group["intermediate_bytes"])

        partial_exists: dict[int, int] = {}
        source_multi: dict[int, int] = {}
        source_single_input: dict[tuple[int, int, int], int] = {}

        for device in range(device_count):
            owner_inputs = [x[(b, device)] for b, _ in active_inputs]
            partial = add_var(f"partial[{group_index},{device}]")
            add_or(partial, sorted(set(owner_inputs)))
            partial_exists[device] = partial

            multi = add_var(f"source_multi[{group_index},{device}]")
            source_multi[device] = multi
            count_expr = LinearExpr()
            for owner in owner_inputs:
                count_expr.add_var(owner, 1.0)
            expr = LinearExpr()
            expr.add_expr(count_expr)
            expr.add_var(multi, -float(input_count))
            add_constraint(expr, upper=1.0)
            expr = LinearExpr()
            expr.add_var(multi, 2.0)
            expr.add_expr(count_expr, -1.0)
            add_constraint(expr, upper=0.0)

            single = add_var(f"source_single[{group_index},{device}]")
            expr = var_expr(single)
            expr.add_var(partial, -1.0)
            expr.add_var(multi, 1.0)
            add_constraint(expr, 0.0, 0.0)

            add_cost_var(device, "accumulation_groups", multi)
            add_cost_var(device, "source_accumulation_groups", multi)
            add_cost_var(device, "zero_fills", multi)
            add_cost_var(device, "temporary_matrices", multi)
            for b, c in active_inputs:
                multi_input = add_var(f"source_multi_input[{group_index},{device},{b},{c}]")
                add_and2(multi_input, x[(b, device)], multi)
                add_cost_var(device, "accumulation_terms", multi_input)
                add_cost_var(device, "source_accumulation_terms", multi_input)
                add_cost_var(device, "source_axpys", multi_input)
                add_cost_var(device, "temporary_accumulate_flops", multi_input, 2.0 * bytes_per_partial / 8.0)

                single_input = add_var(f"source_single_input[{group_index},{device},{b},{c}]")
                add_and2(single_input, x[(b, device)], single)
                source_single_input[(device, b, c)] = single_input

        group_multi = add_var(f"group_multi[{group_index}]")
        partial_count = LinearExpr()
        for partial in partial_exists.values():
            partial_count.add_var(partial, 1.0)
        expr = LinearExpr()
        expr.add_expr(partial_count)
        expr.add_var(group_multi, -float(device_count))
        add_constraint(expr, upper=1.0)
        expr = LinearExpr()
        expr.add_var(group_multi, 2.0)
        expr.add_expr(partial_count, -1.0)
        add_constraint(expr, upper=0.0)

        for output_device in range(device_count):
            output_owner = x[(r, output_device)]
            out_multi = add_var(f"output_multi[{group_index},{output_device}]")
            add_and2(out_multi, output_owner, group_multi)
            add_cost_var(output_device, "final_gemms", output_owner)
            add_cost_var(output_device, "accumulate_flops", output_owner, float(group["accumulate_flops"]))
            add_cost_var(output_device, "direct_final_gemms", output_owner)
            add_cost_var(output_device, "direct_final_gemms", out_multi, -1.0)
            add_cost_var(output_device, "accumulation_groups", out_multi)
            add_cost_var(output_device, "output_accumulation_groups", out_multi)
            add_cost_var(output_device, "zero_fills", out_multi)
            add_cost_var(output_device, "temporary_matrices", out_multi)

            for source_device, partial in partial_exists.items():
                output_partial = add_var(f"output_partial[{group_index},{source_device},{output_device}]")
                add_and2(output_partial, partial, out_multi)
                add_cost_var(output_device, "accumulation_terms", output_partial)
                add_cost_var(output_device, "output_accumulation_terms", output_partial)
                add_cost_var(output_device, "output_axpys", output_partial)
                add_cost_var(output_device, "temporary_accumulate_flops", output_partial, 2.0 * bytes_per_partial / 8.0)

                if source_device == output_device:
                    continue
                q_migration = add_var(f"q_migration[{group_index},{source_device},{output_device}]")
                add_and2(q_migration, source_multi[source_device], output_owner)
                add_cost_var(output_device, "temporary_peer_requests", q_migration)
                add_cost_var(output_device, "temporary_peer_copies", q_migration)
                add_cost_var(output_device, "temporary_peer_request_bytes", q_migration, bytes_per_partial)
                add_cost_var(output_device, "temporary_peer_bytes", q_migration, bytes_per_partial)

                for b, c in active_inputs:
                    raw_occurrence = add_var(
                        f"raw_migration_occurrence[{group_index},{source_device},{output_device},{b},{c}]"
                    )
                    add_and2(raw_occurrence, source_single_input[(source_device, b, c)], output_owner)
                    raw_migration_terms.setdefault((source_device, output_device, b, c), []).append(raw_occurrence)

    for (source_device, output_device, b, c), occurrences in raw_migration_terms.items():
        raw_migration = add_var(f"raw_migration[{source_device},{output_device},{b},{c}]")
        add_or(raw_migration, occurrences)
        bytes_per_partial = float(unique_bc[(b, c)]["intermediate_bytes"])
        for occurrence in occurrences:
            add_cost_var(output_device, "temporary_peer_requests", occurrence)
            add_cost_var(output_device, "temporary_peer_request_bytes", occurrence, bytes_per_partial)
        add_cost_var(output_device, "temporary_peer_copies", raw_migration)
        add_cost_var(output_device, "temporary_peer_bytes", raw_migration, bytes_per_partial)

    makespan = add_var("makespan", binary=False, lower=0.0, upper=math.inf)
    objective[makespan] = 1.0
    for cost in costs:
        expr = LinearExpr()
        expr.add_expr(cost)
        expr.add_var(makespan, -1.0)
        add_constraint(expr, upper=0.0)

    matrix = lil_matrix((len(constraints), len(var_names)), dtype=float)
    lower = np.empty(len(constraints), dtype=float)
    upper = np.empty(len(constraints), dtype=float)
    for row, (expr, row_lower, row_upper) in enumerate(constraints):
        lower[row] = row_lower
        upper[row] = row_upper
        for column, value in expr.terms.items():
            matrix[row, column] = value

    options: dict[str, float] = {}
    if time_limit is not None:
        options["time_limit"] = time_limit
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = mip_rel_gap

    result = milp(
        c=np.array(objective, dtype=float),
        integrality=np.array(integrality, dtype=int),
        bounds=Bounds(np.array(lower_bounds, dtype=float), np.array(upper_bounds, dtype=float)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options=options,
    )
    if result.x is None:
        raise RuntimeError(f"MILP failed without a candidate solution: status={result.status} message={result.message}")

    layout: list[int] = []
    for block in range(block_count):
        values = [float(result.x[x[(block, device)]]) for device in range(device_count)]
        layout.append(max(range(device_count), key=lambda device: values[device]))
    if symmetry_break:
        # The current fitted trace model is device-label symmetric.  Relabeling
        # after the solve selects a stable representative without excluding any
        # optimal solution from the MILP feasible set.
        layout = canonicalize_device_labels(layout)
    score, device_scores = score_ilp_layout(problem, layout, coefficients, include_env_bytes)
    return {
        "layout": layout,
        "score": score,
        "device_scores": device_scores,
        "objective": float(result.fun),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "mip_node_count": getattr(result, "mip_node_count", None),
        "mip_dual_bound": getattr(result, "mip_dual_bound", None),
        "mip_gap": getattr(result, "mip_gap", None),
        "variables": len(var_names),
        "constraints": len(constraints),
        "symmetry_break": "canonical_relabel_first_seen" if symmetry_break and device_count > 1 else "none",
    }


def print_fit_for_names(coefficients: dict[str, float], stats: dict[str, float], names: list[str]) -> None:
    """Print fitted model coefficients for an explicit feature list."""
    print(f"samples={int(stats['samples'])} rmse_s={stats['rmse']:.9g} r2={stats['r2']:.9g}")
    for name in names:
        print(f"{name} {coefficients[name]:.17g}")
    for name in ("bc_flops", "accumulate_flops", "temporary_accumulate_flops"):
        if name not in coefficients:
            continue
        coefficient = coefficients[name]
        if coefficient > 0.0:
            print(f"# implied_{name}_gflops={1.0 / (coefficient * 1.0e9):.9g}")
    for name in (
        "b_local_bytes",
        "b_peer_bytes",
        "temporary_peer_request_bytes",
        "temporary_peer_bytes",
        "a_bytes",
        "c_bytes",
        "output_bytes",
        "intermediate_bytes",
    ):
        if name not in coefficients:
            continue
        coefficient = coefficients[name]
        if coefficient > 0.0:
            print(f"# implied_{name}_gbps={1.0 / (coefficient * 1.0e9):.9g}")


def print_fit(coefficients: dict[str, float], stats: dict[str, float], include_graph_features: bool) -> None:
    """Print fitted trace-model coefficients."""
    print_fit_for_names(coefficients, stats, feature_names(include_graph_features))


def clamp_negative_coefficients(coefficients: dict[str, float]) -> dict[str, float]:
    """Return coefficients with negative values clamped to zero."""
    return {name: max(0.0, value) for name, value in coefficients.items()}


def write_model(
    path: Path,
    coefficients: dict[str, float],
    stats: dict[str, float],
    timing_objective: str,
    graph_features: bool,
    ridge: float,
    clamp_negative: bool,
) -> None:
    """Write a fitted R/A/B/C layout model to a JSON file."""
    model = {
        "kind": "rabc_layout_model",
        "version": 1,
        "timing_objective": timing_objective,
        "graph_features": graph_features,
        "ridge": ridge,
        "clamp_negative": clamp_negative,
        "feature_names": feature_names(graph_features),
        "coefficients": coefficients,
        "stats": stats,
    }
    path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")


def read_model(path: Path) -> dict[str, Any]:
    """Read and validate a fitted R/A/B/C layout model."""
    model = json.loads(path.read_text())
    if model.get("kind") != "rabc_layout_model":
        raise ValueError(f"{path} is not an R/A/B/C layout model")
    if int(model.get("version", 0)) != 1:
        raise ValueError(f"{path} has unsupported R/A/B/C layout model version")
    graph_features = bool(model.get("graph_features", False))
    expected_names = feature_names(graph_features)
    actual_names = [str(name) for name in model.get("feature_names", [])]
    if actual_names != expected_names:
        raise ValueError(f"{path} feature names do not match graph_features={str(graph_features).lower()}")
    coefficients = {str(name): float(value) for name, value in model.get("coefficients", {}).items()}
    missing = [name for name in expected_names if name not in coefficients]
    if missing:
        raise ValueError(f"{path} missing coefficients: {','.join(missing)}")
    model["coefficients"] = coefficients
    model["timing_objective"] = str(model.get("timing_objective", "steady-state"))
    include_environment_bytes_for_objective(model["timing_objective"])
    model["graph_features"] = graph_features
    return model


def leave_one_layout_out(
    records: list[dict[str, Any]],
    ridge: float,
    clamp_negative: bool,
    include_env_bytes: bool,
    include_graph_features: bool,
    problem: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Predict each layout from a model fitted on all other layouts."""
    grouped = group_by_layout(records)
    if len(grouped) < 2:
        raise ValueError("leave-one-layout-out validation requires at least two layouts")

    rows: list[dict[str, Any]] = []
    for key, held_out in grouped.items():
        train = [record for other_key, group in grouped.items() if other_key != key for record in group]
        coefficients, _ = fit_coefficients(train, ridge, include_env_bytes, include_graph_features, problem)
        if clamp_negative:
            coefficients = clamp_negative_coefficients(coefficients)
        layout_problem = problem if problem is not None else term_problem(held_out[0])
        predicted, device_predictions = score_layout(
            layout_problem, list(key), coefficients, include_env_bytes, include_graph_features
        )
        observed = layout_mean_gpu_seconds(held_out)
        rows.append(
            {
                "layout": key,
                "first_line": held_out[0]["_line"],
                "count": len(held_out),
                "observed": observed,
                "predicted": predicted,
                "error": predicted - observed,
                "device_predictions": device_predictions,
            }
        )
    rows.sort(key=lambda row: row["observed"])
    return rows


def print_validation(rows: list[dict[str, Any]]) -> None:
    """Print leave-one-layout-out validation details and aggregate errors."""
    stats = validation_stats(rows)
    print(
        f"layouts={int(stats['layouts'])} mae_s={stats['mae']:.9g} rmse_s={stats['rmse']:.9g} "
        f"r2={stats['r2']:.9g} best_observed_line={int(stats['best_observed_line'])} "
        f"best_predicted_line={int(stats['best_predicted_line'])} "
        f"top1_match={str(bool(stats['top1_match'])).lower()}"
    )
    print("observed_gpu_s predicted_gpu_s error_s count first_line layout")
    for row in rows:
        print(
            f"{row['observed']:.9g} {row['predicted']:.9g} {row['error']:.9g} "
            f"{row['count']} {row['first_line']} {layout_string(list(row['layout']))}"
        )


def validation_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute aggregate validation statistics."""
    errors = [float(row["error"]) for row in rows]
    observed = [float(row["observed"]) for row in rows]
    mean_observed = sum(observed) / len(observed)
    residual_sum = sum(error * error for error in errors)
    total_sum = sum((value - mean_observed) ** 2 for value in observed)
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(residual_sum / len(errors))
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    best_observed = min(rows, key=lambda row: row["observed"])
    best_predicted = min(rows, key=lambda row: row["predicted"])
    return {
        "layouts": float(len(rows)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "best_observed_line": float(best_observed["first_line"]),
        "best_predicted_line": float(best_predicted["first_line"]),
        "top1_match": 1.0 if best_observed["layout"] == best_predicted["layout"] else 0.0,
    }


def parse_float_list(text: str) -> list[float]:
    """Parse a comma-separated list of floating-point values."""
    values: list[float] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def parse_layout(text: str, block_count: int, device_count: int) -> list[int]:
    """Parse and validate a comma-separated device layout."""
    layout: list[int] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        layout.append(int(stripped))
    if len(layout) != block_count:
        raise ValueError(f"layout has {len(layout)} blocks, expected {block_count}")
    for device in layout:
        if not 0 <= device < device_count:
            raise ValueError(f"layout references device {device}, expected [0,{device_count})")
    return layout


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit and print a model."""
    records = trace_records(args)
    problem = trace_problem(records, term_records_for_args(args, records)) if args.graph_features else None
    coefficients, stats = fit_coefficients(
        records, args.ridge, include_environment_bytes(args), args.graph_features, problem
    )
    if args.clamp_negative:
        coefficients = clamp_negative_coefficients(coefficients)
    print(f"timing_objective={args.timing_objective}")
    print(f"graph_features={str(args.graph_features).lower()}")
    print_fit(coefficients, stats, args.graph_features)
    if args.output_model is not None:
        write_model(
            args.output_model,
            coefficients,
            stats,
            args.timing_objective,
            args.graph_features,
            args.ridge,
            args.clamp_negative,
        )
        print(f"model_path={args.output_model}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Summarize trace rows."""
    records = trace_records(args)
    if args.group_layouts:
        summarize_grouped(records)
    else:
        summarize(records)
    return 0


def cmd_bench_record(args: argparse.Namespace) -> int:
    """Convert replay benchmark stdout into benchmark JSONL records."""
    records: list[dict[str, Any]] = []
    for path in args.stdout:
        records.extend(parse_benchmark_file(path, args.layout, args.name, args.input_format))

    output = sys.stdout if args.output is None else args.output.open("a" if args.append else "w")
    try:
        for record in records:
            print(json.dumps(record, sort_keys=True), file=output)
    finally:
        if output is not sys.stdout:
            output.close()
    return 0


def cmd_bench_summary(args: argparse.Namespace) -> int:
    """Summarize no-trace replay benchmark JSONL records."""
    records = benchmark_records_from_paths(args.benchmark)
    summarize_benchmark_records(records, args.compact_layouts)
    return 0


def cmd_calibration_summary(args: argparse.Namespace) -> int:
    """Summarize synthetic calibration JSONL records."""
    records = calibration_records_from_paths(args.calibration)
    summarize_calibration_records(records, args.compact_layouts)
    return 0


def cmd_calibration_fit(args: argparse.Namespace) -> int:
    """Fit operation-cost seed coefficients from synthetic calibration rows."""
    records = calibration_records_from_paths(args.calibration)
    coefficients, stats = fit_calibration_coefficients(records, args.ridge)
    if not args.allow_negative:
        coefficients = clamp_negative_coefficients(coefficients)
    print("timing_objective=steady-state")
    print("target=mean_apply_s")
    print("model_source=calibration")
    print("fit_features=" + ",".join(CALIBRATION_SEED_FEATURE_NAMES))
    print_fit(coefficients, stats, include_graph_features=False)
    if args.output_model is not None:
        write_model(
            args.output_model,
            coefficients,
            stats,
            "steady-state",
            graph_features=False,
            ridge=args.ridge,
            clamp_negative=not args.allow_negative,
        )
        print(f"model_path={args.output_model}")
    return 0


def cmd_dmrg_summary(args: argparse.Namespace) -> int:
    """Summarize live DMRG benchmark table rows."""
    thresholds = [int(item) for item in args.min_states.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("--min-states must include at least one threshold")

    print(
        "#Name #MinStates #HalfSweep #Rows #SolveSumS #SolveMeanS #SplitSumS #SplitMeanS "
        "#EnvSumS #EnvMeanS #LanczosMatvecSumS #LanczosMatvecMeanS #LanczosMatvecPerApplyS #LanczosMatvecN"
    )
    for path in args.benchfile:
        rows = parse_dmrg_benchfile(path)
        for threshold in thresholds:
            filtered = [row for row in rows if dmrg_row_int(row, "States") >= threshold]
            if not filtered:
                continue
            print_dmrg_summary_row(path.stem, threshold, summarize_dmrg_rows(filtered))
            if args.half_sweeps:
                by_half_sweep: dict[int, list[dict[str, Any]]] = {}
                for row in filtered:
                    by_half_sweep.setdefault(dmrg_row_int(row, "SweepNum"), []).append(row)
                for half_sweep, grouped in sorted(by_half_sweep.items()):
                    print_dmrg_summary_row(path.stem, threshold, summarize_dmrg_rows(grouped), str(half_sweep))
    if args.shapes:
        print(
            "#ShapeName #MinStates #RabcOutputBlocks #RabcOutputShape #Rows #SolveSumS #SolveMeanS "
            "#SplitSumS #SplitMeanS #EnvSumS #EnvMeanS #LanczosMatvecSumS #LanczosMatvecMeanS "
            "#LanczosMatvecPerApplyS #LanczosMatvecN"
        )
        for path in args.benchfile:
            rows = parse_dmrg_benchfile(path)
            if not rows or "RabcOutputShape" not in rows[0] or "RabcOutputBlocks" not in rows[0]:
                continue
            for threshold in thresholds:
                filtered = [row for row in rows if dmrg_row_int(row, "States") >= threshold]
                by_shape: dict[tuple[str, str], list[dict[str, Any]]] = {}
                for row in filtered:
                    key = (str(row["RabcOutputBlocks"]), str(row["RabcOutputShape"]))
                    by_shape.setdefault(key, []).append(row)
                for (output_blocks, output_shape), grouped in sorted(
                    by_shape.items(), key=lambda item: summarize_dmrg_rows(item[1]).get("matvec_sum", 0.0), reverse=True
                ):
                    print_dmrg_shape_summary_row(
                        path.stem, threshold, output_blocks, output_shape, summarize_dmrg_rows(grouped)
                    )
    return 0


def cmd_bench_rank(args: argparse.Namespace) -> int:
    """Rank no-trace replay benchmark records by actual layout."""
    records = benchmark_records_from_paths(args.benchmark)
    print_benchmark_layout_rank(records, args.compact_layouts, args.selected_name)
    return 0


def cmd_bench_struct_summary(args: argparse.Namespace) -> int:
    """Summarize benchmark rows with static structural layout features."""
    records = benchmark_records_from_paths(args.benchmark)
    problem = benchmark_problem(records, args.term_trace)
    summarize_benchmark_structure(records, problem, args.sort, args.compact_layouts)
    return 0


def cmd_bench_fit(args: argparse.Namespace) -> int:
    """Fit replay benchmark matvec time from static layout features."""
    records = benchmark_records_from_paths(args.benchmark)
    problem = benchmark_problem(records, args.term_trace)
    records = drop_initial_per_benchmark_run(records, problem, args.drop_first_per_run)
    records = filter_benchmark_records_by_layout(records, problem, args.layout_filter)
    include_env_bytes = include_environment_bytes(args)
    coefficients, stats = fit_benchmark_coefficients(
        records, problem, args.ridge, include_env_bytes, args.graph_features, args.model
    )
    if args.clamp_negative:
        coefficients = clamp_negative_coefficients(coefficients)
    print(f"timing_objective={args.timing_objective}")
    print(f"target=matvec_per_apply_s")
    print(f"model={args.model}")
    print(f"reduction={'device_aware_linear' if args.model == 'device' else 'critical_path_max'}")
    print(f"graph_features={str(args.graph_features).lower()}")
    print(f"layout_filter={args.layout_filter}")
    print(f"drop_first_per_run={args.drop_first_per_run}")
    names = benchmark_feature_names(problem, args.graph_features, args.model)
    print_fit_for_names(coefficients, stats, names)
    return 0


def cmd_bench_validate(args: argparse.Namespace) -> int:
    """Validate replay benchmark layout predictions."""
    records = benchmark_records_from_paths(args.benchmark)
    problem = benchmark_problem(records, args.term_trace)
    records = drop_initial_per_benchmark_run(records, problem, args.drop_first_per_run)
    records = filter_benchmark_records_by_layout(records, problem, args.layout_filter)
    if args.candidate_score == "fit":
        rows = leave_one_benchmark_layout_out(
            records,
            problem,
            args.ridge,
            args.clamp_negative,
            include_environment_bytes(args),
            args.graph_features,
            args.model,
        )
    else:
        rows = leave_one_monotonic_structure_layout_out(
            records,
            problem,
            args.structure_ridge,
            args.structure_iterations,
            structure_feature_names(args.structure_feature_set),
        )
    print(f"timing_objective={args.timing_objective}")
    print(f"target=matvec_per_apply_s")
    print(f"model={args.model}")
    print(f"reduction={'device_aware_linear' if args.model == 'device' else 'critical_path_max'}")
    print(f"graph_features={str(args.graph_features).lower()}")
    print(f"layout_filter={args.layout_filter}")
    print(f"drop_first_per_run={args.drop_first_per_run}")
    print(f"candidate_score={args.candidate_score}")
    if args.candidate_score == "monotonic-structure":
        print(f"structure_ridge={args.structure_ridge:.9g}")
        print(f"structure_iterations={args.structure_iterations}")
        print(f"structure_feature_set={args.structure_feature_set}")
    print_benchmark_validation(rows, args.compact_layouts)
    return 0


def cmd_bench_suggest(args: argparse.Namespace) -> int:
    """Fit benchmark matvec timing and suggest a candidate layout."""
    records = benchmark_records_from_paths(args.benchmark)
    problem = benchmark_problem(records, args.term_trace)
    if args.contiguous_only and args.segmented_only:
        raise ValueError("--contiguous-only and --segmented-only are mutually exclusive")
    if args.observed_only and args.segmented_only:
        raise ValueError("--observed-only and --segmented-only are mutually exclusive")
    records = drop_initial_per_benchmark_run(records, problem, args.drop_first_per_run)
    records = filter_benchmark_records_by_layout(records, problem, args.layout_filter)
    include_env_bytes = include_environment_bytes(args)
    coefficients, stats = fit_benchmark_coefficients(
        records, problem, args.ridge, include_env_bytes, args.graph_features, args.model
    )
    if args.clamp_negative:
        coefficients = clamp_negative_coefficients(coefficients)
    structure_coefficients: dict[str, float] = {}
    structure_stats: dict[str, float] = {}
    structure_names: list[str] = []
    if args.candidate_score == "monotonic-structure":
        structure_names = structure_feature_names(args.structure_feature_set)
        structure_coefficients, structure_stats = fit_monotonic_structure_coefficients(
            records, problem, args.structure_ridge, args.structure_iterations, structure_names
        )

    score_structure_rows: dict[tuple[int, ...], dict[str, Any]] = {}

    def score_candidate(layout: list[int]) -> float:
        if args.candidate_score == "fit":
            score, _ = score_benchmark_layout(
                problem, layout, coefficients, include_env_bytes, args.graph_features, args.model
            )
            return score
        key = tuple(layout)
        structure_row = score_structure_rows.get(key)
        if structure_row is None:
            structure_row = layout_structure_row(problem, layout)
            score_structure_rows[key] = structure_row
        return score_monotonic_structure_layout(
            problem,
            layout,
            structure_stats["baseline"],
            structure_coefficients,
            structure_stats["offsets"],
            structure_names,
            structure_row,
        )

    observed_keys = {tuple(layout) for layout in observed_benchmark_layouts(records, problem)}
    block_count = int(problem["block_count"])
    device_count = int(problem["device_count"])
    if block_count >= device_count:
        contiguous_keys = {tuple(layout) for _, layout in contiguous_range_layouts(block_count, device_count)}
    else:
        contiguous_keys = set()
    byte_balanced_key = tuple(byte_balanced_layout(problem))

    if args.contiguous_only:
        seeds = [layout for _, layout in contiguous_range_layouts(block_count, device_count)]
        search_kind = "contiguous"
    elif args.segmented_only:
        seeds = [
            layout
            for _, layout in segmented_alternating_layouts(
                block_count, device_count, args.max_segments, args.segment_cut_stride, args.max_segment_layouts
            )
        ]
        if not args.allow_shape_extrapolation:
            support_records = benchmark_records_matching_layouts(records, problem, seeds)
            if not support_records:
                raise ValueError(
                    "segmented search has no measured rows from the requested segmented candidate family; "
                    "benchmark at least one matching segmented layout or pass --allow-shape-extrapolation"
                )
            shape_support = observed_layout_shape_support_by_segment_count(support_records, problem)
            seeds = [
                layout
                for layout in seeds
                if layout_shape_is_supported_by_segment_count(problem, layout, shape_support)
            ]
            if not seeds:
                raise ValueError(
                    "segmented search produced no layouts inside observed segmented-family shape support; "
                    "benchmark at least one matching segmented layout or pass --allow-shape-extrapolation"
                )
        search_kind = "segmented"
    elif args.observed_only:
        seeds = observed_benchmark_layouts(records, problem)
        search_kind = "observed"
    else:
        seeds = candidate_benchmark_layouts(problem, records, args.random)
        search_kind = "local"

    ranked: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for seed in seeds:
        layout = (
            seed
            if args.observed_only or args.contiguous_only or args.segmented_only
            else benchmark_local_search(problem, seed, args.passes, score_candidate)
        )
        key = tuple(layout)
        if key in seen:
            continue
        seen.add(key)
        score = score_candidate(layout)
        vector = benchmark_layout_vector(problem, layout, include_env_bytes, args.graph_features, args.model)
        ranked.append(
            {
                "layout": layout,
                "score": score,
                "feature_vector": vector,
                "observed": key in observed_keys,
                "contiguous": key in contiguous_keys,
                "byte_balanced": key == byte_balanced_key,
            }
        )

    if not ranked:
        raise ValueError("benchmark suggestion search generated no candidate layouts")
    ranked.sort(key=lambda row: float(row["score"]))
    best = ranked[0]
    best_layout = best["layout"]
    print(f"timing_objective={args.timing_objective}")
    print(f"target=matvec_per_apply_s")
    print(f"model={args.model}")
    print(f"reduction={'device_aware_linear' if args.model == 'device' else 'critical_path_max'}")
    print(f"graph_features={str(args.graph_features).lower()}")
    print(f"layout_filter={args.layout_filter}")
    print(f"drop_first_per_run={args.drop_first_per_run}")
    print(f"candidate_score={args.candidate_score}")
    names = benchmark_feature_names(problem, args.graph_features, args.model)
    print_fit_for_names(coefficients, stats, names)
    if args.candidate_score == "monotonic-structure":
        print(
            f"monotonic_structure_samples={int(structure_stats['samples'])} "
            f"monotonic_structure_baseline_s={structure_stats['baseline']:.9g} "
            f"monotonic_structure_rmse_s={structure_stats['rmse']:.9g} "
            f"monotonic_structure_r2={structure_stats['r2']:.9g}"
        )
        print(f"structure_feature_set={args.structure_feature_set}")
        print("monotonic_structure_coefficients_order=" + ",".join(structure_names))
        print(
            "monotonic_structure_coefficients="
            + ",".join(f"{structure_coefficients[name]:.17g}" for name in structure_names)
        )
        print("monotonic_structure_offsets_order=" + ",".join(structure_names))
        print(
            "monotonic_structure_offsets="
            + ",".join(f"{structure_stats['offsets'][name]:.17g}" for name in structure_names)
        )
    print(f"search={search_kind}")
    print(f"candidate_layouts={len(ranked)}")
    if args.segmented_only:
        print(f"shape_extrapolation={str(bool(args.allow_shape_extrapolation)).lower()}")
    print(f"predicted_matvec_per_apply_s={float(best['score']):.9g}")
    print(f"observed_layout={str(bool(best['observed'])).lower()}")
    print(f"contiguous_layout={str(bool(best['contiguous'])).lower()}")
    print(f"byte_balanced_layout={str(bool(best['byte_balanced'])).lower()}")
    if args.compact_layouts:
        print("layout_summary=" + compact_layout_string(best_layout))
    else:
        print("layout=" + layout_string(best_layout))
    if not best["observed"]:
        print("warning=selected_layout_not_observed_in_benchmark")
    if args.top > 1:
        layout_column = "layout_summary" if args.compact_layouts else "layout"
        print(f"top_rank predicted_matvec_per_apply_s observed contiguous byte_balanced {layout_column}")
        for rank, row in enumerate(ranked[: args.top], start=1):
            print(
                f"{rank} {float(row['score']):.9g} {str(bool(row['observed'])).lower()} "
                f"{str(bool(row['contiguous'])).lower()} {str(bool(row['byte_balanced'])).lower()} "
                f"{maybe_compact_layout(row['layout'], args.compact_layouts)}"
            )
    if args.show_structure:
        print(
            "structure_rank predicted_matvec_per_apply_s right_max_gflop mixed_max_gflop b_peer_mb "
            "b_peer_blocks b_cut_terms max_terms max_unique_bc max_output_mb segments transitions "
            "max_output_byte_fraction right_duplicate_groups mixed_duplicate_groups mixed_left_groups "
            "mixed_right_groups"
        )
        for rank, row in enumerate(ranked[: args.top], start=1):
            key = tuple(row["layout"])
            structure = score_structure_rows.get(key)
            if structure is None:
                structure = layout_structure_row(problem, row["layout"])
                score_structure_rows[key] = structure
            print(f"{rank} {float(row['score']):.9g} {format_layout_structure_columns(structure)}")
        if args.candidate_score == "monotonic-structure":
            print("score_feature_rank predicted_matvec_per_apply_s " + " ".join(structure_names))
            for rank, row in enumerate(ranked[: args.top], start=1):
                key = tuple(row["layout"])
                structure = score_structure_rows.get(key)
                if structure is None:
                    structure = layout_structure_row(problem, row["layout"])
                    score_structure_rows[key] = structure
                values = " ".join(f"{float(structure[name]):.9g}" for name in structure_names)
                print(f"{rank} {float(row['score']):.9g} {values}")
    if args.compact_layouts:
        print("env_layout_omitted=true")
        print("env_layout_note=rerun_without_compact_layouts_for_full_manual_environment_value")
    else:
        print("env UNI20_TENSORCONTRACTION_RABC_PLACEMENT=manual \\")
        print("    UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT=" + layout_string(best_layout))
    return 0


def cmd_bench_tune(args: argparse.Namespace) -> int:
    """Scan ridge values with benchmark leave-one-layout-out validation."""
    records = benchmark_records_from_paths(args.benchmark)
    problem = benchmark_problem(records, args.term_trace)
    records = drop_initial_per_benchmark_run(records, problem, args.drop_first_per_run)
    records = filter_benchmark_records_by_layout(records, problem, args.layout_filter)
    ridges = parse_float_list(args.ridges)
    clamp_modes = [False, True] if args.include_clamped else [False]
    include_env_bytes = include_environment_bytes(args)

    candidates: list[dict[str, Any]] = []
    for ridge in ridges:
        for clamp_negative in clamp_modes:
            rows = leave_one_benchmark_layout_out(
                records, problem, ridge, clamp_negative, include_env_bytes, args.graph_features, args.model
            )
            stats = validation_stats(rows)
            candidates.append({"ridge": ridge, "clamp_negative": clamp_negative, "stats": stats})

    candidates.sort(
        key=lambda item: (item["stats"]["top1_match"], item["stats"]["r2"], -item["stats"]["rmse"]),
        reverse=True,
    )
    print(f"timing_objective={args.timing_objective}")
    print(f"target=matvec_per_apply_s")
    print(f"model={args.model}")
    print(f"reduction={'device_aware_linear' if args.model == 'device' else 'critical_path_max'}")
    print(f"graph_features={str(args.graph_features).lower()}")
    print(f"layout_filter={args.layout_filter}")
    print(f"drop_first_per_run={args.drop_first_per_run}")
    print("ridge clamp_negative layouts mae_s rmse_s r2 top1_match best_observed_line best_predicted_line")
    for item in candidates:
        stats = item["stats"]
        print(
            f"{item['ridge']:.9g} {str(item['clamp_negative']).lower()} {int(stats['layouts'])} "
            f"{stats['mae']:.9g} {stats['rmse']:.9g} {stats['r2']:.9g} "
            f"{str(bool(stats['top1_match'])).lower()} {int(stats['best_observed_line'])} "
            f"{int(stats['best_predicted_line'])}"
        )

    print_tune_best_summaries(candidates)
    return 0


def cmd_order_summary(args: argparse.Namespace) -> int:
    """Summarize left-first versus right-first costs for traced `f` terms."""
    records = trace_records(args)
    problem = trace_problem(records, term_records_for_args(args, records))
    grouped = group_by_layout(records)
    print(
        "first_line count right_total_flops left_total_flops right_max_device_flops left_max_device_flops "
        "layout"
    )
    for key, rows in grouped.items():
        stats = order_stats_for_layout(problem, list(key))
        print(
            f"{rows[0]['_line']} {len(rows)} {stats['right_total_flops']:.9g} {stats['left_total_flops']:.9g} "
            f"{stats['right_max_device_flops']:.9g} {stats['left_max_device_flops']:.9g} "
            f"{layout_string(list(key))}"
        )
        if args.devices:
            print(
                "  device terms right_total_flops left_total_flops right_unique_first left_unique_first "
                "term_pref_left term_pref_right term_pref_equal "
                "b_long_dim_pref_left b_long_dim_pref_right b_square"
            )
            for row in stats["devices"]:
                print(
                    f"  {row['device']} {row['terms']} {row['right_total_flops']:.9g} "
                    f"{row['left_total_flops']:.9g} {row['right_unique_first']} {row['left_unique_first']} "
                    f"{row['term_pref_left']} {row['term_pref_right']} {row['term_pref_equal']} "
                    f"{row['b_long_dim_pref_left']} {row['b_long_dim_pref_right']} {row['b_square']}"
                )
        if args.blocks:
            print(
                "  device b b_rows b_cols terms right_total_flops left_total_flops preference "
                "long_dim_heuristic"
            )
            block_rows = block_order_preferences_for_layout(problem, list(key))
            if args.top_blocks > 0:
                block_rows = block_rows[: args.top_blocks]
            for row in block_rows:
                print(
                    f"  {row['device']} {row['b']} {row['b_rows']} {row['b_cols']} {row['terms']} "
                    f"{row['right_total_flops']:.9g} {row['left_total_flops']:.9g} "
                    f"{row['preference']} {row['long_dim_heuristic']}"
                )
    return 0


def cmd_graph_summary(args: argparse.Namespace) -> int:
    """Summarize graph cuts and first-stage reuse for R/A/B/C term layouts."""
    records = trace_records(args)
    problem = trace_problem(records, term_records_for_args(args, records))
    grouped = group_by_layout(records)

    layouts: list[tuple[str, int, int | str, list[int]]] = []
    if args.layout:
        for index, text in enumerate(args.layout):
            layouts.append(
                (
                    f"manual{index}",
                    0,
                    "manual",
                    parse_layout(text, int(problem["block_count"]), int(problem["device_count"])),
                )
            )
        if args.include_observed:
            for key, rows in grouped.items():
                layouts.append((layout_string(list(key)), len(rows), rows[0]["_line"], list(key)))
    else:
        for key, rows in grouped.items():
            layouts.append((layout_string(list(key)), len(rows), rows[0]["_line"], list(key)))

    print(
        "first_line count b_cut_terms b_peer_blocks b_peer_bytes "
        "right_first_uses right_dup_extra left_first_uses left_dup_extra "
        "mixed_first_uses mixed_dup_extra mixed_left_groups mixed_right_groups "
        "right_max_flops left_max_flops mixed_max_flops layout"
    )
    for layout_name, count, first_line, layout in layouts:
        stats = graph_metrics_for_layout(problem, layout)
        print(
            f"{first_line} {count} {stats['b_cut_terms']} {stats['b_peer_blocks']} {stats['b_peer_bytes']} "
            f"{stats['right_first']['uses']} {stats['right_first']['duplicate_extra']} "
            f"{stats['left_first']['uses']} {stats['left_first']['duplicate_extra']} "
            f"{stats['mixed']['uses']} {stats['mixed']['duplicate_extra']} "
            f"{stats['mixed_left_groups']} {stats['mixed_right_groups']} "
            f"{stats['right_max_device_flops']:.9g} {stats['left_max_device_flops']:.9g} "
            f"{stats['mixed_max_device_flops']:.9g} {layout_name}"
        )
        if args.devices:
            print(
                "  device terms b_cut_terms b_peer_blocks b_peer_bytes "
                "right_first_groups left_first_groups mixed_first_groups "
                "right_duplicate_groups left_duplicate_groups mixed_duplicate_groups "
                "right_total_flops left_total_flops mixed_total_flops "
                "mixed_left_groups mixed_right_groups"
            )
            for row in stats["devices"]:
                print(
                    f"  {row['device']} {row['terms']} {row['b_cut_terms']} {row['b_peer_blocks']} "
                    f"{row['b_peer_bytes']} {row['right_first_groups']} {row['left_first_groups']} "
                    f"{row['mixed_first_groups']} {row['right_duplicate_groups']} {row['left_duplicate_groups']} "
                    f"{row['mixed_duplicate_groups']} {row['right_total_flops']:.9g} "
                    f"{row['left_total_flops']:.9g} {row['mixed_total_flops']:.9g} "
                    f"{row['mixed_left_groups']} {row['mixed_right_groups']}"
                )
    return 0


def cmd_hypergraph_summary(args: argparse.Namespace) -> int:
    """Summarize the layout-relevant hypergraph induced by sparse `f` terms."""
    records = trace_records(args)
    problem = trace_problem(records, term_records_for_args(args, records))
    print_hypergraph_summary(rabc_hypergraph_summary(problem), args.top)
    return 0


def cmd_layouts(args: argparse.Namespace) -> int:
    """Generate simple manual placement layouts."""
    rng = random.Random(args.seed)
    layouts = [
        ("range", [min(args.device_count - 1, block * args.device_count // args.block_count)
                   for block in range(args.block_count)]),
        ("alternating", [block % args.device_count for block in range(args.block_count)]),
    ]
    if args.contiguous_cuts:
        layouts.extend(contiguous_range_layouts(args.block_count, args.device_count))
    if args.segmented_cuts:
        layouts.extend(
            segmented_alternating_layouts(
                args.block_count,
                args.device_count,
                args.max_segments,
                args.segment_cut_stride,
                args.max_segment_layouts,
            )
        )
    for index in range(args.random):
        layouts.append((f"random{index}", [rng.randrange(args.device_count) for _ in range(args.block_count)]))
    for name, layout in layouts:
        print(f"{name} {layout_string(layout)}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run leave-one-layout-out validation."""
    records = trace_records(args)
    problem = trace_problem(records, term_records_for_args(args, records))
    rows = leave_one_layout_out(
        records, args.ridge, args.clamp_negative, include_environment_bytes(args), args.graph_features, problem
    )
    print(f"timing_objective={args.timing_objective}")
    print(f"graph_features={str(args.graph_features).lower()}")
    print_validation(rows)
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """Scan ridge values using leave-one-layout-out validation."""
    records = trace_records(args)
    problem = trace_problem(records, term_records_for_args(args, records))
    ridges = parse_float_list(args.ridges)
    clamp_modes = [False, True] if args.include_clamped else [False]
    include_env_bytes = include_environment_bytes(args)

    candidates: list[dict[str, Any]] = []
    for ridge in ridges:
        for clamp_negative in clamp_modes:
            rows = leave_one_layout_out(records, ridge, clamp_negative, include_env_bytes, args.graph_features, problem)
            stats = validation_stats(rows)
            candidates.append({"ridge": ridge, "clamp_negative": clamp_negative, "stats": stats})

    candidates.sort(key=lambda item: (item["stats"]["top1_match"], item["stats"]["r2"], -item["stats"]["rmse"]),
                    reverse=True)
    print(f"timing_objective={args.timing_objective}")
    print(f"graph_features={str(args.graph_features).lower()}")
    print("ridge clamp_negative layouts mae_s rmse_s r2 top1_match best_observed_line best_predicted_line")
    for item in candidates:
        stats = item["stats"]
        print(
            f"{item['ridge']:.9g} {str(item['clamp_negative']).lower()} {int(stats['layouts'])} "
            f"{stats['mae']:.9g} {stats['rmse']:.9g} {stats['r2']:.9g} "
            f"{str(bool(stats['top1_match'])).lower()} {int(stats['best_observed_line'])} "
            f"{int(stats['best_predicted_line'])}"
        )

    print_tune_best_summaries(candidates)
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    """Fit a model and suggest the best searched layout."""
    records = trace_records(args)
    problem = trace_problem(records, term_records_for_args(args, records))
    model: dict[str, Any] | None = read_model(args.model) if args.model is not None else None
    if model is None:
        timing_objective = args.timing_objective
        graph_features = args.graph_features
        include_env_bytes = include_environment_bytes(args)
        coefficients, stats = fit_coefficients(records, args.ridge, include_env_bytes, graph_features, problem)
        if not args.allow_negative:
            coefficients = clamp_negative_coefficients(coefficients)
        model_source = "fit"
    else:
        timing_objective = str(model["timing_objective"])
        graph_features = bool(model["graph_features"])
        include_env_bytes = include_environment_bytes_for_objective(timing_objective)
        coefficients = dict(model["coefficients"])
        stats = {str(name): float(value) for name, value in model.get("stats", {}).items()}
        model_source = str(args.model)
    observed_keys = {tuple(layout) for layout in observed_layouts(records)}
    if int(problem["block_count"]) >= int(problem["device_count"]):
        contiguous_keys = {
            tuple(layout)
            for _, layout in contiguous_range_layouts(int(problem["block_count"]), int(problem["device_count"]))
        }
    else:
        contiguous_keys = set()
    byte_balanced_key = tuple(byte_balanced_layout(problem))

    if args.contiguous_only:
        seeds = [
            layout
            for _, layout in contiguous_range_layouts(int(problem["block_count"]), int(problem["device_count"]))
        ]
    elif args.observed_only:
        seeds = observed_layouts(records)
    else:
        seeds = candidate_layouts(problem, records, args.random)

    ranked: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for seed in seeds:
        layout = (
            seed
            if args.observed_only or args.contiguous_only
            else local_search(problem, seed, coefficients, args.passes, include_env_bytes, graph_features)
        )
        key = tuple(layout)
        if key in seen:
            continue
        seen.add(key)
        score, device_predictions = score_layout(problem, layout, coefficients, include_env_bytes, graph_features)
        ranked.append(
            {
                "layout": layout,
                "score": score,
                "device_predictions": device_predictions,
                "observed": key in observed_keys,
                "contiguous": key in contiguous_keys,
                "byte_balanced": key == byte_balanced_key,
            }
        )

    if not ranked:
        raise ValueError("suggestion search generated no candidate layouts")
    ranked.sort(key=lambda row: float(row["score"]))
    best = ranked[0]
    best_layout = best["layout"]
    best_devices = best["device_predictions"]
    print(f"timing_objective={timing_objective}")
    print(f"graph_features={str(graph_features).lower()}")
    print(f"model_source={model_source}")
    print_fit(coefficients, stats, graph_features)
    if args.contiguous_only:
        print("search=contiguous")
    else:
        print(f"search={'observed' if args.observed_only else 'local'}")
    print(f"predicted_gpu_s={float(best['score']):.9g}")
    print("predicted_device_gpu_s=" + ",".join(f"{value:.9g}" for value in best_devices))
    print(f"observed_layout={str(bool(best['observed'])).lower()}")
    print(f"contiguous_layout={str(bool(best['contiguous'])).lower()}")
    print(f"byte_balanced_layout={str(bool(best['byte_balanced'])).lower()}")
    print("layout=" + layout_string(best_layout))
    if not best["observed"]:
        print("warning=selected_layout_not_observed_in_trace")
    if args.top > 1:
        print("top_rank predicted_gpu_s observed contiguous byte_balanced layout")
        for rank, row in enumerate(ranked[: args.top], start=1):
            print(
                f"{rank} {float(row['score']):.9g} {str(bool(row['observed'])).lower()} "
                f"{str(bool(row['contiguous'])).lower()} {str(bool(row['byte_balanced'])).lower()} "
                f"{layout_string(row['layout'])}"
            )
    print("env UNI20_TENSORCONTRACTION_RABC_PLACEMENT=manual \\")
    print("    UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT=" + layout_string(best_layout))
    return 0


def cmd_ilp_suggest(args: argparse.Namespace) -> int:
    """Fit a model and solve the grouped right-first placement MILP."""
    records = trace_records(args)
    problem = trace_problem(records, term_records_for_args(args, records))
    if args.device_count is not None:
        if args.device_count <= 0:
            raise ValueError("--device-count must be positive")
        problem = dict(problem)
        problem["device_count"] = args.device_count

    model: dict[str, Any] | None = read_model(args.model) if args.model is not None else None
    if model is not None and (args.benchmark or args.calibration):
        raise ValueError("--model is mutually exclusive with --benchmark and --calibration for ilp-suggest")
    max_layout_segments = args.max_layout_segments
    if model is None:
        timing_objective = args.timing_objective
        graph_features = False
        include_env_bytes = include_environment_bytes(args)
        calibration_coefficients: dict[str, float] | None = None
        calibration_stats: dict[str, float] | None = None
        if args.calibration:
            calibration_records = calibration_records_from_paths(args.calibration)
            calibration_coefficients, calibration_stats = fit_calibration_coefficients(
                calibration_records, args.calibration_ridge
            )
            if not args.allow_negative:
                calibration_coefficients = clamp_negative_coefficients(calibration_coefficients)
        if args.benchmark:
            benchmark_drop_first = (
                args.drop_first_per_run
                if args.drop_first_per_run is not None
                else args.drop_first_per_layout
            )
            benchmark_records = benchmark_records_from_paths(args.benchmark)
            benchmark_records = drop_initial_per_benchmark_run(
                benchmark_records, problem, benchmark_drop_first
            )
            benchmark_records = filter_benchmark_records_by_layout(
                benchmark_records, problem, args.benchmark_layout_filter
            )
            if max_layout_segments is None and not args.allow_shape_extrapolation:
                max_layout_segments = max(
                    int(layout_shape_features(problem, benchmark_layout(record, problem))["layout_segments"])
                    for record in benchmark_records
                )
            coefficients, stats = fit_ilp_benchmark_coefficients(
                benchmark_records,
                problem,
                args.ridge,
                include_env_bytes,
                calibration_prior_coefficients(calibration_coefficients) if calibration_coefficients is not None else None,
                args.calibration_prior_weight if calibration_coefficients is not None else 0.0,
            )
            model_source = "benchmark+calibration_prior" if calibration_coefficients is not None else "benchmark"
        elif calibration_coefficients is not None and calibration_stats is not None:
            coefficients = calibration_coefficients
            stats = calibration_stats
            model_source = "calibration"
        else:
            coefficients, stats = fit_coefficients(records, args.ridge, include_env_bytes, graph_features, problem)
            model_source = "fit"
        if not args.allow_negative:
            coefficients = clamp_negative_coefficients(coefficients)
    else:
        timing_objective = str(model["timing_objective"])
        graph_features = bool(model["graph_features"])
        if graph_features:
            raise ValueError("ilp-suggest does not yet support graph-feature model files")
        include_env_bytes = include_environment_bytes_for_objective(timing_objective)
        coefficients = dict(model["coefficients"])
        stats = {str(name): float(value) for name, value in model.get("stats", {}).items()}
        model_source = str(args.model)

    result = solve_grouped_right_first_ilp(
        problem,
        coefficients,
        include_env_bytes,
        args.time_limit,
        args.mip_rel_gap,
        max_layout_segments,
        not args.no_symmetry_break,
    )
    layout = result["layout"]
    observed_keys = {tuple(item) for item in observed_layouts(records)}
    if int(problem["block_count"]) >= int(problem["device_count"]):
        contiguous_keys = {
            tuple(item)
            for _, item in contiguous_range_layouts(int(problem["block_count"]), int(problem["device_count"]))
        }
    else:
        contiguous_keys = set()
    byte_balanced_key = tuple(byte_balanced_layout(problem))

    print(f"timing_objective={timing_objective}")
    print("graph_features=false")
    print(f"model_source={model_source}")
    if args.benchmark:
        print("target=matvec_per_apply_s")
        print(f"benchmark_layout_filter={args.benchmark_layout_filter}")
        benchmark_drop_first = (
            args.drop_first_per_run if args.drop_first_per_run is not None else args.drop_first_per_layout
        )
        print(f"drop_first_per_run={benchmark_drop_first}")
    print(f"trace_drop_first_per_layout={args.drop_first_per_layout}")
    print(
        "max_layout_segments="
        + ("none" if max_layout_segments is None else str(max_layout_segments))
    )
    if args.calibration:
        print("calibration_target=mean_apply_s")
        print("calibration_fit_features=" + ",".join(CALIBRATION_SEED_FEATURE_NAMES))
        print(f"calibration_prior_weight={args.calibration_prior_weight:.9g}")
    print_fit_for_names(coefficients, stats, ilp_feature_names())
    print("search=ilp")
    print("solver=scipy.optimize.milp")
    print(f"milp_success={str(bool(result['success'])).lower()}")
    print(f"milp_status={result['status']}")
    print(f"milp_message={result['message']}")
    print(f"milp_variables={result['variables']}")
    print(f"milp_constraints={result['constraints']}")
    print(f"symmetry_break={result['symmetry_break']}")
    print(f"milp_objective_s={float(result['objective']):.9g}")
    if result["mip_dual_bound"] is not None:
        print(f"milp_dual_bound_s={float(result['mip_dual_bound']):.9g}")
    if result["mip_gap"] is not None:
        print(f"milp_gap={float(result['mip_gap']):.9g}")
    if result["mip_node_count"] is not None:
        print(f"milp_node_count={int(result['mip_node_count'])}")
    print(f"predicted_gpu_s={float(result['score']):.9g}")
    print("predicted_device_gpu_s=" + ",".join(f"{value:.9g}" for value in result["device_scores"]))
    print(f"observed_layout={str(tuple(layout) in observed_keys).lower()}")
    print(f"contiguous_layout={str(tuple(layout) in contiguous_keys).lower()}")
    print(f"byte_balanced_layout={str(tuple(layout) == byte_balanced_key).lower()}")
    if args.compact_layouts:
        print("layout_summary=" + compact_layout_string(layout))
        print("env_layout_omitted=true")
        print("env_layout_note=rerun_without_compact_layouts_for_full_manual_environment_value")
    else:
        print("layout=" + layout_string(layout))
        print("env UNI20_TENSORCONTRACTION_RABC_PLACEMENT=manual \\")
        print("    UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT=" + layout_string(layout))
    if abs(float(result["objective"]) - float(result["score"])) > 1.0e-7 * max(1.0, abs(float(result["score"]))):
        print("warning=milp_objective_differs_from_rescored_layout")
    if not result["success"]:
        print("warning=milp_solver_did_not_prove_optimality")
    return 0


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    root = argparse.ArgumentParser(description=__doc__)
    subcommands = root.add_subparsers(dest="command", required=True)

    def add_timing_objective(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--timing-objective",
            choices=("steady-state", "cold-start"),
            default="steady-state",
            help=(
                "model steady-state resident matvecs by ignoring A/C staging bytes, or cold-start setup by including "
                "them"
            ),
        )

    def add_graph_features(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--graph-features",
            action="store_true",
            help="augment fitting/scoring with graph cut and first-stage reuse counters from term traces",
        )

    def add_benchmark_model(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--model",
            choices=("critical", "device"),
            default="critical",
            help="benchmark model shape: anonymous critical path or device-identity-aware",
        )

    def add_benchmark_layout_filter(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--layout-filter",
            choices=("all", "contiguous"),
            default="all",
            help="restrict benchmark rows used for fitting and validation to a layout class",
        )

    def add_benchmark_drop_first(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--drop-first-per-run",
            "--drop-first-per-layout",
            dest="drop_first_per_run",
            type=int,
            default=0,
            help=(
                "drop the first N replay rows for each benchmark source/layout run; "
                "--drop-first-per-layout is kept as a compatibility alias"
            ),
        )

    def add_term_trace(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--term-trace",
            type=Path,
            help="read term-level problem metadata from this companion trace when the timing trace omits terms",
        )

    def add_required_term_trace(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--term-trace",
            type=Path,
            required=True,
            help="read term-level problem metadata from this companion trace",
        )

    summary = subcommands.add_parser("summary", help="summarize observed trace rows")
    summary.add_argument("trace", type=Path)
    summary.add_argument("--drop-first-per-layout", type=int, default=0)
    summary.add_argument("--group-layouts", action="store_true")
    summary.set_defaults(func=cmd_summary)

    bench_record = subcommands.add_parser("bench-record", help="convert replay benchmark stdout to JSONL records")
    bench_record.add_argument("stdout", nargs="+", type=Path, help="benchmark stdout file(s) to parse")
    bench_record.add_argument("--name", required=True, help="short label for this measured layout")
    bench_record.add_argument("--layout", required=True, help="layout string associated with this benchmark run")
    bench_record.add_argument(
        "--input-format",
        choices=("auto", "stdout", "benchfile"),
        default="auto",
        help="input file format; auto tries console stdout first, then MP_BENCHFILE table rows",
    )
    bench_record.add_argument("--output", type=Path, help="write JSONL records to this file instead of stdout")
    bench_record.add_argument("--append", action="store_true", help="append to --output instead of replacing it")
    bench_record.set_defaults(func=cmd_bench_record)

    bench_summary = subcommands.add_parser("bench-summary", help="summarize replay benchmark JSONL records")
    bench_summary.add_argument("benchmark", nargs="+", type=Path, help="benchmark JSONL file(s)")
    bench_summary.add_argument(
        "--compact-layouts", action="store_true", help="print layout summaries instead of full placement lists"
    )
    bench_summary.set_defaults(func=cmd_bench_summary)

    calibration_summary = subcommands.add_parser(
        "calibration-summary", help="summarize synthetic calibration JSONL records"
    )
    calibration_summary.add_argument("calibration", nargs="+", type=Path, help="calibration JSONL file(s)")
    calibration_summary.add_argument(
        "--compact-layouts", action="store_true", help="print layout summaries instead of full placement lists"
    )
    calibration_summary.set_defaults(func=cmd_calibration_summary)

    calibration_fit = subcommands.add_parser(
        "calibration-fit", help="fit operation-cost coefficients from synthetic calibration rows"
    )
    calibration_fit.add_argument("calibration", nargs="+", type=Path, help="calibration JSONL file(s)")
    calibration_fit.add_argument("--ridge", type=float, default=1.0e-9)
    calibration_fit.add_argument("--allow-negative", action="store_true")
    calibration_fit.add_argument("--output-model", type=Path, help="write fitted calibration model JSON")
    calibration_fit.set_defaults(func=cmd_calibration_fit)

    dmrg_summary = subcommands.add_parser("dmrg-summary", help="summarize live DMRG MP_BENCHFILE rows")
    dmrg_summary.add_argument("benchfile", nargs="+", type=Path, help="live DMRG MP_BENCHFILE table(s)")
    dmrg_summary.add_argument(
        "--min-states",
        default="16,64,256,512",
        help="comma-separated kept-rank thresholds to summarize",
    )
    dmrg_summary.add_argument(
        "--half-sweeps", action="store_true", help="also print rows grouped by half-sweep number"
    )
    dmrg_summary.add_argument(
        "--shapes",
        action="store_true",
        help="also group live rows by R/A/B/C output block count and shape signature",
    )
    dmrg_summary.set_defaults(func=cmd_dmrg_summary)

    bench_rank = subcommands.add_parser(
        "bench-rank", help="rank replay benchmark JSONL records after grouping identical layouts"
    )
    bench_rank.add_argument("benchmark", nargs="+", type=Path, help="benchmark JSONL file(s)")
    bench_rank.add_argument(
        "--compact-layouts", action="store_true", help="print layout summaries instead of full placement lists"
    )
    bench_rank.add_argument(
        "--selected-name",
        action="append",
        default=[],
        help="also report the measured rank of layouts produced by this benchmark name; may be repeated",
    )
    bench_rank.set_defaults(func=cmd_bench_rank)

    bench_struct_summary = subcommands.add_parser(
        "bench-struct-summary", help="summarize replay benchmark rows with static layout structure"
    )
    bench_struct_summary.add_argument("benchmark", nargs="+", type=Path, help="benchmark JSONL file(s)")
    bench_struct_summary.add_argument(
        "--sort",
        choices=(
            "observed",
            "right-flops",
            "mixed-flops",
            "peer-bytes",
            "peer-blocks",
            "segments",
            "transitions",
            "right-duplicates",
            "mixed-duplicates",
            "terms",
            "name",
        ),
        default="observed",
        help="summary sort order",
    )
    bench_struct_summary.add_argument(
        "--compact-layouts", action="store_true", help="print layout summaries instead of full placement lists"
    )
    add_required_term_trace(bench_struct_summary)
    bench_struct_summary.set_defaults(func=cmd_bench_struct_summary)

    bench_fit = subcommands.add_parser("bench-fit", help="fit no-trace replay benchmark timing coefficients")
    bench_fit.add_argument("benchmark", nargs="+", type=Path, help="benchmark JSONL file(s)")
    bench_fit.add_argument("--ridge", type=float, default=1.0e-9)
    bench_fit.add_argument("--clamp-negative", action="store_true", help="clamp negative coefficients before printing")
    add_benchmark_model(bench_fit)
    add_benchmark_layout_filter(bench_fit)
    add_benchmark_drop_first(bench_fit)
    add_timing_objective(bench_fit)
    add_graph_features(bench_fit)
    add_required_term_trace(bench_fit)
    bench_fit.set_defaults(func=cmd_bench_fit)

    bench_validate = subcommands.add_parser(
        "bench-validate", help="leave-one-layout-out validation for replay benchmark timing"
    )
    bench_validate.add_argument("benchmark", nargs="+", type=Path, help="benchmark JSONL file(s)")
    bench_validate.add_argument("--ridge", type=float, default=1.0e-9)
    bench_validate.add_argument("--clamp-negative", action="store_true")
    bench_validate.add_argument(
        "--candidate-score",
        choices=("fit", "monotonic-structure"),
        default="fit",
        help="score model to validate against held-out benchmark layouts",
    )
    bench_validate.add_argument(
        "--structure-ridge",
        type=float,
        default=1.0e-6,
        help="ridge penalty for --candidate-score=monotonic-structure",
    )
    bench_validate.add_argument(
        "--structure-iterations",
        type=int,
        default=1000,
        help="coordinate-descent iterations for --candidate-score=monotonic-structure",
    )
    bench_validate.add_argument(
        "--structure-feature-set",
        choices=tuple(sorted(STRUCTURE_SCORE_FEATURE_SETS)),
        default="all",
        help="structural counters used by --candidate-score=monotonic-structure",
    )
    bench_validate.add_argument(
        "--compact-layouts", action="store_true", help="print layout summaries instead of full placement lists"
    )
    add_benchmark_model(bench_validate)
    add_benchmark_layout_filter(bench_validate)
    add_benchmark_drop_first(bench_validate)
    add_timing_objective(bench_validate)
    add_graph_features(bench_validate)
    add_required_term_trace(bench_validate)
    bench_validate.set_defaults(func=cmd_bench_validate)

    bench_suggest = subcommands.add_parser(
        "bench-suggest", help="fit no-trace benchmark timing and suggest a layout"
    )
    bench_suggest.add_argument("benchmark", nargs="+", type=Path, help="benchmark JSONL file(s)")
    bench_suggest.add_argument("--ridge", type=float, default=1.0e-9)
    bench_suggest.add_argument("--passes", type=int, default=4)
    bench_suggest.add_argument("--random", type=int, default=16)
    bench_suggest.add_argument("--top", type=int, default=1, help="print the top N ranked candidate layouts")
    bench_suggest.add_argument(
        "--candidate-score",
        choices=("fit", "monotonic-structure"),
        default="fit",
        help="score used to rank candidate layouts",
    )
    bench_suggest.add_argument(
        "--structure-ridge",
        type=float,
        default=1.0e-6,
        help="ridge penalty for --candidate-score=monotonic-structure",
    )
    bench_suggest.add_argument(
        "--structure-iterations",
        type=int,
        default=1000,
        help="coordinate-descent iterations for --candidate-score=monotonic-structure",
    )
    bench_suggest.add_argument(
        "--structure-feature-set",
        choices=tuple(sorted(STRUCTURE_SCORE_FEATURE_SETS)),
        default="all",
        help="structural counters used by --candidate-score=monotonic-structure",
    )
    bench_suggest.add_argument("--observed-only", action="store_true")
    bench_suggest.add_argument(
        "--contiguous-only",
        action="store_true",
        help="search only layouts that assign contiguous block-index ranges to ordered devices",
    )
    bench_suggest.add_argument(
        "--segmented-only",
        action="store_true",
        help="search only alternating two-device segmented layouts with bounded cut enumeration",
    )
    bench_suggest.add_argument(
        "--max-segments",
        type=int,
        default=2,
        help="maximum number of alternating segments for --segmented-only",
    )
    bench_suggest.add_argument(
        "--segment-cut-stride",
        type=int,
        default=1,
        help="only consider segmented cuts at multiples of this block stride",
    )
    bench_suggest.add_argument(
        "--max-segment-layouts",
        type=int,
        default=20000,
        help="reject --segmented-only searches that would enumerate more layouts than this",
    )
    bench_suggest.add_argument(
        "--allow-shape-extrapolation",
        action="store_true",
        help="allow segmented candidates whose shape features were not observed in benchmark rows",
    )
    bench_suggest.add_argument("--clamp-negative", action="store_true", help="clamp negative coefficients before search")
    bench_suggest.add_argument(
        "--compact-layouts", action="store_true", help="print layout summaries instead of full placement lists"
    )
    bench_suggest.add_argument(
        "--show-structure",
        action="store_true",
        help="print graph-derived structural counters for ranked candidate layouts",
    )
    add_benchmark_model(bench_suggest)
    add_benchmark_layout_filter(bench_suggest)
    add_benchmark_drop_first(bench_suggest)
    add_timing_objective(bench_suggest)
    add_graph_features(bench_suggest)
    add_required_term_trace(bench_suggest)
    bench_suggest.set_defaults(func=cmd_bench_suggest)

    bench_tune = subcommands.add_parser(
        "bench-tune", help="scan ridge values with replay benchmark leave-one-layout-out validation"
    )
    bench_tune.add_argument("benchmark", nargs="+", type=Path, help="benchmark JSONL file(s)")
    bench_tune.add_argument("--ridges", default="1e-9,1e-7,1e-5,1e-3,1e-2,1e-1")
    bench_tune.add_argument("--include-clamped", action="store_true")
    add_benchmark_model(bench_tune)
    add_benchmark_layout_filter(bench_tune)
    add_benchmark_drop_first(bench_tune)
    add_timing_objective(bench_tune)
    add_graph_features(bench_tune)
    add_required_term_trace(bench_tune)
    bench_tune.set_defaults(func=cmd_bench_tune)

    order_summary = subcommands.add_parser(
        "order-summary", help="summarize left-first versus right-first term costs"
    )
    order_summary.add_argument("trace", type=Path)
    order_summary.add_argument("--drop-first-per-layout", type=int, default=0)
    order_summary.add_argument("--devices", action="store_true", help="print per-device order statistics")
    order_summary.add_argument("--blocks", action="store_true", help="print per-center-block order statistics")
    order_summary.add_argument("--top-blocks", type=int, default=0, help="limit per-layout block rows by delta")
    add_term_trace(order_summary)
    order_summary.set_defaults(func=cmd_order_summary)

    graph_summary = subcommands.add_parser(
        "graph-summary",
        help="summarize B-owner cuts and first-stage hypergraph reuse for traced term layouts",
    )
    graph_summary.add_argument("trace", type=Path)
    graph_summary.add_argument("--drop-first-per-layout", type=int, default=0)
    graph_summary.add_argument(
        "--layout",
        action="append",
        help="manual comma-separated output layout to summarize; may be supplied more than once",
    )
    graph_summary.add_argument(
        "--include-observed",
        action="store_true",
        help="include observed trace layouts after any manual layouts",
    )
    graph_summary.add_argument("--devices", action="store_true", help="print per-device graph metrics")
    add_term_trace(graph_summary)
    graph_summary.set_defaults(func=cmd_graph_summary)

    hypergraph_summary = subcommands.add_parser(
        "hypergraph-summary",
        help="summarize layout-relevant sparse f-tensor hyperedges independent of a chosen layout",
    )
    hypergraph_summary.add_argument("trace", type=Path)
    hypergraph_summary.add_argument("--drop-first-per-layout", type=int, default=0)
    hypergraph_summary.add_argument("--top", type=int, default=12, help="number of top hyperedges to print")
    add_term_trace(hypergraph_summary)
    hypergraph_summary.set_defaults(func=cmd_hypergraph_summary)

    fit = subcommands.add_parser("fit", help="fit per-device timing coefficients")
    fit.add_argument("trace", type=Path)
    fit.add_argument("--ridge", type=float, default=1.0e-9)
    fit.add_argument("--drop-first-per-layout", type=int, default=0)
    fit.add_argument("--clamp-negative", action="store_true", help="clamp negative coefficients before printing/saving")
    fit.add_argument("--output-model", type=Path, help="write fitted coefficients and metadata to this JSON file")
    add_timing_objective(fit)
    add_graph_features(fit)
    add_term_trace(fit)
    fit.set_defaults(func=cmd_fit)

    suggest = subcommands.add_parser("suggest", help="fit a model and suggest a layout")
    suggest.add_argument("trace", type=Path)
    suggest.add_argument("--model", type=Path, help="use a previously saved model JSON instead of fitting this trace")
    suggest.add_argument("--ridge", type=float, default=1.0e-9)
    suggest.add_argument("--passes", type=int, default=4)
    suggest.add_argument("--random", type=int, default=16)
    suggest.add_argument("--top", type=int, default=1, help="print the top N ranked candidate layouts")
    suggest.add_argument("--drop-first-per-layout", type=int, default=0)
    suggest.add_argument("--observed-only", action="store_true")
    suggest.add_argument(
        "--contiguous-only",
        action="store_true",
        help="search only layouts that assign contiguous block-index ranges to ordered devices",
    )
    suggest.add_argument("--allow-negative", action="store_true")
    add_timing_objective(suggest)
    add_graph_features(suggest)
    add_term_trace(suggest)
    suggest.set_defaults(func=cmd_suggest)

    ilp_suggest = subcommands.add_parser("ilp-suggest", help="solve the grouped right-first layout MILP")
    ilp_suggest.add_argument("trace", type=Path)
    ilp_suggest.add_argument("--model", type=Path, help="use a previously saved model JSON instead of fitting this trace")
    ilp_suggest.add_argument(
        "--benchmark",
        nargs="+",
        type=Path,
        help="fit MILP-compatible coefficients from no-trace replay benchmark JSONL instead of trace GPU timing",
    )
    ilp_suggest.add_argument(
        "--calibration",
        nargs="+",
        type=Path,
        help=(
            "fit synthetic calibration coefficients; with --benchmark these become a ridge prior, otherwise they "
            "are used directly"
        ),
    )
    ilp_suggest.add_argument(
        "--benchmark-layout-filter",
        choices=("all", "contiguous"),
        default="all",
        help="restrict benchmark rows used for --benchmark fitting",
    )
    ilp_suggest.add_argument("--ridge", type=float, default=1.0e-9)
    ilp_suggest.add_argument("--calibration-ridge", type=float, default=1.0e-9)
    ilp_suggest.add_argument(
        "--calibration-prior-weight",
        type=float,
        default=1.0e-10,
        help=(
            "ridge weight used to pull --benchmark fitting toward --calibration coefficients; keep this very small "
            "because live replay timings are the optimization target"
        ),
    )
    ilp_suggest.add_argument(
        "--drop-first-per-layout",
        type=int,
        default=0,
        help=(
            "drop the first N trace rows for each observed trace layout; when --benchmark is supplied this also "
            "acts as a compatibility fallback for --drop-first-per-run"
        ),
    )
    ilp_suggest.add_argument(
        "--drop-first-per-run",
        type=int,
        default=None,
        help="drop the first N replay rows for each benchmark source/layout run",
    )
    ilp_suggest.add_argument("--allow-negative", action="store_true")
    ilp_suggest.add_argument(
        "--device-count",
        type=int,
        help="override the term-trace device count when solving the placement problem",
    )
    ilp_suggest.add_argument("--time-limit", type=float, help="optional SciPy/HiGHS MILP time limit in seconds")
    ilp_suggest.add_argument("--mip-rel-gap", type=float, help="optional relative MIP gap target")
    ilp_suggest.add_argument(
        "--max-layout-segments",
        type=int,
        help="cap the number of contiguous owner segments in the solved layout",
    )
    ilp_suggest.add_argument(
        "--allow-shape-extrapolation",
        action="store_true",
        help="do not cap layout segments to the segment counts observed in --benchmark rows",
    )
    ilp_suggest.add_argument(
        "--no-symmetry-break",
        action="store_true",
        help="do not pin block 0 to device 0; useful when explicitly studying equivalent device permutations",
    )
    ilp_suggest.add_argument(
        "--compact-layouts", action="store_true", help="print a compact layout summary instead of full placement"
    )
    add_timing_objective(ilp_suggest)
    add_term_trace(ilp_suggest)
    ilp_suggest.set_defaults(func=cmd_ilp_suggest)

    validate = subcommands.add_parser("validate", help="leave-one-layout-out model validation")
    validate.add_argument("trace", type=Path)
    validate.add_argument("--ridge", type=float, default=1.0e-9)
    validate.add_argument("--drop-first-per-layout", type=int, default=0)
    validate.add_argument("--clamp-negative", action="store_true")
    add_timing_objective(validate)
    add_graph_features(validate)
    add_term_trace(validate)
    validate.set_defaults(func=cmd_validate)

    tune = subcommands.add_parser("tune", help="scan ridge values with leave-one-layout-out validation")
    tune.add_argument("trace", type=Path)
    tune.add_argument("--ridges", default="1e-9,1e-7,1e-5,1e-3,1e-2,1e-1")
    tune.add_argument("--drop-first-per-layout", type=int, default=0)
    tune.add_argument("--include-clamped", action="store_true")
    add_timing_objective(tune)
    add_graph_features(tune)
    add_term_trace(tune)
    tune.set_defaults(func=cmd_tune)

    layouts = subcommands.add_parser("layouts", help="generate manual placement layouts")
    layouts.add_argument("--block-count", type=int, required=True)
    layouts.add_argument("--device-count", type=int, required=True)
    layouts.add_argument("--random", type=int, default=0)
    layouts.add_argument("--seed", type=int, default=1)
    layouts.add_argument(
        "--contiguous-cuts",
        action="store_true",
        help="also emit every nonempty contiguous range partition for the requested device count",
    )
    layouts.add_argument(
        "--segmented-cuts",
        action="store_true",
        help="also emit bounded alternating two-device segmented layouts",
    )
    layouts.add_argument(
        "--max-segments",
        type=int,
        default=2,
        help="maximum number of alternating segments for --segmented-cuts",
    )
    layouts.add_argument(
        "--segment-cut-stride",
        type=int,
        default=1,
        help="only consider segmented cuts at multiples of this block stride",
    )
    layouts.add_argument(
        "--max-segment-layouts",
        type=int,
        default=20000,
        help="reject --segmented-cuts searches that would enumerate more layouts than this",
    )
    layouts.set_defaults(func=cmd_layouts)
    return root


def main(argv: list[str]) -> int:
    """Run the selected subcommand."""
    args = parser().parse_args(argv[1:])
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"rabc-trace-model.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
