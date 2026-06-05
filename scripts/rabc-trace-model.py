#!/usr/bin/env python3
"""Fit and query empirical R/A/B/C layout cost models from JSONL traces."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any


BASE_FEATURE_NAMES = [
    "intercept",
    "bc_flops",
    "accumulate_flops",
    "b_local_bytes",
    "b_peer_bytes",
    "a_bytes",
    "c_bytes",
    "output_bytes",
    "intermediate_bytes",
    "terms",
    "unique_bc",
    "unique_a",
    "unique_b",
    "unique_c",
]

GRAPH_FEATURE_NAMES = [
    "b_cut_terms",
    "b_peer_blocks",
    "right_duplicate_groups",
]

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


def summarize_benchmark_records(records: list[dict[str, Any]]) -> None:
    """Print no-trace replay benchmark timing aggregates."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record.get("name", "")), str(record.get("layout", ""))), []).append(record)

    print("count mean_matvec_s min_matvec_s max_matvec_s mean_wall_s name layout")
    best_key: tuple[str, str] | None = None
    best_mean = float("inf")
    for key, rows in grouped.items():
        matvec = [float(record["matvec_s"]) for record in rows]
        wall = [float(record["wall_s"]) for record in rows]
        mean_matvec = sum(matvec) / len(matvec)
        mean_wall = sum(wall) / len(wall)
        if mean_matvec < best_mean:
            best_mean = mean_matvec
            best_key = key
        print(
            f"{len(rows)} {mean_matvec:.9g} {min(matvec):.9g} {max(matvec):.9g} "
            f"{mean_wall:.9g} {key[0]} {key[1]}"
        )
    if best_key is not None:
        print(f"best_mean_matvec_s={best_mean:.9g} name={best_key[0]} layout={best_key[1]}")


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


def benchmark_layout(record: dict[str, Any], problem: dict[str, Any]) -> list[int]:
    """Parse the benchmark record's manual placement layout."""
    return parse_layout(str(record.get("layout", "")), int(problem["block_count"]), int(problem["device_count"]))


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


def benchmark_layout_mean_matvec_seconds(records: list[dict[str, Any]]) -> float:
    """Return the mean resident matvec time for one no-trace benchmark layout."""
    timings = [float(record["matvec_s"]) for record in records]
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
    return values


def fit_benchmark_coefficients(
    records: list[dict[str, Any]],
    problem: dict[str, Any],
    ridge: float,
    include_env_bytes: bool,
    include_graph_features: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit whole-layout replay matvec time from critical-path layout features."""
    samples = [
        (
            layout_critical_path_vector(
                problem, benchmark_layout(record, problem), include_env_bytes, include_graph_features
            ),
            float(record["matvec_s"]),
        )
        for record in records
    ]
    return fit_linear_samples(
        samples, feature_names(include_graph_features), ridge, "benchmark data contains no replay matvec samples"
    )


def score_benchmark_layout(
    problem: dict[str, Any],
    layout: list[int],
    coefficients: dict[str, float],
    include_env_bytes: bool,
    include_graph_features: bool,
) -> tuple[float, list[float]]:
    """Predict no-trace resident matvec time for one candidate layout."""
    vector = layout_critical_path_vector(problem, layout, include_env_bytes, include_graph_features)
    names = feature_names(include_graph_features)
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
    coefficients: dict[str, float],
    passes: int,
    include_env_bytes: bool,
    include_graph_features: bool,
) -> list[int]:
    """Improve a layout using the benchmark-targeted critical-path score."""
    device_count = int(problem["device_count"])
    layout = start[:]
    best_score, _ = score_benchmark_layout(problem, layout, coefficients, include_env_bytes, include_graph_features)
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
                trial_score, _ = score_benchmark_layout(
                    problem, trial, coefficients, include_env_bytes, include_graph_features
                )
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
) -> list[dict[str, Any]]:
    """Predict each benchmarked layout from a fit over all other layouts."""
    grouped = group_benchmarks_by_layout(records, problem)
    if len(grouped) < 2:
        raise ValueError("benchmark validation requires at least two distinct layouts")

    rows: list[dict[str, Any]] = []
    for key, held_out in grouped.items():
        train = [record for other_key, group in grouped.items() if other_key != key for record in group]
        coefficients, _ = fit_benchmark_coefficients(
            train, problem, ridge, include_env_bytes, include_graph_features
        )
        if clamp_negative:
            coefficients = clamp_negative_coefficients(coefficients)
        predicted, _ = score_benchmark_layout(problem, list(key), coefficients, include_env_bytes, include_graph_features)
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


def print_benchmark_validation(rows: list[dict[str, Any]]) -> None:
    """Print leave-one-layout-out validation for no-trace benchmark timing."""
    stats = validation_stats(rows)
    print(
        f"layouts={int(stats['layouts'])} mae_s={stats['mae']:.9g} rmse_s={stats['rmse']:.9g} "
        f"r2={stats['r2']:.9g} best_observed_line={int(stats['best_observed_line'])} "
        f"best_predicted_line={int(stats['best_predicted_line'])} "
        f"top1_match={str(bool(stats['top1_match'])).lower()}"
    )
    print("observed_matvec_s predicted_matvec_s error_s count first_line names layout")
    for row in rows:
        print(
            f"{row['observed']:.9g} {row['predicted']:.9g} {row['error']:.9g} "
            f"{row['count']} {row['first_line']} {row['names']} {layout_string(list(row['layout']))}"
        )


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
    for term in terms:
        b = int(term["b"])
        r = int(term["r"])
        if not 0 <= b < block_count or not 0 <= r < block_count:
            raise ValueError("trace term references a center block outside block_count")
        input_bytes[b] = int(term["b_rows"]) * int(term["b_cols"]) * 8
        output_bytes[r] = int(term["r_rows"]) * int(term["r_cols"]) * 8
    return {
        "block_count": block_count,
        "device_count": device_count,
        "terms": terms,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
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
        "bc_flops": 0.0,
        "accumulate_flops": 0.0,
        "b_local_bytes": 0,
        "b_peer_bytes": 0,
        "a_bytes": 0,
        "c_bytes": 0,
        "output_bytes": 0,
        "intermediate_bytes": 0,
    }


def features_for_layout(
    problem: dict[str, Any], layout: list[int], include_env_bytes: bool, include_graph_features: bool
) -> list[dict[str, Any]]:
    """Compute right-first aggregate features for a candidate center layout."""
    validate_layout(problem, layout)
    device_count = int(problem["device_count"])
    devices = [empty_device_features(device) for device in range(device_count)]
    staged_bc: list[set[tuple[int, int]]] = [set() for _ in range(device_count)]
    staged_a: list[set[int]] = [set() for _ in range(device_count)]
    staged_b: list[set[int]] = [set() for _ in range(device_count)]
    staged_c: list[set[int]] = [set() for _ in range(device_count)]

    for block, device in enumerate(layout):
        devices[device]["input_blocks"] += 1
        devices[device]["output_blocks"] += 1
        devices[device]["output_bytes"] += problem["output_bytes"][block]

    for term in problem["terms"]:
        r = int(term["r"])
        a = int(term["a"])
        b = int(term["b"])
        c = int(term["c"])
        device = layout[r]
        row = devices[device]
        row["terms"] += 1

        bc_key = (b, c)
        if bc_key not in staged_bc[device]:
            staged_bc[device].add(bc_key)
            row["bc_flops"] += float(term["bc_flops"])
            row["intermediate_bytes"] += int(term["intermediate_bytes"])

        row["accumulate_flops"] += float(term["accumulate_flops"])

        if b not in staged_b[device]:
            staged_b[device].add(b)
            if layout[b] == device:
                row["b_local_bytes"] += problem["input_bytes"][b]
            else:
                row["b_peer_bytes"] += problem["input_bytes"][b]

        if a not in staged_a[device]:
            staged_a[device].add(a)
            row["a_bytes"] += int(term["a_rows"]) * int(term["a_cols"]) * 8

        if c not in staged_c[device]:
            staged_c[device].add(c)
            row["c_bytes"] += int(term["c_rows"]) * int(term["c_cols"]) * 8

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


def print_fit(coefficients: dict[str, float], stats: dict[str, float], include_graph_features: bool) -> None:
    """Print fitted model coefficients."""
    print(f"samples={int(stats['samples'])} rmse_s={stats['rmse']:.9g} r2={stats['r2']:.9g}")
    for name in feature_names(include_graph_features):
        print(f"{name} {coefficients[name]:.17g}")
    for name in ("bc_flops", "accumulate_flops"):
        coefficient = coefficients[name]
        if coefficient > 0.0:
            print(f"# implied_{name}_gflops={1.0 / (coefficient * 1.0e9):.9g}")
    for name in ("b_local_bytes", "b_peer_bytes", "a_bytes", "c_bytes", "output_bytes", "intermediate_bytes"):
        coefficient = coefficients[name]
        if coefficient > 0.0:
            print(f"# implied_{name}_gbps={1.0 / (coefficient * 1.0e9):.9g}")


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
        records.extend(parse_benchmark_stdout(path, args.layout, args.name))

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
    summarize_benchmark_records(records)
    return 0


def cmd_bench_fit(args: argparse.Namespace) -> int:
    """Fit replay benchmark matvec time from static layout features."""
    records = benchmark_records_from_paths(args.benchmark)
    problem = trace_problem([], read_trace(args.term_trace))
    include_env_bytes = include_environment_bytes(args)
    coefficients, stats = fit_benchmark_coefficients(
        records, problem, args.ridge, include_env_bytes, args.graph_features
    )
    if args.clamp_negative:
        coefficients = clamp_negative_coefficients(coefficients)
    print(f"timing_objective={args.timing_objective}")
    print(f"target=matvec_s")
    print(f"reduction=critical_path_max")
    print(f"graph_features={str(args.graph_features).lower()}")
    print_fit(coefficients, stats, args.graph_features)
    return 0


def cmd_bench_validate(args: argparse.Namespace) -> int:
    """Validate replay benchmark layout predictions."""
    records = benchmark_records_from_paths(args.benchmark)
    problem = trace_problem([], read_trace(args.term_trace))
    rows = leave_one_benchmark_layout_out(
        records,
        problem,
        args.ridge,
        args.clamp_negative,
        include_environment_bytes(args),
        args.graph_features,
    )
    print(f"timing_objective={args.timing_objective}")
    print(f"target=matvec_s")
    print(f"reduction=critical_path_max")
    print(f"graph_features={str(args.graph_features).lower()}")
    print_benchmark_validation(rows)
    return 0


def cmd_bench_suggest(args: argparse.Namespace) -> int:
    """Fit benchmark matvec timing and suggest a candidate layout."""
    records = benchmark_records_from_paths(args.benchmark)
    problem = trace_problem([], read_trace(args.term_trace))
    include_env_bytes = include_environment_bytes(args)
    coefficients, stats = fit_benchmark_coefficients(
        records, problem, args.ridge, include_env_bytes, args.graph_features
    )
    if not args.allow_negative:
        coefficients = clamp_negative_coefficients(coefficients)

    observed_keys = {tuple(layout) for layout in observed_benchmark_layouts(records, problem)}
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
        seeds = observed_benchmark_layouts(records, problem)
    else:
        seeds = candidate_benchmark_layouts(problem, records, args.random)

    ranked: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for seed in seeds:
        layout = (
            seed
            if args.observed_only or args.contiguous_only
            else benchmark_local_search(
                problem, seed, coefficients, args.passes, include_env_bytes, args.graph_features
            )
        )
        key = tuple(layout)
        if key in seen:
            continue
        seen.add(key)
        score, vector = score_benchmark_layout(problem, layout, coefficients, include_env_bytes, args.graph_features)
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
    print(f"target=matvec_s")
    print(f"reduction=critical_path_max")
    print(f"graph_features={str(args.graph_features).lower()}")
    print_fit(coefficients, stats, args.graph_features)
    if args.contiguous_only:
        print("search=contiguous")
    else:
        print(f"search={'observed' if args.observed_only else 'local'}")
    print(f"predicted_matvec_s={float(best['score']):.9g}")
    print(f"observed_layout={str(bool(best['observed'])).lower()}")
    print(f"contiguous_layout={str(bool(best['contiguous'])).lower()}")
    print(f"byte_balanced_layout={str(bool(best['byte_balanced'])).lower()}")
    print("layout=" + layout_string(best_layout))
    if not best["observed"]:
        print("warning=selected_layout_not_observed_in_benchmark")
    if args.top > 1:
        print("top_rank predicted_matvec_s observed contiguous byte_balanced layout")
        for rank, row in enumerate(ranked[: args.top], start=1):
            print(
                f"{rank} {float(row['score']):.9g} {str(bool(row['observed'])).lower()} "
                f"{str(bool(row['contiguous'])).lower()} {str(bool(row['byte_balanced'])).lower()} "
                f"{layout_string(row['layout'])}"
            )
    print("env UNI20_TENSORCONTRACTION_RABC_PLACEMENT=manual \\")
    print("    UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT=" + layout_string(best_layout))
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

    best = candidates[0]
    print(
        f"best ridge={best['ridge']:.9g} clamp_negative={str(best['clamp_negative']).lower()} "
        f"r2={best['stats']['r2']:.9g} rmse_s={best['stats']['rmse']:.9g} "
        f"top1_match={str(bool(best['stats']['top1_match'])).lower()}"
    )
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
    bench_record.add_argument("--output", type=Path, help="write JSONL records to this file instead of stdout")
    bench_record.add_argument("--append", action="store_true", help="append to --output instead of replacing it")
    bench_record.set_defaults(func=cmd_bench_record)

    bench_summary = subcommands.add_parser("bench-summary", help="summarize replay benchmark JSONL records")
    bench_summary.add_argument("benchmark", nargs="+", type=Path, help="benchmark JSONL file(s)")
    bench_summary.set_defaults(func=cmd_bench_summary)

    bench_fit = subcommands.add_parser("bench-fit", help="fit no-trace replay benchmark timing coefficients")
    bench_fit.add_argument("benchmark", nargs="+", type=Path, help="benchmark JSONL file(s)")
    bench_fit.add_argument("--ridge", type=float, default=1.0e-9)
    bench_fit.add_argument("--clamp-negative", action="store_true", help="clamp negative coefficients before printing")
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
    bench_suggest.add_argument("--observed-only", action="store_true")
    bench_suggest.add_argument(
        "--contiguous-only",
        action="store_true",
        help="search only layouts that assign contiguous block-index ranges to ordered devices",
    )
    bench_suggest.add_argument("--allow-negative", action="store_true")
    add_timing_objective(bench_suggest)
    add_graph_features(bench_suggest)
    add_required_term_trace(bench_suggest)
    bench_suggest.set_defaults(func=cmd_bench_suggest)

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
