#!/usr/bin/env python3
"""Fit and query empirical R/A/B/C layout cost models from JSONL traces."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


FEATURE_NAMES = [
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


def layout_string(layout: list[int]) -> str:
    """Format a layout as the manual placement environment value."""
    return ",".join(str(device) for device in layout)


def feature_vector(features: dict[str, Any]) -> list[float]:
    """Convert one per-device feature dictionary into the model vector."""
    values = [1.0]
    values.extend(float(features.get(name, 0.0)) for name in FEATURE_NAMES[1:])
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


def fit_coefficients(records: list[dict[str, Any]], ridge: float) -> tuple[dict[str, float], dict[str, float]]:
    """Fit per-device elapsed time as a linear function of aggregate features."""
    samples: list[tuple[list[float], float]] = []
    for record in records:
        timings = {int(item["device"]): float(item["gpu_s"]) for item in record.get("device_timings", [])}
        for features in record.get("devices", []):
            device = int(features["device"])
            if device not in timings:
                continue
            samples.append((feature_vector(features), timings[device]))

    if not samples:
        raise ValueError("trace contains no per-device timing samples")

    scales = [1.0] * len(FEATURE_NAMES)
    for column in range(1, len(FEATURE_NAMES)):
        scales[column] = max(abs(vector[column]) for vector, _ in samples) or 1.0

    n = len(FEATURE_NAMES)
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
    coefficients = {
        name: scaled_coefficients[index] / scales[index] for index, name in enumerate(FEATURE_NAMES)
    }

    predictions = [sum(coefficients[name] * raw[index] for index, name in enumerate(FEATURE_NAMES))
                   for raw, _ in samples]
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
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(layout_key(record), []).append(record)

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


def features_for_layout(problem: dict[str, Any], layout: list[int]) -> list[dict[str, Any]]:
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

    for device in range(device_count):
        devices[device]["unique_bc"] = len(staged_bc[device])
        devices[device]["unique_a"] = len(staged_a[device])
        devices[device]["unique_b"] = len(staged_b[device])
        devices[device]["unique_c"] = len(staged_c[device])
    return devices


def predict_device_seconds(features: dict[str, Any], coefficients: dict[str, float]) -> float:
    """Predict elapsed seconds for one device."""
    return sum(coefficients[name] * value for name, value in zip(FEATURE_NAMES, feature_vector(features)))


def score_layout(problem: dict[str, Any], layout: list[int], coefficients: dict[str, float]) -> tuple[float, list[float]]:
    """Score a layout by the predicted maximum per-device elapsed time."""
    predictions = [predict_device_seconds(features, coefficients) for features in features_for_layout(problem, layout)]
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


def local_search(problem: dict[str, Any], start: list[int], coefficients: dict[str, float], passes: int) -> list[int]:
    """Improve a layout with deterministic single-block moves."""
    device_count = int(problem["device_count"])
    layout = start[:]
    best_score, _ = score_layout(problem, layout, coefficients)
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
                trial_score, _ = score_layout(problem, trial, coefficients)
                if trial_score < best_score:
                    best_score = trial_score
                    best_device = device
            if best_device != original:
                layout[block] = best_device
                improved = True
        if not improved:
            break
    return layout


def print_fit(coefficients: dict[str, float], stats: dict[str, float]) -> None:
    """Print fitted model coefficients."""
    print(f"samples={int(stats['samples'])} rmse_s={stats['rmse']:.9g} r2={stats['r2']:.9g}")
    for name in FEATURE_NAMES:
        print(f"{name} {coefficients[name]:.17g}")
    for name in ("bc_flops", "accumulate_flops"):
        coefficient = coefficients[name]
        if coefficient > 0.0:
            print(f"# implied_{name}_gflops={1.0 / (coefficient * 1.0e9):.9g}")
    for name in ("b_local_bytes", "b_peer_bytes", "a_bytes", "c_bytes", "output_bytes", "intermediate_bytes"):
        coefficient = coefficients[name]
        if coefficient > 0.0:
            print(f"# implied_{name}_gbps={1.0 / (coefficient * 1.0e9):.9g}")


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit and print a model."""
    records = trace_records(args)
    coefficients, stats = fit_coefficients(records, args.ridge)
    print_fit(coefficients, stats)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Summarize trace rows."""
    records = trace_records(args)
    if args.group_layouts:
        summarize_grouped(records)
    else:
        summarize(records)
    return 0


def cmd_layouts(args: argparse.Namespace) -> int:
    """Generate simple manual placement layouts."""
    rng = random.Random(args.seed)
    layouts = [
        ("range", [min(args.device_count - 1, block * args.device_count // args.block_count)
                   for block in range(args.block_count)]),
        ("alternating", [block % args.device_count for block in range(args.block_count)]),
    ]
    for index in range(args.random):
        layouts.append((f"random{index}", [rng.randrange(args.device_count) for _ in range(args.block_count)]))
    for name, layout in layouts:
        print(f"{name} {layout_string(layout)}")
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    """Fit a model and suggest the best searched layout."""
    records = trace_records(args)
    coefficients, stats = fit_coefficients(records, args.ridge)
    if not args.allow_negative:
        coefficients = {name: max(0.0, value) for name, value in coefficients.items()}
    problem = term_problem(records[-1])

    best_layout: list[int] | None = None
    best_score = float("inf")
    best_devices: list[float] = []
    for seed in candidate_layouts(problem, records, args.random):
        layout = local_search(problem, seed, coefficients, args.passes)
        score, device_predictions = score_layout(problem, layout, coefficients)
        if score < best_score:
            best_score = score
            best_layout = layout
            best_devices = device_predictions

    assert best_layout is not None
    print_fit(coefficients, stats)
    print(f"predicted_gpu_s={best_score:.9g}")
    print("predicted_device_gpu_s=" + ",".join(f"{value:.9g}" for value in best_devices))
    print("layout=" + layout_string(best_layout))
    print("env UNI20_TENSORCONTRACTION_RABC_PLACEMENT=manual \\")
    print("    UNI20_TENSORCONTRACTION_RABC_PLACEMENT_LAYOUT=" + layout_string(best_layout))
    return 0


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    root = argparse.ArgumentParser(description=__doc__)
    subcommands = root.add_subparsers(dest="command", required=True)

    summary = subcommands.add_parser("summary", help="summarize observed trace rows")
    summary.add_argument("trace", type=Path)
    summary.add_argument("--drop-first-per-layout", type=int, default=0)
    summary.add_argument("--group-layouts", action="store_true")
    summary.set_defaults(func=cmd_summary)

    fit = subcommands.add_parser("fit", help="fit per-device timing coefficients")
    fit.add_argument("trace", type=Path)
    fit.add_argument("--ridge", type=float, default=1.0e-9)
    fit.add_argument("--drop-first-per-layout", type=int, default=0)
    fit.set_defaults(func=cmd_fit)

    suggest = subcommands.add_parser("suggest", help="fit a model and suggest a layout")
    suggest.add_argument("trace", type=Path)
    suggest.add_argument("--ridge", type=float, default=1.0e-9)
    suggest.add_argument("--passes", type=int, default=4)
    suggest.add_argument("--random", type=int, default=16)
    suggest.add_argument("--drop-first-per-layout", type=int, default=0)
    suggest.add_argument("--allow-negative", action="store_true")
    suggest.set_defaults(func=cmd_suggest)

    layouts = subcommands.add_parser("layouts", help="generate manual placement layouts")
    layouts.add_argument("--block-count", type=int, required=True)
    layouts.add_argument("--device-count", type=int, required=True)
    layouts.add_argument("--random", type=int, default=0)
    layouts.add_argument("--seed", type=int, default=1)
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
