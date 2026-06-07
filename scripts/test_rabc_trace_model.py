#!/usr/bin/env python3
"""Tests for the R/A/B/C cost-model helpers in rabc-trace-model.py.

Focus: the incremental ``InputAnchoredProxyScorer`` must stay exactly in sync
with a full recompute after arbitrary single-block moves. The standalone
``input_anchored_proxy_score`` is an independent full-recompute implementation,
so it doubles as a second oracle.

Runnable standalone (``python3 scripts/test_rabc_trace_model.py``) with no
pytest dependency; pytest will also collect the ``test_*`` functions.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import Any

# The script filename is hyphenated, so import it by path.
_MODULE_PATH = Path(__file__).resolve().parent / "rabc-trace-model.py"
_spec = importlib.util.spec_from_file_location("rabc_trace_model", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
rtm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rtm)


# Parameter sets exercise every term in the proxy: flop-only, peer-byte-only,
# launch-only, and the segment-transition penalty path.
PARAM_SETS: list[dict[str, float]] = [
    {"gflops": 12.0, "peer_gbps": 8.0, "launch_us": 5.0, "transition_us": 0.0},
    {"gflops": 1.0, "peer_gbps": 1000.0, "launch_us": 0.0, "transition_us": 0.0},
    {"gflops": 1000.0, "peer_gbps": 1.0, "launch_us": 0.0, "transition_us": 0.0},
    {"gflops": 1000.0, "peer_gbps": 1000.0, "launch_us": 50.0, "transition_us": 0.0},
    {"gflops": 12.0, "peer_gbps": 8.0, "launch_us": 5.0, "transition_us": 25.0},
]


def make_random_problem(
    rng: random.Random,
    block_count: int,
    device_count: int,
    n_terms: int,
    c_count: int,
    force_rb_equal_fraction: float = 0.0,
) -> dict[str, Any]:
    """Build a synthetic term problem.

    ``bc_flops`` and ``intermediate_bytes`` are functions of the ``(b, c)``
    product (matching real data, where one B_b*C_c product has one cost),
    while ``accumulate_flops`` is per term.
    """

    def bc_flops_for(b: int, c: int) -> float:
        return 1000.0 * (b + 1) + 7.0 * (c + 1)

    def intermediate_bytes_for(b: int, c: int) -> int:
        return 64 * ((b * 3 + c * 5) % 11 + 1)

    terms: list[dict[str, Any]] = []
    for i in range(n_terms):
        b = rng.randrange(block_count)
        if rng.random() < force_rb_equal_fraction:
            r = b  # same center block is both output R_r and input B_b
        else:
            r = rng.randrange(block_count)
        c = rng.randrange(c_count)
        terms.append(
            {
                "r": r,
                "b": b,
                "c": c,
                "bc_flops": bc_flops_for(b, c),
                "accumulate_flops": float(13 * (i + 1)),
                "intermediate_bytes": intermediate_bytes_for(b, c),
            }
        )
    return {"block_count": block_count, "device_count": device_count, "terms": terms}


def _assert_lists_close(actual: list[float], expected: list[float], label: str) -> None:
    assert len(actual) == len(expected), f"{label}: length {len(actual)} != {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        if isinstance(e, int) and isinstance(a, int):
            assert a == e, f"{label}[{i}]: {a} != {e}"
        else:
            tol = 1e-9 * max(1.0, abs(e))
            assert abs(a - e) <= tol, f"{label}[{i}]: {a} != {e} (|diff|={abs(a - e)})"


def assert_incremental_matches_recompute(
    problem: dict[str, Any], layout: list[int], params: dict[str, float], scorer: Any
) -> None:
    """Compare a mutated scorer's state to a fresh build for the same layout."""
    fresh = rtm.InputAnchoredProxyScorer(problem, layout, **params)
    _assert_lists_close(scorer.first_flops, fresh.first_flops, "first_flops")
    _assert_lists_close(scorer.final_flops, fresh.final_flops, "final_flops")
    _assert_lists_close(scorer.peer_bytes, fresh.peer_bytes, "peer_bytes")
    _assert_lists_close(scorer.launches, fresh.launches, "launches")
    assert scorer.transitions == fresh.transitions, f"transitions {scorer.transitions} != {fresh.transitions}"

    incremental_score = scorer.score()
    fresh_score = fresh.score()
    standalone_score = rtm.input_anchored_proxy_score(problem, layout, **params)
    tol = 1e-9 * max(1.0, abs(fresh_score))
    assert abs(incremental_score - fresh_score) <= tol, f"score {incremental_score} != fresh {fresh_score}"
    assert abs(standalone_score - fresh_score) <= tol, f"standalone {standalone_score} != fresh {fresh_score}"


def test_move_equivalence() -> None:
    """Random move sequences keep the incremental scorer exact."""
    for seed in range(60):
        rng = random.Random(seed)
        block_count = rng.randint(2, 9)
        device_count = rng.randint(2, 4)
        n_terms = rng.randint(1, 24)
        c_count = rng.randint(1, 5)
        # Bias some problems toward r==b to stress the dual input/output role.
        rb_fraction = rng.choice([0.0, 0.3, 0.7])
        problem = make_random_problem(rng, block_count, device_count, n_terms, c_count, rb_fraction)
        params = rng.choice(PARAM_SETS)

        layout = [rng.randrange(device_count) for _ in range(block_count)]
        scorer = rtm.InputAnchoredProxyScorer(problem, layout, **params)
        assert_incremental_matches_recompute(problem, layout, params, scorer)

        for _ in range(40):
            block = rng.randrange(block_count)
            device = rng.randrange(device_count)
            scorer.move(block, device)
            assert_incremental_matches_recompute(problem, scorer.layout[:], params, scorer)


def test_rb_equal_focused() -> None:
    """A term whose output and input are the same center block stays correct."""
    rng = random.Random(123)
    # Every term has r == b, and one block (0) is reused heavily.
    terms = [
        {"r": 0, "b": 0, "c": 0, "bc_flops": 500.0, "accumulate_flops": 10.0, "intermediate_bytes": 128},
        {"r": 1, "b": 1, "c": 0, "bc_flops": 300.0, "accumulate_flops": 20.0, "intermediate_bytes": 256},
        {"r": 0, "b": 0, "c": 1, "bc_flops": 400.0, "accumulate_flops": 30.0, "intermediate_bytes": 512},
    ]
    problem = {"block_count": 3, "device_count": 3, "terms": terms}
    params = {"gflops": 10.0, "peer_gbps": 5.0, "launch_us": 4.0, "transition_us": 9.0}
    layout = [0, 0, 0]
    scorer = rtm.InputAnchoredProxyScorer(problem, layout, **params)
    # Drive block 0 (the r==b block) and its neighbours across all devices.
    for block, device in [(0, 1), (2, 2), (0, 2), (1, 2), (0, 0), (2, 0)]:
        scorer.move(block, device)
        assert_incremental_matches_recompute(problem, scorer.layout[:], params, scorer)


def test_move_to_same_device_is_noop() -> None:
    rng = random.Random(7)
    problem = make_random_problem(rng, 6, 3, 12, 3, 0.3)
    params = PARAM_SETS[0]
    layout = [rng.randrange(3) for _ in range(6)]
    scorer = rtm.InputAnchoredProxyScorer(problem, layout, **params)
    before = scorer.score()
    scorer.move(2, layout[2])
    assert scorer.score() == before
    assert scorer.layout == layout


def _distinct_layouts(rng: random.Random, block_count: int, device_count: int, count: int) -> list[list[int]]:
    layouts: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(layouts) < count and attempts < count * 40:
        attempts += 1
        layout = [rng.randrange(device_count) for _ in range(block_count)]
        key = tuple(layout)
        if key not in seen:
            seen.add(key)
            layouts.append(layout)
    return layouts


def test_calibration_recovers_known_model() -> None:
    """Fitting exact proxy-generated timings reproduces the true ranking."""
    for seed in range(20):
        rng = random.Random(1000 + seed)
        block_count = rng.randint(5, 9)
        device_count = rng.randint(2, 3)
        problem = make_random_problem(rng, block_count, device_count, rng.randint(8, 24), rng.randint(2, 5), 0.3)
        true = {
            "gflops": rng.choice([20.0, 100.0, 500.0]),
            "peer_gbps": rng.choice([5.0, 20.0, 80.0]),
            "launch_us": rng.choice([2.0, 8.0]),
            "transition_us": rng.choice([0.0, 15.0]),
        }
        layouts = _distinct_layouts(rng, block_count, device_count, 24)
        layout_seconds = [(layout, rtm.input_anchored_proxy_score(problem, layout, **true)) for layout in layouts]
        if len({round(y, 15) for _, y in layout_seconds}) < 4:
            continue  # degenerate: too few distinct timings to rank

        _weights, _physical, stats = rtm.fit_proxy_weights(problem, layout_seconds, ridge=1.0e-12)
        mean_y = sum(y for _, y in layout_seconds) / len(layout_seconds)
        assert stats["rmse"] <= 1.0e-3 * max(mean_y, 1.0e-18), f"seed {seed}: rmse {stats['rmse']} vs mean {mean_y}"
        assert stats["top1_regret"] == 0.0, f"seed {seed}: top1_regret {stats['top1_regret']}"
        assert stats["kendall_tau"] >= 0.98, f"seed {seed}: kendall_tau {stats['kendall_tau']}"


def test_calibration_leave_one_out_runs() -> None:
    """Leave-one-out validation runs and recovers a clean ranking on exact data."""
    rng = random.Random(42)
    problem = make_random_problem(rng, 7, 3, 18, 4, 0.3)
    true = {"gflops": 100.0, "peer_gbps": 20.0, "launch_us": 5.0, "transition_us": 10.0}
    layouts = _distinct_layouts(rng, 7, 3, 20)
    layout_seconds = [(layout, rtm.input_anchored_proxy_score(problem, layout, **true)) for layout in layouts]
    loo = rtm.proxy_leave_one_out(problem, layout_seconds, ridge=1.0e-12)
    assert loo["n"] == float(len(layout_seconds))
    assert loo["kendall_tau"] >= 0.9, f"loo kendall_tau {loo['kendall_tau']}"


def _run_all() -> int:
    tests = [
        test_move_equivalence,
        test_rb_equal_focused,
        test_move_to_same_device_is_noop,
        test_calibration_recovers_known_model,
        test_calibration_leave_one_out_runs,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_all())
