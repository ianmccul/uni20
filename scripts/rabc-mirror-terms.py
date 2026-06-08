#!/usr/bin/env python3
"""Produce the mirror (reflected) R/A/B/C term structures from a term trace.

At the chain centre the ground state is reflection-symmetric, so the forward
ramp's asymmetric `left x right` central-bond records (e.g. M x M/2, where the
left bond grew first) have an exact mirror (M/2 x M) obtained by reflecting the
two-site problem: swap the left/right environments (A <-> C) and transpose every
block. This is a valid, consistent contraction (R^T += C^T . B^T . A^T) and gives
the distinct stage-cost balance the right-first bridge sees for the other
orientation -- without needing a reverse-direction DMRG sweep (which the generator
cannot do: a right-to-left sweep requires the left environments a left-to-right
sweep builds).

Usage: rabc-mirror-terms.py <forward_terms.jsonl> <mirror_terms.jsonl>
"""
import json
import sys


def mirror_term(t: dict) -> dict:
    m = dict(t)
    # Swap the A (left) and C (right) environment block indices.
    m["a"], m["c"] = t["c"], t["a"]
    # Transpose every tensor's (rows, cols); A' = C^T, C' = A^T, B' = B^T, R' = R^T.
    m["a_rows"], m["a_cols"] = t["c_cols"], t["c_rows"]
    m["c_rows"], m["c_cols"] = t["a_cols"], t["a_rows"]
    m["b_rows"], m["b_cols"] = t["b_cols"], t["b_rows"]
    m["r_rows"], m["r_cols"] = t["r_cols"], t["r_rows"]
    # Derived weights from the mirrored dims (real, 8-byte scalar).
    m["bc_flops"] = 2 * m["b_rows"] * m["b_cols"] * m["c_cols"]
    m["accumulate_flops"] = 2 * m["a_rows"] * m["a_cols"] * m["c_cols"]
    m["intermediate_bytes"] = m["b_rows"] * m["c_cols"] * 8
    return m


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: rabc-mirror-terms.py <forward_terms.jsonl> <mirror_terms.jsonl>", file=sys.stderr)
        return 2
    with open(sys.argv[1]) as fin, open(sys.argv[2], "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("kind") != "rabc_matvec" or not d.get("terms"):
                continue
            d["policy"] = "structure-mirror"
            d["terms"] = [mirror_term(t) for t in d["terms"]]
            fout.write(json.dumps(d) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
