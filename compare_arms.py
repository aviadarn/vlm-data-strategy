#!/usr/bin/env python3
"""Compare the arms against the baseline, with intervals and the slice that matters.

Four numbers on a 2,500-question eval invite over-reading, so this reports Wilson intervals
on every cell and, for each arm against the random control, a paired test on the questions
they both answered. Paired matters here: the arms see the identical frozen test set, so
comparing them as independent samples throws away the pairing and widens the interval for
no reason.

The headline is deliberately not the overall number. The baseline already answers 94% of
the machine-generated questions correctly before any fine-tuning, so the overall figure is
dominated by a slice with almost no headroom left. The human-written slice is where an
effect can appear at all.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Paired comparison of two arms on the same questions.

    b01 = b right where a wrong, b10 = a right where b wrong. Under the null the
    discordant pairs split evenly, so a normal approximation with continuity correction
    gives a two-sided p without needing scipy.
    """
    b01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    b10 = sum(1 for x, y in zip(a, b) if x and (not y))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    chi = (abs(b01 - b10) - 1) ** 2 / n if n > 0 else 0.0
    # two-sided p from chi-square with 1 df == erfc(sqrt(chi/2))
    p = math.erfc(math.sqrt(max(chi, 0.0) / 2))
    return b01, b10, p


def load(path: str) -> dict:
    d = json.load(open(path))
    d["name"] = os.path.basename(path).replace(".eval.json", "")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results")
    ap.add_argument("--out", default="results/comparison.json")
    args = ap.parse_args()

    order = ["baseline", "random", "human", "scored"]
    runs = {}
    for name in order:
        p = os.path.join(args.runs, f"{name}.eval.json")
        if os.path.exists(p):
            runs[name] = load(p)
    if not runs:
        raise SystemExit(f"no eval JSONs in {args.runs}")

    def cell(d: dict, pred=lambda r: True) -> tuple[int, int]:
        sub = [r for r in d["predictions"] if pred(r)]
        return sum(r["correct"] for r in sub), len(sub)

    slices = {
        "overall": lambda r: True,
        "human": lambda r: r["source"] == "human",
        "machine": lambda r: r["source"] == "machine",
    }

    print(f"{'arm':<10}" + "".join(f"{s:>26}" for s in slices))
    print("-" * (10 + 26 * len(slices)))
    table = {}
    for name in order:
        if name not in runs:
            continue
        cells, line = {}, f"{name:<10}"
        for s, fn in slices.items():
            k, n = cell(runs[name], fn)
            lo, hi = wilson(k, n)
            cells[s] = dict(k=k, n=n, acc=round(k / n, 4),
                            ci=[round(lo, 4), round(hi, 4)])
            line += f"{k/n:>14.4f} [{lo:.2f},{hi:.2f}]"
        table[name] = cells
        print(line)

    # Paired tests against the random control, on the slice with headroom.
    print("\npaired comparison vs the random control (McNemar, human-written slice):")
    if "random" in runs:
        ctrl = {r["query"]: r["correct"] for r in runs["random"]["predictions"]
                if r["source"] == "human"}
        for name in ("human", "scored"):
            if name not in runs:
                continue
            arm = {r["query"]: r["correct"] for r in runs[name]["predictions"]
                   if r["source"] == "human"}
            keys = [q for q in ctrl if q in arm]
            a = [ctrl[q] for q in keys]
            b = [arm[q] for q in keys]
            b01, b10, p = mcnemar(a, b)
            delta = (sum(b) - sum(a)) / max(len(keys), 1)
            table.setdefault(name, {})["vs_random_human_slice"] = dict(
                n_paired=len(keys), gained=b01, lost=b10,
                delta=round(delta, 4), p=round(p, 5))
            verdict = "significant" if p < 0.05 else "not distinguishable"
            print(f"  {name:<8} n={len(keys):<5} gained {b01:<4} lost {b10:<4} "
                  f"delta {delta:+.4f}  p={p:.4f}  ({verdict})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(table, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
