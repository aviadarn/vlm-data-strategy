#!/usr/bin/env python3
"""Choose the training examples for each arm of the data-strategy experiment.

The pool is ChartQA's train split: 28,299 question/answer pairs over charts, of which
20,901 (74%) were machine-generated from templates and 7,398 (26%) were written by people.
That is the shape of a real sourcing decision - synthetic augmentation is cheap and
plentiful, human annotation is scarce and expensive - so the question "at a fixed training
budget, which examples should we buy?" is worth an answer with a control attached.

Three arms, all the same size, all trained identically:

  random    N drawn from the pool as it is (74% machine). The control.
  human     N human-authored only. What a team buys when it decides human data is better.
  scored    N ranked by an intrinsic quality score that never sees the source label.

The third arm is the one worth building. If a cheap score computed from the question and
answer alone can pick out the good examples, a vendor pipeline can gate on it without
knowing or trusting who produced each item - which is the practical version of "evaluating
visual data and annotation quality". How well the score's ranking agrees with the
human/machine label is reported here as a finding in its own right, not used as an input.

Signals, all intrinsic (nothing reads the eval, the outcome, or the source):

  templatedness   how much this question's 4-gram shape repeats across the pool; generated
                  data comes from a handful of templates and repeats heavily
  near_dup        near-duplicate questions on the same chart
  q_specific      question length and presence of concrete referents
  answer_leak     whether the answer is copied verbatim from the question text
  degenerate      trivial or empty answers, single characters, placeholder text
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

STOP = {"the", "a", "an", "of", "in", "is", "are", "what", "which", "how", "to", "for",
        "and", "on", "was", "were", "does", "do", "with", "by", "at", "that", "this"}


def tokens(q: str) -> list[str]:
    return re.findall(r"[a-z0-9.]+", str(q).lower())


def shape(q: str) -> str:
    """The question with its content words removed - what's left is the template."""
    out = []
    for t in tokens(q):
        if re.fullmatch(r"-?\d*\.?\d+", t):
            out.append("#")
        elif t in STOP:
            out.append(t)
        else:
            out.append("*")
    return " ".join(out)


def ngrams(seq: list[str], n: int = 4) -> set[str]:
    return {" ".join(seq[i : i + n]) for i in range(max(len(seq) - n + 1, 1))}


def score_pool(df: pd.DataFrame) -> pd.DataFrame:
    q = df["query"].astype(str)
    a = df["answer"].astype(str)

    shapes = q.map(shape)
    shape_freq = Counter(shapes)
    # A template used by thousands of questions is a template; one used twice is a coincidence.
    df["templatedness"] = np.log1p(shapes.map(shape_freq).astype(float))

    # Near-duplicate questions asked about the same chart.
    key = df["image_key"].astype(str)
    dup = Counter(zip(key, q.str.lower()))
    df["near_dup"] = [dup[(k, qq)] - 1 for k, qq in zip(key, q.str.lower())]

    qlen = q.str.split().str.len().astype(float)
    df["q_specific"] = qlen

    # Answer copied verbatim out of the question: nothing to learn from the image.
    df["answer_leak"] = [
        1.0 if (len(str(aa)) > 2 and str(aa).lower() in str(qq).lower()) else 0.0
        for qq, aa in zip(q, a)
    ]

    df["degenerate"] = [
        1.0 if (len(str(aa).strip()) < 1 or str(aa).strip().lower() in
                {"n/a", "na", "none", "-", "?", "unknown"}) else 0.0
        for aa in a
    ]
    return df


def robust_z(v: np.ndarray) -> np.ndarray:
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    scale = 1.4826 * mad
    if scale < 1e-9:
        dev = np.abs(v - med).max()
        if dev < 1e-9:
            return np.zeros_like(v)
        scale = dev
    return (v - med) / scale


def quality(df: pd.DataFrame) -> np.ndarray:
    """Higher is better. Penalties for templated, duplicated, leaky or degenerate items;
    a mild bonus for specificity, capped so verbosity alone cannot buy rank."""
    pen = (1.2 * np.maximum(robust_z(df["templatedness"].values), 0)
           + 1.0 * np.maximum(robust_z(df["near_dup"].values.astype(float)), 0)
           + 2.0 * df["answer_leak"].values
           + 3.0 * df["degenerate"].values)
    bonus = np.clip(robust_z(df["q_specific"].values), -1.0, 1.0) * 0.4
    return -(pen) + bonus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/train-*.parquet")
    ap.add_argument("--n", type=int, default=4000, help="examples per arm")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="arms.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pool))
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=["query", "label", "human_or_machine", "image"])
        d["image_key"] = d["image"].map(lambda x: (x.get("path") or "")[:80])
        d = d.drop(columns=["image"])
        d["file"] = os.path.basename(f)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["row"] = np.arange(len(df))
    df["answer"] = df["label"].map(lambda x: x[0] if len(x) else "")
    df["human"] = df["human_or_machine"] == 0
    print(f"pool {len(df)}  human {int(df.human.sum())}  machine {int((~df.human).sum())}")

    df = score_pool(df)
    df["quality"] = quality(df)

    rng = np.random.default_rng(args.seed)
    if args.n > int(df.human.sum()):
        raise SystemExit(f"--n {args.n} exceeds the {int(df.human.sum())} human examples")

    arms = {
        "random": rng.choice(df["row"].values, size=args.n, replace=False),
        "human": rng.choice(df.loc[df.human, "row"].values, size=args.n, replace=False),
        # Stable sort with an explicit tie-break on row id. The default quicksort is not
        # stable, and this score produces many exact ties, so without this the selected set
        # differs between machines - it picked 24.8% human on one box and 20.5% on another
        # from identical inputs.
        "scored": df.sort_values(["quality", "row"], ascending=[False, True],
                                 kind="stable")["row"].values[: args.n],
    }

    print(f"\n{'arm':<10}{'n':>6}{'% human':>10}{'mean quality':>14}{'mean templated':>16}")
    print("-" * 56)
    summary = {}
    for name, rows in arms.items():
        sub = df[df["row"].isin(set(rows.tolist()))]
        summary[name] = dict(
            n=int(len(sub)),
            pct_human=round(100 * float(sub.human.mean()), 1),
            mean_quality=round(float(sub.quality.mean()), 3),
            mean_templatedness=round(float(sub.templatedness.mean()), 3),
            pct_answer_leak=round(100 * float(sub.answer_leak.mean()), 2),
            pct_numeric_answer=round(100 * float(
                sub.answer.map(lambda s: bool(re.fullmatch(r"-?[\d.,]+%?", str(s).strip()))).mean()), 1),
        )
        print(f"{name:<10}{len(sub):>6}{summary[name]['pct_human']:>9.1f}%"
              f"{summary[name]['mean_quality']:>14.3f}{summary[name]['mean_templatedness']:>16.3f}")
    print(f"{'pool':<10}{len(df):>6}{100*df.human.mean():>9.1f}%"
          f"{df.quality.mean():>14.3f}{df.templatedness.mean():>16.3f}")

    # Reported, never used as an input: does a label-free score recover the human data?
    k = args.n
    top = df.sort_values(["quality", "row"], ascending=[False, True], kind="stable").head(k)
    base = float(df.human.mean())
    print(f"\nlabel-free score vs the source label it never saw:")
    print(f"  human share in the top {k} by score: {100*float(top.human.mean()):.1f}%"
          f"   (pool baseline {100*base:.1f}%)")
    print(f"  lift: {float(top.human.mean())/base:.2f}x")

    json.dump(dict(n=args.n, seed=args.seed, summary=summary,
                   score_human_recovery=dict(
                       top_k=k, human_share_top_k=round(float(top.human.mean()), 4),
                       pool_human_share=round(base, 4),
                       lift=round(float(top.human.mean()) / base, 3)),
                   arms={k2: sorted(int(x) for x in v) for k2, v in arms.items()}),
              open(args.out, "w"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
