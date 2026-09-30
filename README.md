# Does paying for human-annotated data beat taking it at random?

A controlled data-strategy experiment on a vision-language model. At a fixed training
budget, three ways of choosing which examples to train on, measured against a frozen
held-out evaluation.

**Human-authored data wins by 1.7 points (p = 0.042). A label-free quality score that
recovers human data at 1.51× base rate does *not* — it performs indistinguishably from
random.** Provenance metadata has value that a cheap text heuristic cannot substitute for.

Total cost: **$1.81** of rented RTX 3090 time.

## The question

ChartQA's training pool is 28,299 chart question–answer pairs, of which **20,901 (74%)
were generated from templates** and **7,398 (26%) were written by people**. That is the
shape of a real sourcing decision: synthetic augmentation is cheap and plentiful, human
annotation is scarce and expensive. If you can afford to train on 4,000 examples, which
4,000 should you buy?

## Setup

| | |
|---|---|
| Model | Qwen2.5-VL-3B-Instruct, LoRA r=16 on attention + MLP (37.2M trainable) |
| Task | ChartQA — charts, the dense visual material of knowledge work |
| Budget | 4,000 examples per arm, 500 optimiser steps, effective batch 8 |
| Eval | ChartQA test, **2,500 questions, frozen before any training run** |
| Metric | ChartQA relaxed accuracy — 5% relative tolerance on numbers, exact match on text |
| Decoding | greedy, so no arm moves on sampling noise |

Everything except *which examples the arm contains* is identical: base model, adapter rank
and targets, learning rate, schedule, steps, seed, image budget, prompt, and the evaluation
itself.

Three arms:

- **random** — 4,000 drawn from the pool as it is (26.2% human). The control.
- **human** — 4,000 human-authored only. What a team buys when it decides human data is worth paying for.
- **scored** — 4,000 ranked by an intrinsic quality score that never sees the source label.

The third arm is the one worth building. If a cheap score computed from question and answer
text alone can pick out the good examples, a vendor pipeline can gate on it without knowing
or trusting who produced each item.

## Results

| arm | overall | human-written Qs | machine Qs | numeric | text |
|---|---|---|---|---|---|
| baseline (no fine-tuning) | 0.8268 | 0.7120 | 0.9416 | 0.8454 | 0.7658 |
| random (control) | 0.8404 | 0.7344 | 0.9464 | 0.8486 | 0.8137 |
| **human** | **0.8500** | **0.7536** | 0.9464 | **0.8595** | 0.8188 |
| scored | 0.8404 | 0.7392 | 0.9416 | 0.8514 | 0.8112 |

Paired McNemar against the random control, on the human-written slice (the two arms see the
identical frozen test set, so the comparison is paired rather than independent):

| arm | gained | lost | delta | p |
|---|---|---|---|---|
| **human** | 59 | 38 | **+1.7 pts** | **0.042** |
| scored | 51 | 46 | +0.4 pts | 0.685 |

### Read the slices, not the headline

The overall column understates everything, because the baseline already answers **94.2% of
the machine-generated questions** correctly before any fine-tuning. That slice is at its
ceiling — both fine-tuned arms land on exactly 0.9464 — so averaging over it dilutes any
effect. All the headroom is in the human-written half.

The two arms also buy different things. Against the baseline, `random` gains **+4.8 on text
answers** and **+0.3 on numeric** — it mostly learned ChartQA's answer *format*. The human
arm adds only **+0.5** more on text but **+1.1** on numeric, which is the part that requires
actually reading the chart.

### The negative result

The quality score works as a detector and fails as a selector.

It ranks by templatedness, near-duplication, answer leakage and degeneracy — nothing that
reads the source label. Its top 4,000 is **39.6% human against a 26.1% base rate, a lift of
1.51×**. So surface form genuinely carries provenance signal.

But the model trained on that selection scored **0.7392 on the human slice against the
control's 0.7344 — p = 0.685**. Recovering 1.5× more human data bought no measurable
benefit. Whatever makes human annotations better training material is not what this score
is detecting.

The practical reading: **demand provenance metadata from data vendors rather than inferring
it.** A heuristic that half-works as a classifier can still be worthless as a filter, and
the classifier metric would have told you it was working.

## Caveats

- **p = 0.042 is marginal, and this is one seed, one model, one task.** Suggestive, not
  settled. A second seed costs about $0.50 and should come before the claim carries weight.
- The effect is small in absolute terms: 1.7 points on 1,250 questions.
- The paired test used 1,228 of 1,250 questions — 22 dropped because pairs are keyed on
  question text and exact duplicates collapse.
- `scored` is not a coverage-maximising selection. A selection built for diversity rather
  than quality is the obvious next arm, and on a previous experiment in another modality
  it was diversity, not quality, that mattered.

## Two bugs worth naming

Both were caught in a probe before the real runs, and both would have produced confident
wrong numbers rather than crashes.

**Right-padding silently corrupted generation.** Decoder-only models need left padding for
batched generation; with right padding the pad tokens sit between the prompt and the first
generated token. The baseline scored **0.104** instead of **0.630** on the same 200
questions. Nothing errored.

**An unstable sort made the experiment unreproducible.** `sort_values` defaults to
quicksort, and the quality score produces many exact ties, so the `scored` arm differed
between two machines from identical inputs — 24.8% human on one, 20.5% on the other. Fixed
with a stable sort and an explicit tie-break on row id; all three arm selections now hash
identically across machines.

The first version of this README reported "the score does not recover human data at all,"
which was that bug, not a result. The real figure is 1.51×.

## Reproduce

```bash
python select_arms.py --n 4000 --out arms.json          # the three selections
python eval_chartqa.py --model Qwen/Qwen2.5-VL-3B-Instruct \
    --out results/baseline.eval.json                     # freeze the baseline first
python train_lora.py --arm human --out runs/human --steps 500
python eval_chartqa.py --model Qwen/Qwen2.5-VL-3B-Instruct \
    --adapter runs/human --out results/human.eval.json
python compare_arms.py --runs results                    # Wilson intervals + paired McNemar
```

`eval_chartqa.py` and `compare_arms.py` both have unit tests for the parts that decide the
answer — the relaxed-accuracy metric and the statistics — because they grade every arm and
a bug in either is indistinguishable from a result.
