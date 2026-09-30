#!/usr/bin/env python3
"""Frozen held-out evaluation for ChartQA, scored the way the benchmark defines it.

Built and frozen before any training run, so no arm can be tuned against it. Everything
here is fixed: the split (test, 2,500 questions), the prompt, the decoding parameters, and
the metric. The only thing that varies between arms is which training examples the model
saw.

Metric is ChartQA's relaxed accuracy: a numeric answer counts as correct within 5% relative
error, a text answer must match exactly after normalisation. Answering "35.6" when the chart
says "35.62" is not a model failure, and scoring it as one would make the whole comparison
noisier than the effect being measured.

The report slices by `human_or_machine`, because that is the axis the experiment is about
and a single headline number would hide exactly the effect worth seeing - the same mistake
a published 0.8096 mIoU made when it averaged over two very different platforms.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

RELAXED_TOL = 0.05


def normalise(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("%", "").replace("$", "").replace(",", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def as_number(s: str) -> float | None:
    t = normalise(s)
    m = re.fullmatch(r"-?\d*\.?\d+", t)
    return float(t) if m else None


def relaxed_correct(pred: str, gold: str) -> bool:
    """ChartQA relaxed accuracy: 5% relative tolerance on numbers, exact match on text."""
    p, g = as_number(pred), as_number(gold)
    if p is not None and g is not None:
        if g == 0:
            return abs(p) < 1e-9
        return abs(p - g) / abs(g) <= RELAXED_TOL
    return normalise(pred) == normalise(gold)


def extract_answer(raw: str) -> str:
    """Take the model's answer out of whatever it wrapped it in.

    The prompt asks for a bare answer, but an instruct model will sometimes add a sentence
    anyway. Grading that as wrong would measure instruction-following, not chart reading,
    and would do it unevenly across arms - so pull the last non-empty line and strip common
    lead-ins rather than punishing the wrapper.
    """
    t = raw.strip()
    t = re.sub(r"(?i)^(the\s+)?answer\s*(is)?\s*[:\-]?\s*", "", t)
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if lines:
        t = lines[-1]
    return t.strip().strip(".").strip()


PROMPT = (
    "Answer the question about the chart with the shortest possible answer - a single "
    "number, word, or phrase. Do not explain.\n\nQuestion: {q}\nAnswer:"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model id or path")
    ap.add_argument("--adapter", default="", help="LoRA adapter dir, or empty for zero-shot")
    ap.add_argument("--split", default="data/test.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole frozen split")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--max-pixels", type=int, default=602112, help="cap image tokens")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from PIL import Image
    import io
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    df = pd.read_parquet(args.split)
    if args.limit:
        df = df.head(args.limit)
    print(f"evaluating {len(df)} questions from {args.split}")

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    # Left padding is mandatory for batched generation on a decoder-only model: with right
    # padding the pad tokens sit between the prompt and the first generated token, and the
    # outputs are quietly wrong rather than failing. It scored 0.104 instead of ~0.70.
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  adapter: {args.adapter}")
    model.eval()

    rows, t0 = [], time.time()
    for start in range(0, len(df), args.batch_size):
        chunk = df.iloc[start : start + args.batch_size]
        msgs, images = [], []
        for _, r in chunk.iterrows():
            im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            images.append(im)
            msgs.append([{ "role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT.format(q=r["query"])}]}])
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        batch = processor(text=texts, images=images, return_tensors="pt",
                          padding=True).to("cuda")
        with torch.inference_mode():
            # Greedy: the comparison must not move because of sampling noise.
            out = model.generate(**batch, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None, top_k=None)
        gen = out[:, batch["input_ids"].shape[1]:]
        decoded = processor.batch_decode(gen, skip_special_tokens=True)
        for (_, r), raw in zip(chunk.iterrows(), decoded):
            pred = extract_answer(raw)
            gold = r["label"][0] if len(r["label"]) else ""
            rows.append(dict(query=r["query"], gold=gold, raw=raw.strip()[:120], pred=pred,
                             source="human" if r["human_or_machine"] == 0 else "machine",
                             correct=bool(relaxed_correct(pred, gold))))
        if start % (args.batch_size * 20) == 0:
            done = len(rows)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(df)}  {rate:.1f} q/s  "
                  f"running acc {sum(x['correct'] for x in rows)/done:.3f}")

    n = len(rows)
    overall = sum(r["correct"] for r in rows) / n
    by = {}
    for src in ("human", "machine"):
        sub = [r for r in rows if r["source"] == src]
        by[src] = dict(n=len(sub), acc=round(sum(r["correct"] for r in sub) / max(len(sub), 1), 4))
    numeric = [r for r in rows if as_number(r["gold"]) is not None]
    text = [r for r in rows if as_number(r["gold"]) is None]
    report = dict(
        model=args.model, adapter=args.adapter or None, split=args.split, n=n,
        relaxed_accuracy=round(overall, 4), by_source=by,
        by_answer_type=dict(
            numeric=dict(n=len(numeric),
                         acc=round(sum(r["correct"] for r in numeric) / max(len(numeric), 1), 4)),
            text=dict(n=len(text),
                      acc=round(sum(r["correct"] for r in text) / max(len(text), 1), 4))),
        seconds=round(time.time() - t0, 1),
        predictions=rows,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nrelaxed accuracy {overall:.4f} over {n}")
    print(f"  human   {by['human']['acc']:.4f}  (n={by['human']['n']})")
    print(f"  machine {by['machine']['acc']:.4f}  (n={by['machine']['n']})")
    print(f"  numeric {report['by_answer_type']['numeric']['acc']:.4f}"
          f"   text {report['by_answer_type']['text']['acc']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
