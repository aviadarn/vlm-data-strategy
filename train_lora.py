#!/usr/bin/env python3
"""LoRA fine-tune Qwen2.5-VL on one arm's training examples.

Everything except which examples the arm contains is held fixed: the base model, adapter
rank and targets, learning rate, schedule, batch size, number of optimiser steps, sequence
and image budgets, and the seed. Arms are trained for the same number of steps rather than
the same number of epochs, because equal epochs on equal-sized sets is the same thing here
and equal steps is the claim that survives if a set size ever changes.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["random", "human", "scored", "coverage"])
    ap.add_argument("--arms-file", default="arms.json")
    ap.add_argument("--pool", default="data/train-*.parquet")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    # Effective batch is batch_size * grad_accum = 8, held fixed across arms. A 3B VLM with
    # chart images OOMs at batch 4 on a 24 GB card, so the split is 2x4 with checkpointing.
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--probe", type=int, default=0, help="stop after N steps and report rate")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="skip gradient checkpointing (faster, needs more VRAM)")
    args = ap.parse_args()

    import glob
    import pandas as pd
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    rows = set(json.load(open(args.arms_file))["arms"][args.arm])
    parts = []
    for f in sorted(glob.glob(args.pool)):
        parts.append(pd.read_parquet(f))
    pool = pd.concat(parts, ignore_index=True)
    pool["row"] = np.arange(len(pool))
    df = pool[pool["row"].isin(rows)].reset_index(drop=True)
    print(f"arm={args.arm}  examples={len(df)}  "
          f"human={int((df.human_or_machine==0).sum())} machine={int((df.human_or_machine==1).sum())}")

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "right"

    PROMPT = ("Answer the question about the chart with the shortest possible answer - a "
              "single number, word, or phrase. Do not explain.\n\nQuestion: {q}\nAnswer:")

    class Arm(Dataset):
        def __len__(self): return len(df)
        def __getitem__(self, i):
            r = df.iloc[i]
            return (Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"),
                    PROMPT.format(q=r["query"]),
                    str(r["label"][0] if len(r["label"]) else ""))

    def collate(batch):
        images = [b[0] for b in batch]
        texts, label_texts = [], []
        for _, prompt, answer in batch:
            msg = [{"role": "user", "content": [{"type": "image"},
                                                {"type": "text", "text": prompt}]}]
            head = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            texts.append(head + answer + processor.tokenizer.eos_token)
            label_texts.append(head)
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # Train on the answer only: the prompt and the image tokens are context, and letting
        # the loss cover them would mostly teach the model to reproduce its own prompt.
        for i, head in enumerate(label_texts):
            n_head = len(processor(text=[head], images=[images[i]],
                                   return_tensors="pt")["input_ids"][0])
            labels[i, :n_head] = -100
        enc["labels"] = labels
        return enc

    loader = DataLoader(Arm(), batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=2, drop_last=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    # Recompute activations instead of storing them: without this the backward pass through
    # the vision tower plus 36 language layers does not fit in 24 GB.
    if not args.no_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e6:.1f}M")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * args.steps), args.steps)

    model.train()
    step, t0, losses = 0, time.time(), []
    target = args.probe or args.steps
    while step < target:
        for batch in loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            out = model(**batch)
            (out.loss / args.grad_accum).backward()
            losses.append(float(out.loss))
            if (len(losses)) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0 or step == 1:
                    el = time.time() - t0
                    print(f"  step {step}/{target}  loss {np.mean(losses[-50:]):.4f}  "
                          f"{step/el:.2f} step/s  eta {(target-step)/max(step/el,1e-9)/60:.1f} min",
                          flush=True)
                if step >= target:
                    break

    el = time.time() - t0
    if args.probe:
        print(f"\nPROBE: {args.probe} steps in {el:.0f}s -> {args.probe/el:.3f} step/s")
        print(f"  {args.steps} steps would take {args.steps/(args.probe/el)/60:.1f} min")
        return

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    json.dump(dict(arm=args.arm, steps=args.steps, seed=args.seed, lr=args.lr,
                   rank=args.rank, batch_size=args.batch_size, grad_accum=args.grad_accum,
                   n_examples=int(len(df)), final_loss=round(float(np.mean(losses[-50:])), 4),
                   minutes=round(el / 60, 1)),
              open(os.path.join(args.out, "train_info.json"), "w"), indent=1)
    print(f"\nsaved {args.out}  final loss {np.mean(losses[-50:]):.4f}  {el/60:.1f} min")


if __name__ == "__main__":
    main()
