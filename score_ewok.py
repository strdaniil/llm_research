# score_ewok.py
import argparse, json, math, os
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class ColMap:
    context: str = "context"
    pos: str = "plausible"
    neg: str = "implausible"

COMMON_CTX = ["context", "prompt", "input", "premise", "prefix", "question"]
COMMON_POS = ["plausible", "positive", "true", "correct", "choice1", "target_pos", "answer_true"]
COMMON_NEG = ["implausible", "negative", "false", "incorrect", "choice2", "target_neg", "answer_false"]

def auto_cols(df: pd.DataFrame) -> ColMap:
    def pick(cands):
        for c in cands:
            if c in df.columns: return c
        return None
    ctx, pos, neg = pick(COMMON_CTX), pick(COMMON_POS), pick(COMMON_NEG)
    if not (ctx and pos and neg):
        raise ValueError(f"Couldn’t find needed columns in {list(df.columns)}.\n"
                         f"Pass --columns context=<..>,pos=<..>,neg=<..>")
    return ColMap(ctx, pos, neg)

def device_dtype():
    if torch.cuda.is_available(): return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available(): return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32

class LM:
    def __init__(self, model_id="gpt2", max_len=1024):
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dev, dt = device_dtype()
        self.dev, self.dt = dev, dt
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt).to(dev).eval()
        self.max_len = max_len

    def _ids(self, s: str):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def _forward_logits(self, batch_ids: List[List[int]]):
        maxL = max(len(x) for x in batch_ids)
        pad = self.tok.pad_token_id
        input_ids = torch.full((len(batch_ids), maxL), pad, dtype=torch.long)
        attn = torch.zeros((len(batch_ids), maxL), dtype=torch.long)
        for i, seq in enumerate(batch_ids):
            input_ids[i, -len(seq):] = torch.tensor(seq, dtype=torch.long)
            attn[i, -len(seq):] = 1
        input_ids, attn = input_ids.to(self.dev), attn.to(self.dev)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn)
        return out.logits  # [B,T,V]

    def logprob_continuation(self, context: str, cont: str) -> float:
        ctx_ids = self._ids(context)
        full_ids = ctx_ids + self._ids(cont)
        start = len(ctx_ids)
        if start == 0:
            ctx_ids = [self.tok.eos_token_id]
            full_ids = ctx_ids + self._ids(cont)
            start = 1
        # truncate from left to fit
        if len(full_ids) > self.max_len:
            drop = len(full_ids) - self.max_len
            full_ids = full_ids[drop:]
            start = max(0, start - drop)
        logits = self._forward_logits([full_ids])[0]  # [T,V]
        lsm = logits.log_softmax(dim=-1)
        total = 0.0
        for pos in range(start, len(full_ids)):
            prev = pos - 1
            tok = full_ids[pos]
            total += float(lsm[prev, tok].item())
        return total

def build_fewshot_prefix(examples: List[Dict[str,str]]) -> str:
    # very light template to avoid injecting style-specific quirks
    parts = []
    for ex in examples:
        parts.append(
            ex["context"].rstrip() + "\n" +
            f"Option A: {ex['plausible'].strip()}\n" +
            f"Option B: {ex['implausible'].strip()}\n" +
            "Answer: Option A\n\n"
        )
    return "".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="gpt2")
    ap.add_argument("--dataset_glob", type=str, required=True,
                    help='Glob to CSV(s) with columns context,plausible,implausible')
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--columns", type=str, default=None,
                    help="Override: context=<col>,pos=<col>,neg=<col>")
    ap.add_argument("--fewshot_json", type=str, default=None,
                    help="Optional JSON list of {context,plausible,implausible}")
    ap.add_argument("--max_ctx_tokens", type=int, default=900,
                    help="Keep last N tokens of (fewshot+context) so cont fits in 1024")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lm = LM(args.model_id, max_len=1024)

    fewshot_prefix = ""
    if args.fewshot_json:
        with open(args.fewshot_json, "r", encoding="utf-8") as f:
            few = json.load(f)
        fewshot_prefix = build_fewshot_prefix(few)

    # Gather files
    import glob
    files = sorted(glob.glob(args.dataset_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.dataset_glob}")

    all_rows = []
    for path in files:
        df = pd.read_csv(path)
        # map columns
        if args.columns:
            parts = dict(p.split("=",1) for p in args.columns.split(","))
            cmap = ColMap(parts["context"], parts["pos"], parts["neg"])
        else:
            cmap = auto_cols(df)

        for i, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(path)):
            ctx = str(row[cmap.context])
            pos = str(row[cmap.pos])
            neg = str(row[cmap.neg])

            # compose prompt: few-shot + context, then truncate tail-aware
            prompt = (fewshot_prefix + ctx).strip()
            ids = lm._ids(prompt)
            if len(ids) > args.max_ctx_tokens:
                ids = ids[-args.max_ctx_tokens:]
                prompt = lm.tok.decode(ids, skip_special_tokens=True)

            lp_pos = lm.logprob_continuation(prompt, pos)
            lp_neg = lm.logprob_continuation(prompt, neg)
            delta = lp_pos - lp_neg

            all_rows.append({
                "source_file": os.path.basename(path),
                "row_id": i,
                "context": ctx,
                "plausible": pos,
                "implausible": neg,
                "fewshot_k": 0 if not args.fewshot_json else len(json.loads(open(args.fewshot_json).read())),
                "logprob_pos": lp_pos,
                "logprob_neg": lp_neg,
                "delta": delta,
            })

    out = pd.DataFrame(all_rows)
    # quick per-file summary
    summ = out.groupby("source_file")["delta"].agg(["mean","std","count"]).reset_index().rename(
        columns={"mean":"delta_mean","std":"delta_std","count":"n"}
    )
    print("\nPer-file Δ summary:\n", summ.to_string(index=False))
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print("Wrote:", args.output_csv)

if __name__ == "__main__":
    main()
