# import os, argparse, pandas as pd
# from transformers import AutoTokenizer

# ap = argparse.ArgumentParser()
# ap.add_argument("--scored_csv", required=True)  # e.g., ../output/left_right_gpt2_scores.csv
# ap.add_argument("--out_task", default="ewok-social")
# ap.add_argument("--model_family", default="gpt2")   # filename expected by EAPDataset
# ap.add_argument("--tokenizer_id", default="gpt2")   # which tokenizer to use for token ids
# args = ap.parse_args()

# df = pd.read_csv(args.scored_csv)

# tok = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=True)

# def last_token_id(s: str) -> int:
#     # assume s already ends with the discriminative word and no trailing punctuation
#     ids = tok(" " + s.strip().split()[-1], add_special_tokens=False).input_ids
#     # for GPT-2, " left"/" right" become single tokens; take the last piece defensively
#     return ids[-1]

# clean_list, corrupt_list, clean_idx, corrupt_idx = [], [], [], []

# for _, r in df.iterrows():
#     ctx = str(r["context"]).strip().rstrip(".")
#     pos = str(r["plausible"]).strip().rstrip(".")
#     neg = str(r["implausible"]).strip().rstrip(".")

#     # Extract the discriminative word (left/right) by picking the differing word in pos/neg.
#     # Minimal robust approach: try to find 'left'/'right' explicitly.
#     # this was done for left right logic
#     def pick_word(s: str) -> str:
#         words = s.split()
#         for w in ("left", "right"):
#             if w in [w2.strip(",.") for w2 in words]:
#                 return w
#         # fallback: last word
#         return words[-1].strip(",.")


#     pos_word = pick_word(pos)
#     neg_word = pick_word(neg)

#     clean = f"{ctx} {pos_word}"
#     corrupt = f"{ctx} {neg_word}"

#     clean_list.append(clean)
#     corrupt_list.append(corrupt)
#     clean_idx.append(last_token_id(pos_word))
#     corrupt_idx.append(last_token_id(neg_word))


# out = pd.DataFrame({
#     "clean": clean_list,
#     "corrupted": corrupt_list,
#     "clean_idx": clean_idx,
#     "corrupted_idx": corrupt_idx,
# })

# os.makedirs(f"data/{args.out_task}", exist_ok=True)
# out_path = f"data/{args.out_task}/{args.model_family}.csv"
# out.to_csv(out_path, index=False)
# print("Wrote", out_path, "rows=", len(out))



# make_eap_from_scored_general.py
# make_eap_from_scored_general.py
import os, re, argparse
import pandas as pd
from transformers import AutoTokenizer

# Common fallbacks for column names if --columns isn't provided
COMMON_CTX = ["context", "prompt", "input", "premise", "prefix", "question"]
COMMON_POS = ["plausible", "positive", "true", "correct", "choice1", "target_pos", "answer_true"]
COMMON_NEG = ["implausible", "negative", "false", "incorrect", "choice2", "target_neg", "answer_false"]

def auto_cols(df: pd.DataFrame):
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None
    ctx, pos, neg = pick(COMMON_CTX), pick(COMMON_POS), pick(COMMON_NEG)
    if not (ctx and pos and neg):
        raise ValueError(
            f"Couldn't find needed columns in {list(df.columns)}.\n"
            f"Either rename your columns to one of:\n"
            f"  ctx={COMMON_CTX}\n  pos={COMMON_POS}\n  neg={COMMON_NEG}\n"
            f"or pass --columns context=<..>,pos=<..>,neg=<..>"
        )
    return ctx, pos, neg

def parse_cols_arg(s: str):
    parts = dict(p.split("=", 1) for p in s.split(","))
    return parts["context"], parts["pos"], parts["neg"]

PUNCT_RE = re.compile(r"[.,!?;:]+$")

def last_token_id_of_phrase(tok, phrase: str) -> int:
    # strip trailing spaces + punctuation
    text = PUNCT_RE.sub("", str(phrase).strip())
    if not text:
        raise ValueError("Empty phrase after stripping punctuation.")
    last_word = text.split()[-1]
    # space-prefix for GPT-2 BPE consistency
    ids = tok(" " + last_word, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"No token ids for last word: {last_word!r}")
    return ids[-1]

def main():
    ap = argparse.ArgumentParser(description="Convert scored pair CSV to EAP-ready CSV.")
    ap.add_argument("--scored_csv", required=True, help="CSV with columns: context, plausible, implausible (names can vary).")
    ap.add_argument("--out_task", required=True, help="Task name for output folder, e.g. ewok-social-relations")
    ap.add_argument("--model_family", default="gpt2", help="File name used by EAPDataset (e.g., gpt2, llama, etc.)")
    ap.add_argument("--tokenizer_id", default="gpt2", help="HF tokenizer to map last tokens")
    ap.add_argument("--columns", default=None, help="Override columns: context=<col>,pos=<col>,neg=<col>")
    args = ap.parse_args()

    df = pd.read_csv(args.scored_csv)
    # Choose columns
    if args.columns:
        ctx_col, pos_col, neg_col = parse_cols_arg(args.columns)
    else:
        ctx_col, pos_col, neg_col = auto_cols(df)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    clean_list, corrupt_list, clean_idx, corrupt_idx = [], [], [], []
    filtered_out = 0

    for _, r in df.iterrows():
        ctx = str(r.get(ctx_col, "")).strip()
        pos = str(r.get(pos_col, "")).strip()
        neg = str(r.get(neg_col, "")).strip()
        # Skip blanks
        if not ctx or not pos or not neg:
            filtered_out += 1
            continue

        # Build full prompts
        clean = f"{ctx.rstrip()} {pos.lstrip()}".strip()
        corrupt = f"{ctx.rstrip()} {neg.lstrip()}".strip()

        try:
            ci = last_token_id_of_phrase(tok, pos)
            ni = last_token_id_of_phrase(tok, neg)
        except Exception:
            filtered_out += 1
            continue

        clean_list.append(clean)
        corrupt_list.append(corrupt)
        clean_idx.append(ci)
        corrupt_idx.append(ni)

    out = pd.DataFrame({
        "clean": clean_list,
        "corrupted": corrupt_list,
        "clean_idx": clean_idx,
        "corrupted_idx": corrupt_idx,
    })

    out_dir = os.path.join("data", args.out_task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.model_family}.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {out_path} rows={len(out)} filtered_out={filtered_out}")

if __name__ == "__main__":
    main()
