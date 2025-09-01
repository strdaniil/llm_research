# make_ewok_csvs.py — EWOK-only exporter for your schema
import os, argparse, random, collections
from typing import List, Optional, Tuple
import pandas as pd
from datasets import load_dataset, DatasetDict

def clean(x):
    return x.strip() if isinstance(x, str) else x

def choose_split(ds: DatasetDict, preferred: Optional[str] = None) -> str:
    if preferred and preferred in ds and len(ds[preferred]) > 0:
        return preferred
    for name in ["train", "validation", "test"]:
        if name in ds and len(ds[name]) > 0:
            return name
    for name in ds.keys():
        if len(ds[name]) > 0:
            return name
    raise RuntimeError("No non-empty splits found.")

def load_ewok_rows(ewok_split: Optional[str]) -> List[dict]:
    # Requires you to be logged in: `huggingface-cli login`
    ds = load_dataset("ewok-core/ewok-core-1.0")
    split = choose_split(ds, ewok_split)
    rows = list(ds[split])
    print(f"[EWOK] Using split: {split} (rows={len(rows)})")
    print(f"[EWOK] Keys: {list(rows[0].keys())}")
    return rows

def list_inventory(rows: List[dict], topk: int = 200):
    ctr = collections.Counter()
    for ex in rows:
        d = ex.get("Domain")
        a = ex.get("ConceptA")
        b = ex.get("ConceptB")
        if d and a and b:
            ctr[(d, a, b)] += 1
    print(f"[EWOK] Found {len(ctr)} (Domain,ConceptA,ConceptB) triples.")
    for (d,a,b), c in ctr.most_common(topk):
        print(f"{d}:{a},{b}  -> {c}")

def parse_pair_arg(s: str) -> Tuple[str, str, str]:
    # Format: "Domain:ConceptA,ConceptB" e.g., "spatial-relations:left,right"
    if ":" not in s or "," not in s:
        raise ValueError(f"Bad --pairs format: {s}. Use Domain:ConceptA,ConceptB")
    dom, rest = s.split(":", 1)
    a, b = rest.split(",", 1)
    return dom.strip(), a.strip(), b.strip()

def export_pair(rows: List[dict], dom: str, a: str, b: str, out_csv: str,
                limit_src: Optional[int], seed: int):
    # Exact match on Domain/ConceptA/ConceptB (case-sensitive)
    subset = [ex for ex in rows if ex.get("Domain")==dom and ex.get("ConceptA")==a and ex.get("ConceptB")==b]
    if not subset:
        raise RuntimeError(f"No rows matched {dom}:{a},{b}. Run with --list to see valid pairs.")

    rng = random.Random(seed)
    rng.shuffle(subset)
    if limit_src is not None:
        subset = subset[:limit_src]

    out_rows = []
    for ex in subset:
        c1, c2 = clean(ex.get("Context1")), clean(ex.get("Context2"))
        t1, t2 = clean(ex.get("Target1")),  clean(ex.get("Target2"))
        if c1 and c2 and t1 and t2:
            # Expand to two minimal-pair rows:
            out_rows.append({"context": c1, "plausible": t1, "implausible": t2})
            out_rows.append({"context": c2, "plausible": t2, "implausible": t1})

    if not out_rows:
        raise RuntimeError(f"After expansion, no usable rows for {dom}:{a},{b}.")
    pd.DataFrame(out_rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[EWOK] Wrote {out_csv}  ({len(out_rows)} rows)")

def main():
    ap = argparse.ArgumentParser(description="Export specific EWOK concept pairs to 3-column CSV.")
    ap.add_argument("--out_dir", type=str, default="data/tasks")
    ap.add_argument("--ewok_split", type=str, default=None, help="Force a split name if needed (e.g., 'test').")
    ap.add_argument("--list", action="store_true", help="List available (Domain,ConceptA,ConceptB) triples and exit.")
    ap.add_argument("--pairs", type=str, nargs="*", default=[],
                    help='Pairs like: "spatial-relations:left,right" "social-interactions:help,hinder"')
    ap.add_argument("--limit_src", type=int, default=500,
                    help="Limit SOURCE records per pair before 2x expansion (None for all).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_ewok_rows(args.ewok_split)

    if args.list or not args.pairs:
        print("\n[EWOK] Inventory of available (Domain:ConceptA,ConceptB):")
        list_inventory(rows, topk=500)
        if not args.pairs:
            print("\nPass one or more --pairs to export. Example:")
            print('  python make_ewok_csvs.py --pairs "spatial-relations:left,right" --out_dir data/tasks')
            return

    for pair in args.pairs:
        dom, a, b = parse_pair_arg(pair)
        fname = f"{dom}_{a}_{b}".replace("/", "_").replace(" ", "_")
        out_csv = os.path.join(args.out_dir, f"{fname}.csv")
        export_pair(rows, dom, a, b, out_csv, limit_src=args.limit_src, seed=args.seed)

if __name__ == "__main__":
    main()
