# tasks/ewok_left_right.py
from typing import List, Dict, Callable, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def build_score_fn(
    model_id: str = "gpt2",
    max_len: int = 1024,
    batch_size: int = 8,
    apply_mask_ctx_factory: Optional[Callable[[dict], "ContextManager"]] = None,
) -> Callable[[dict, List[Dict[str, str]]], float]:
    """
    Returns score_fn(mask, items) -> mean Δ for items of form:
      {"context": str, "plausible": str, "implausible": str}
    If apply_mask_ctx_factory is provided, we run under that masking context
    (so heads/MLPs ablations from your circuit code actually take effect).
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt  = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt).to(dev).eval()

    def ids(s: str): return tok(s, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def batch_logprob(prompts: List[str], conts: List[str]) -> np.ndarray:
        # Build full sequences (prompt+cont), left-truncate to fit max_len
        batch_ids, starts = [], []
        for p, c in zip(prompts, conts):
            p_ids = ids(p) or [tok.eos_token_id]
            c_ids = ids(c)
            full  = p_ids + c_ids
            if len(full) > max_len:
                drop = len(full) - max_len
                full  = full[drop:]
                start = max(0, len(p_ids) - drop)
            else:
                start = len(p_ids)
            batch_ids.append(full); starts.append(start)

        pad = tok.pad_token_id
        T = max(len(x) for x in batch_ids)
        input_ids = torch.full((len(batch_ids), T), pad, dtype=torch.long, device=dev)
        attn      = torch.zeros((len(batch_ids), T), dtype=torch.long, device=dev)
        for i, seq in enumerate(batch_ids):
            input_ids[i, -len(seq):] = torch.tensor(seq, device=dev)
            attn[i, -len(seq):] = 1

        lsm = model(input_ids=input_ids, attention_mask=attn).logits.log_softmax(-1)
        out = []
        for i, seq in enumerate(batch_ids):
            s = 0.0
            for pos in range(max(1, starts[i]), len(seq)):
                s += float(lsm[i, -len(seq) + pos - 1, seq[pos]].item())
            out.append(s)
        return np.array(out, dtype=np.float32)

    class _null:
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    def score_fn(mask: dict, items: List[Dict[str, str]]) -> float:
        ctx_mgr = apply_mask_ctx_factory(mask) if apply_mask_ctx_factory else _null()
        deltas: List[float] = []
        with ctx_mgr:
            for s in range(0, len(items), batch_size):
                batch = items[s:s+batch_size]
                prompts = [b["context"] for b in batch]
                pos     = [b["plausible"] for b in batch]
                neg     = [b["implausible"] for b in batch]
                lp_pos = batch_logprob(prompts, pos)
                lp_neg = batch_logprob(prompts, neg)
                deltas.extend((lp_pos - lp_neg).tolist())
        return float(np.mean(deltas)) if deltas else 0.0

    return score_fn
