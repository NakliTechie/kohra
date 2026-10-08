"""BD3LM block KV cache — torch reference: does the cached loop reproduce the cache-free one?

BD3LM is block-causal: block q attends to key blocks <= q. So once a block is finished its
keys/values never change, and a cache of every completed block is exact (not approximate,
unlike Fast-dLLM's cache on a bidirectional MDLM). Each denoise step then runs only the
current block, attending to [cache + itself] with no mask at all.

Cached loop, per generated block b (physical grid pos // block_size):
  1. prefill: one forward over all prompt tokens that sit in complete blocks before the
     first generated block -> the starting cache.
  2. denoise: forward over the current block only (its prompt prefix included), past = cache,
     positions = [start, end). Commit tokens exactly as the cache-free sampler does.
  3. commit: one more forward over the finished block to write its clean K/V into the cache.

Prints, for each prompt: cache-free vs cached text, whether the token ids are identical,
forward counts, and wall time. Greedy (temperature 0), fixed steps, fp32 on CPU.

  .venv/bin/python scripts/kv_reference.py
"""

import importlib.util
import os
import sys
import time

import torch
import transformers
from transformers.cache_utils import DynamicCache

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_ID = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
BLOCK = 32
NEG = -1e9


def load_a2d():
    """Register a2d-qwen3 without importing the whole dllm package (its training stack)."""
    path = os.path.join(ROOT, "vendor", "dllm", "dllm", "pipelines", "a2d", "models", "qwen3", "modeling_qwen3.py")
    spec = importlib.util.spec_from_file_location("a2d_qwen3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def block_causal_mask(T, dtype):
    bid = torch.arange(T) // BLOCK
    allow = bid.view(1, T) <= bid.view(T, 1)
    return torch.where(allow, 0.0, NEG).view(1, 1, T, T).to(dtype)


def schedule(n, steps):
    ks, rem = [], n
    for i in range(steps):
        if rem == 0:
            break
        k = min(round(rem / (steps - i)), rem)
        if k > 0:
            ks.append(k)
            rem -= k
    return ks


def commit_step(x, logits_rows, rows_pos, start, end, k, mask_id):
    """Commit the k most confident masked positions in [start,end). logits_rows[i] is the
    logits for absolute position rows_pos[i]."""
    cand = []
    for i, p in enumerate(rows_pos):
        if start <= p < end and x[p] == mask_id:
            pr = torch.softmax(logits_rows[i].float(), -1)
            tok = int(pr.argmax())
            cand.append((float(pr[tok]), p, tok))
    cand.sort(key=lambda c: -c[0])
    for _, p, tok in cand[:k]:
        x[p] = tok


@torch.no_grad()
def cache_free(model, prompt, n_new, steps, mask_id, eos_id):
    P = len(prompt)
    T = P + n_new
    x = torch.full((T,), eos_id, dtype=torch.long)
    x[:P] = torch.tensor(prompt)
    x[P:] = mask_id
    attn = block_causal_mask(T, model.dtype)
    blocks = list(range(P // BLOCK, (T - 1) // BLOCK + 1))
    spb = -(-steps // len(blocks))
    fwd = 0
    for b in blocks:
        s, e = b * BLOCK, min((b + 1) * BLOCK, T)
        for k in schedule(int((x[s:e] == mask_id).sum()), spb):
            logits = model(input_ids=x.view(1, T), attention_mask=attn).logits[0]
            fwd += 1
            commit_step(x, logits[s:e], list(range(s, e)), s, e, k, mask_id)
    return x[P:], fwd


@torch.no_grad()
def cached(model, prompt, n_new, steps, mask_id, eos_id):
    P = len(prompt)
    T = P + n_new
    x = torch.full((T,), eos_id, dtype=torch.long)
    x[:P] = torch.tensor(prompt)
    x[P:] = mask_id
    blocks = list(range(P // BLOCK, (T - 1) // BLOCK + 1))
    spb = -(-steps // len(blocks))
    fwd = 0

    def run(ids_slice, start, cache):
        L = ids_slice.numel()
        past = cache.get_seq_length() if cache is not None else 0
        mask = torch.zeros(1, 1, L, past + L, dtype=model.dtype)   # attend to all of cache + block
        out = model(input_ids=ids_slice.view(1, L), attention_mask=mask,
                    position_ids=torch.arange(start, start + L).view(1, L),
                    past_key_values=cache, use_cache=True)
        return out.logits[0], out.past_key_values

    # 1. prefill: complete prompt blocks before the first generated block
    cache = DynamicCache()
    pre_end = blocks[0] * BLOCK
    if pre_end > 0:
        _, cache = run(x[:pre_end], 0, cache)
        fwd += 1

    for b in blocks:
        s, e = b * BLOCK, min((b + 1) * BLOCK, T)
        for k in schedule(int((x[s:e] == mask_id).sum()), spb):
            snap = cache.get_seq_length()
            logits, cache = run(x[s:e], s, cache)
            cache.crop(snap)                      # denoise passes must not grow the cache
            fwd += 1
            commit_step(x, logits, list(range(s, e)), s, e, k, mask_id)
        _, cache = run(x[s:e], s, cache)          # 3. commit the finished block
        fwd += 1
    return x[P:], fwd


def main():
    load_a2d()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager").eval()
    mask_id, eos_id = tok.mask_token_id, tok.eos_token_id
    prompts = sys.argv[1:] or [
        "Lily runs 12 km/h for 4 hours. How far in 8 hours?",
        "Explain in three sentences why the sky is blue.",
    ]
    all_same = True
    for text in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, tokenize=True)
        t0 = time.time(); a, fa = cache_free(model, ids, 128, 128, mask_id, eos_id); ta = time.time() - t0
        t0 = time.time(); c, fc = cached(model, ids, 128, 128, mask_id, eos_id); tc = time.time() - t0
        same = torch.equal(a, c)
        all_same &= same
        dec = lambda t: tok.decode(t[: (t == eos_id).nonzero()[0, 0] if (t == eos_id).any() else len(t)],
                                   skip_special_tokens=True)
        print(f"\n=== {text}  (prompt {len(ids)} tok)")
        print(f"cache-free: {fa} fwd, {ta:.1f}s | cached: {fc} fwd, {tc:.1f}s | identical ids: {same}")
        if not same:
            diff = (a != c).nonzero().flatten().tolist()
            print(f"  first diff at gen pos {diff[0]}, {len(diff)} positions differ")
            print("  cache-free:", dec(a)[:300].replace("\n", " / "))
        print("  cached:    ", dec(c)[:300].replace("\n", " / "))
    print("\nALL IDENTICAL" if all_same else "\nDIFFERENCES FOUND")


if __name__ == "__main__":
    main()
