"""Block-diffusion (bd3lm) denoising loop on the ONNX export — the template for the JS port.

Pure numpy + onnxruntime. Unlike MDLM (no mask), bd3lm's graph takes a 2nd input: a
[1,1,T,T] additive **block-causal** attention_mask (0 = attend, -1e9 = block) on the
*physical* grid block_id(pos) = pos // block_size from position 0. The mask is STATIC —
built once, passed every forward; block-causality is what lets the cache-free full-canvas
forward give correct current-block logits regardless of still-masked future blocks.

Faithful to dllm BD3LMSampler at defaults BUT for a 2-input graph (no position_ids input):
we rely on the model's default arange positions, so the prompt is placed at [0,P) with NO
left-padding (padding would corrupt arange RoPE). Generation denoises physical blocks
left-to-right starting from the block containing P; the prompt's last partial block is
completed by the first generated tokens (bidirectional within that block). This is exactly
the config export_bd3lm.py parity verified (arange positions + block_causal_mask(T,32)).

Usage:
  .venv/bin/python scripts/gencheck_bd3lm.py \
      --model models/qwen3-0.6b-bd3lm-onnx/model_fp32_fused.onnx \
      --prompt "Lily runs 12 km/h for 4 hours. How far in 8 hours?" \
      --max-new-tokens 128 --steps 128 --block-size 32
"""

import argparse
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_ID = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
NEG = -1e9


def block_causal_mask(T, block_size):
    """[1,1,T,T] additive block-causal mask: 0 where block(key)<=block(query), else -1e9."""
    pos = np.arange(T)
    bid = pos // block_size
    allow = bid[None, :] <= bid[:, None]            # [query, key]
    return np.where(allow, 0.0, NEG).astype(np.float32).reshape(1, 1, T, T)


def transfer_schedule(n_masked, steps):
    """Per-step reveal counts for one block (linear scheduler). Mirrors sample_onnx.py."""
    ks, remaining = [], n_masked
    for i in range(steps):
        if remaining == 0:
            break
        k = min(int(round(remaining * (1.0 / (steps - i)))), remaining)
        if k > 0:
            ks.append(k)
            remaining -= k
    return ks


def softmax_conf(logits_row, token):
    m = logits_row.max()
    e = np.exp(logits_row - m)
    return float(e[token] / e.sum())


def bd3lm_generate(sess, tok, prompt_ids, max_new_tokens, steps, block_size, verbose=True):
    mask_id = tok.mask_token_id
    eos_id = tok.eos_token_id
    P = len(prompt_ids)
    T = P + max_new_tokens
    x = np.full((1, T), eos_id, dtype=np.int64)
    x[0, :P] = prompt_ids
    x[0, P:] = mask_id
    attn = block_causal_mask(T, block_size)             # static, reused every forward

    first_block = P // block_size                        # block containing the first gen pos
    last_block = (T - 1) // block_size
    gen_blocks = last_block - first_block + 1
    steps_per_block = -(-steps // gen_blocks)

    n_forwards = 0
    t0 = time.time()
    for blk in range(first_block, last_block + 1):
        start = blk * block_size
        end = min(start + block_size, T)
        n_masked = int((x[0, start:end] == mask_id).sum())
        if n_masked == 0:
            continue
        for k in transfer_schedule(n_masked, steps_per_block):
            logits = sess.run(["logits"], {"input_ids": x, "attention_mask": attn})[0][0]  # [T,V]
            n_forwards += 1
            mask_pos = np.where(x[0] == mask_id)[0]
            mask_pos = mask_pos[(mask_pos >= start) & (mask_pos < end)]
            if len(mask_pos) == 0:
                break
            preds = logits[mask_pos].argmax(-1)
            confs = np.array([softmax_conf(logits[p], t) for p, t in zip(mask_pos, preds)])
            commit = mask_pos[np.argsort(-confs)[:k]]
            x[0, commit] = logits[commit].argmax(-1)
            if verbose:
                done = int((x[0, P:] != mask_id).sum())
                print(f"\r  block {blk - first_block + 1}/{gen_blocks} "
                      f"forwards={n_forwards} revealed={done}/{max_new_tokens}", end="")
    dt = time.time() - t0
    if verbose:
        print(f"\n  {n_forwards} forwards in {dt:.1f}s ({dt / n_forwards:.2f}s/forward)")

    gen = x[0, P:].tolist()
    if eos_id in gen:
        gen = gen[: gen.index(eos_id)]
    return tok.decode(gen, skip_special_tokens=True), dt, n_forwards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-0.6b-bd3lm-onnx/model_fp32_fused.onnx")
    ap.add_argument("--prompt", default="Lily runs 12 km/h for 4 hours. How far in 8 hours?")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, tokenize=True,
    )
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    print(f"model={args.model}  mask_id={tok.mask_token_id}  eos_id={tok.eos_token_id}  "
          f"prompt_tokens={len(prompt_ids)}")
    text, dt, nf = bd3lm_generate(
        sess, tok, prompt_ids, args.max_new_tokens, args.steps, args.block_size
    )
    print("-" * 60)
    print(text)


if __name__ == "__main__":
    main()
