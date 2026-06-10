"""Minimal MDLM denoising loop on the ONNX export — the direct template for the JS port.

Pure numpy + onnxruntime; no torch, no dllm. Mirrors dllm's MDLMSampler at defaults:
linear scheduler, low_confidence remasking, temperature 0, no CFG.

Usage:
  .venv/bin/python scripts/sample_onnx.py --model models/qwen3-0.6b-mdlm-onnx/model_fp32.onnx \
      --prompt "Lily runs 12 km/h for 4 hours. How far in 8 hours?" \
      --max-new-tokens 128 --steps 128 --block-size 32
"""

import argparse
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_ID = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
MASK_ID = 151669
EOS_ID = 151645


def transfer_schedule(n_masked: int, steps: int) -> list[int]:
    """Per-step reveal counts for one block (linear alpha scheduler, deterministic).

    Mirrors dllm get_num_transfer_tokens: at step i of S (i=0..S-1),
    reverse_transfer_prob = 1 - (1 - s/S)/(1 - t/S) with t=S-i, s=S-i-1
    => k_i = round(remaining * 1/(S-i)); zero entries are skipped.
    """
    ks = []
    remaining = n_masked
    for i in range(steps):
        if remaining == 0:
            break
        p = 1.0 / (steps - i)
        k = int(round(remaining * p))
        k = min(k, remaining)
        if k > 0:
            ks.append(k)
            remaining -= k
    return ks


def softmax_conf(logits_row: np.ndarray, token: int) -> float:
    """P(token) under softmax of one [vocab] row, computed stably."""
    m = logits_row.max()
    e = np.exp(logits_row - m)
    return float(e[token] / e.sum())


def mdlm_generate(sess, tokenizer, prompt_ids, max_new_tokens, steps, block_size, verbose=True):
    P = len(prompt_ids)
    T = P + max_new_tokens
    x = np.full((1, T), EOS_ID, dtype=np.int64)
    x[0, :P] = prompt_ids
    x[0, P:] = MASK_ID

    num_blocks = -(-max_new_tokens // block_size)
    steps_per_block = -(-steps // num_blocks)

    n_forwards = 0
    t0 = time.time()
    for b in range(num_blocks):
        start = P + b * block_size
        end = min(start + block_size, T)
        n_masked = int((x[0, start:end] == MASK_ID).sum())
        for k in transfer_schedule(n_masked, steps_per_block):
            logits = sess.run(["logits"], {"input_ids": x})[0][0]  # [T, vocab]
            n_forwards += 1
            mask_pos = np.where(x[0] == MASK_ID)[0]
            mask_pos = mask_pos[(mask_pos >= start) & (mask_pos < end)]
            preds = logits[mask_pos].argmax(-1)
            confs = np.array(
                [softmax_conf(logits[p], t) for p, t in zip(mask_pos, preds)]
            )
            commit = mask_pos[np.argsort(-confs)[:k]]
            x[0, commit] = logits[commit].argmax(-1)
            if verbose:
                done = int((x[0, P:] != MASK_ID).sum())
                print(f"\r  block {b + 1}/{num_blocks} forwards={n_forwards} revealed={done}/{max_new_tokens}", end="")
    dt = time.time() - t0
    if verbose:
        print(f"\n  {n_forwards} forwards in {dt:.1f}s ({dt / n_forwards:.2f}s/forward)")

    gen = x[0, P:].tolist()
    if EOS_ID in gen:
        gen = gen[: gen.index(EOS_ID)]
    return tokenizer.decode(gen, skip_special_tokens=True), dt, n_forwards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-0.6b-mdlm-onnx/model_fp32.onnx")
    ap.add_argument("--prompt", default="Lily runs 12 km/h for 4 hours. How far in 8 hours?")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=32)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, tokenize=True,
    )
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    print(f"prompt tokens: {len(prompt_ids)}")
    text, dt, nf = mdlm_generate(
        sess, tokenizer, prompt_ids, args.max_new_tokens, args.steps, args.block_size
    )
    print("-" * 60)
    print(text)


if __name__ == "__main__":
    main()
