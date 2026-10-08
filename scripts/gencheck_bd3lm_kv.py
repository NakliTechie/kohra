"""End-to-end check of the KV-cache BD3LM ONNX graph: generate with it (onnxruntime, CPU)
and compare token ids against the cache-free torch sampler from kv_reference.py.

  .venv/bin/python scripts/gencheck_bd3lm_kv.py [--model models/qwen3-0.6b-bd3lm-onnx/model_kv_fp16_fused.onnx]
"""

import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
import transformers

sys.path.insert(0, os.path.dirname(__file__))
from kv_reference import BLOCK, MODEL_ID, cache_free, load_a2d, schedule  # noqa: E402

N_LAYERS, KV_HEADS, HEAD_DIM = 28, 8, 128


def kv_generate(sess, prompt, n_new, steps, mask_id, eos_id):
    fdt = np.float16 if "float16" in sess.get_inputs()[2].type else np.float32
    P, T = len(prompt), len(prompt) + n_new
    x = np.full(T, eos_id, dtype=np.int64)
    x[:P] = prompt
    x[P:] = mask_id
    blocks = list(range(P // BLOCK, (T - 1) // BLOCK + 1))
    spb = -(-steps // len(blocks))
    past = [np.zeros((1, KV_HEADS, 0, HEAD_DIM), dtype=fdt) for _ in range(2 * N_LAYERS)]
    names = [i.name for i in sess.get_inputs()[2:]]
    fwd = 0

    def run(s, e, past):
        feeds = {"input_ids": x[s:e].reshape(1, -1),
                 "position_ids": np.arange(s, e, dtype=np.int64).reshape(1, -1),
                 **dict(zip(names, past))}
        out = sess.run(None, feeds)
        return out[0][0], out[1:]

    if blocks[0] > 0:                                   # prefill complete prompt blocks
        _, past = run(0, blocks[0] * BLOCK, past)
        fwd += 1
    for b in blocks:
        s, e = b * BLOCK, min((b + 1) * BLOCK, T)
        for k in schedule(int((x[s:e] == mask_id).sum()), spb):
            logits, _ = run(s, e, past)                 # denoise: present is discarded
            fwd += 1
            cand = []
            for i, p in enumerate(range(s, e)):
                if x[p] != mask_id:
                    continue
                row = logits[i].astype(np.float64)
                tok = int(row.argmax())
                conf = 1.0 / np.exp(row - row[tok]).sum()
                cand.append((conf, p, tok))
            cand.sort(key=lambda c: -c[0])
            for _, p, tok in cand[:k]:
                x[p] = tok
        _, past = run(s, e, past)                       # commit the finished block
        fwd += 1
    return x[P:], fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-0.6b-bd3lm-onnx/model_kv_fp16_fused.onnx")
    args = ap.parse_args()
    load_a2d()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager").eval()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    m, eos = tok.mask_token_id, tok.eos_token_id
    for text in ["Lily runs 12 km/h for 4 hours. How far in 8 hours?",
                 "Explain in three sentences why the sky is blue.",
                 "Hi"]:                                   # short prompt: first block has no cache
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, tokenize=True)
        ref, _ = cache_free(model, ids, 128, 128, m, eos)
        t0 = time.time()
        got, fwd = kv_generate(sess, ids, 128, 128, m, eos)
        dt = time.time() - t0
        ref = ref.numpy()
        same = int((ref == got).sum())
        cut = lambda t: t[: list(t).index(eos)] if eos in t else t
        print(f"\n=== {text}  (prompt {len(ids)})  onnx-kv: {fwd} fwd {dt:.1f}s  "
              f"matching ids vs torch cache-free: {same}/128")
        print("  ", tok.decode(cut(got), skip_special_tokens=True)[:240].replace("\n", " / "))


if __name__ == "__main__":
    main()
