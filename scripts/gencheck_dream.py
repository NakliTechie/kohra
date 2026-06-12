"""Coherent-prose gate for the Dream-7B ONNX export (run on the Studio before pulling back).

Numpy MDLM-style denoising loop (Dream is masked diffusion, bidirectional, no cache) on the
exported ONNX, CPU EP. Mask 151666. Coherent output ⇒ the export + fuse + q4 are sound.

  .venv/bin/python scripts/gencheck_dream.py --model models/dream-7b-onnx/model_q4f16_rtn_sym.onnx
"""

import argparse
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666
EOS_ID = 151643


def transfer_schedule(n_masked, steps):
    ks, rem = [], n_masked
    for i in range(steps):
        if rem == 0:
            break
        k = min(int(round(rem / (steps - i))), rem)
        if k > 0:
            ks.append(k); rem -= k
    return ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Lily runs 12 km/h for 4 hours. How far in 8 hours?")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                  add_generation_prompt=True, tokenize=True)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])

    P = len(ids); T = P + args.max_new_tokens
    x = np.full((1, T), EOS_ID, dtype=np.int64); x[0, :P] = ids; x[0, P:] = MASK_ID
    nb = -(-args.max_new_tokens // args.block_size)
    spb = -(-args.steps // nb)
    t0 = time.time(); nf = 0
    for b in range(nb):
        s, e = P + b * args.block_size, min(P + (b + 1) * args.block_size, T)
        nm = int((x[0, s:e] == MASK_ID).sum())
        for k in transfer_schedule(nm, spb):
            logits = sess.run(["logits"], {"input_ids": x})[0][0]; nf += 1
            mp = np.where(x[0] == MASK_ID)[0]; mp = mp[(mp >= s) & (mp < e)]
            preds = logits[mp].argmax(-1)
            conf = np.array([np.exp(logits[p] - logits[p].max())[t] /
                             np.exp(logits[p] - logits[p].max()).sum() for p, t in zip(mp, preds)])
            commit = mp[np.argsort(-conf)[:k]]
            x[0, commit] = logits[commit].argmax(-1)
    dt = time.time() - t0
    raw = x[0, P:].tolist()
    print(f"{nf} forwards in {dt:.1f}s ({dt/nf:.2f}s/fwd, CPU)")
    print("DEBUG prompt_len P =", P, "| first 12 raw gen ids:", raw[:12])
    print("DEBUG decode(raw, keep specials):", repr(tok.decode(raw, skip_special_tokens=False)[:200]))
    gen = raw[:raw.index(EOS_ID)] if EOS_ID in raw else raw
    print("-" * 60)
    print(tok.decode(gen, skip_special_tokens=True))


if __name__ == "__main__":
    main()
