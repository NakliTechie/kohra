"""Reference generations from Nemotron-Labs-Diffusion-3B's own `generate` (torch bf16, CPU).

kohra's seeded KV path in kohra.js reimplements that loop (causal prefill and commit, zero-mask
denoise, next-block seed). This script records what upstream produces on the same prompts, so the
WebGPU run can be compared text to text. Greedy (temperature 0), block 32.

  .venv-nemotron/bin/python scripts/gencheck_nemotron.py [out.json]
"""

import json
import os
import sys
import time

import torch
import transformers

SRC = os.environ.get("NEMOTRON_SRC", "/Volumes/Models2/Models/diffusion/nemotron-labs-diffusion-3b")
PROMPTS = [
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. "
    "How many kilometers can she run in 8 hours?",
    "Write one short paragraph explaining why the sky is blue.",
    "List three prime numbers greater than 50.",
]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "nemotron-gencheck.json"
    tok = transformers.AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
    model = transformers.AutoModel.from_pretrained(
        SRC, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager").eval()
    rows = []
    for prompt in PROMPTS:
        s = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                    add_generation_prompt=True)
        ids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids
        for threshold in (None, 0.9):
            t0 = time.time()
            with torch.no_grad():
                out, nfe = model.generate(ids, max_new_tokens=128, block_length=32, threshold=threshold,
                                          eos_token_id=tok.eos_token_id)
            gen = out[0, ids.shape[1]:].tolist()
            if tok.eos_token_id in gen:
                gen = gen[: gen.index(tok.eos_token_id)]
            text = tok.decode(gen, skip_special_tokens=True)
            rows.append(dict(prompt=prompt, threshold=threshold, nfe=nfe, seconds=round(time.time() - t0, 1),
                             token_ids=gen, text=text))
            print(f"--- threshold={threshold} nfe={nfe} {time.time() - t0:.0f}s\n{text}\n")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
