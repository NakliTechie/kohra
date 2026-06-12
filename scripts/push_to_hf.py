"""Publish a fused-fp16 ONNX export (+ tokenizer + model card) to the Hugging Face Hub.

The fused-fp16 graph is the in-browser deliverable: it runs on WebGPU (the unfused fp16
graph silently zeros on WebGPU — see reference/MDLM-algorithm.md). Two models are
supported: `mdlm` (the original masked-diffusion checkpoint) and `bd3lm` (block diffusion;
a 2-input block-causal graph, stronger benchmarks).

Usage:
  .venv/bin/python scripts/push_to_hf.py --model mdlm  --stage meta     # repo + tokenizer + card + graph + kohra.js
  .venv/bin/python scripts/push_to_hf.py --model mdlm  --stage weights  # the fp16 external-data blob
  .venv/bin/python scripts/push_to_hf.py --model bd3lm --stage meta
  .venv/bin/python scripts/push_to_hf.py --model bd3lm --stage weights
  .venv/bin/python scripts/push_to_hf.py --model bd3lm --stage q4       # also publish the q4f16 (RTN) graph + data
  .venv/bin/python scripts/push_to_hf.py --stage space                  # (model-agnostic) the static demo Space
"""

import argparse
import glob
import os

from huggingface_hub import HfApi, upload_file

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
SPACE_ID = "naklitechie/kohra"

TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "chat_template.jinja",
]

SPACE_README = """---
title: kohra
emoji: 🌫️
colorFrom: indigo
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
short_description: Masked text-diffusion LM, fully in your browser (WebGPU)
---

# kohra — text diffusion in your browser

A masked-diffusion language model (Qwen3-0.6B-MDLM) generating text **client-side** on WebGPU —
no server inference. Open this Space in a WebGPU browser (Chrome/Edge 121+); first load pulls
~1.4 GB (cached after). Source + the `kohra.js` loader: https://github.com/NakliTechie/kohra
"""

MDLM_CARD = """---
license: apache-2.0
base_model: dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1
library_name: onnxruntime-web
tags:
- text-diffusion
- masked-diffusion
- diffusion-lm
- webgpu
- onnx
- browser
- kohra
pipeline_tag: text-generation
---

# Qwen3-0.6B-diffusion-mdlm-ONNX (fused fp16, for the browser)

ONNX export of [`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1)
(Tiny-A2D: Qwen3-0.6B adapted to **masked text diffusion**) packaged to run **client-side in a
web browser** on WebGPU via [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/). To our
knowledge this is the first *text*-diffusion LM whose denoising loop runs in-browser (image
diffusion in-browser is common; text diffusion has been server-side until now).

Generation is a JS denoising loop — start fully masked, run a full forward over the canvas, lock the
highest-confidence tokens, repeat — implemented in **[kohra](https://github.com/NakliTechie/kohra)**
(`kohra.js`), a small transformers.js-style module.

## Use it

Grab the loader (`kohra.js`, one file — it's in this repo, or in the
[kohra project](https://github.com/NakliTechie/kohra) at `kohra.js`) and import it locally;
the model + tokenizer load from this HF repo:

```js
import { pipeline } from './kohra.js';

const generate = await pipeline('text-diffusion', {
  model: 'https://huggingface.co/naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX/resolve/main/onnx/model_fp16_fused.onnx',
  tokenizer: 'naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX',
});

const { text, tokensPerSecond } = await generate('Lily runs 12 km/h for 4 hours. How far in 8 hours?');
console.log(text, `(${tokensPerSecond.toFixed(1)} tok/s)`);
```

Requires a WebGPU browser (Chrome/Edge 121+) served over https or localhost. ~1.4 GB download
(fp16, cached by the browser after first load); ~9.8 tok/s on an M-series Mac at 128 denoise steps.

## Files

- `onnx/model_fp16_fused.onnx` (+ `.onnx.data`) — fp16 weights with RMSNorm fused to
  `SimplifiedLayerNormalization`. The fusion is **required**: a decomposed `Pow(x,2)` RMSNorm
  overflows native fp16 on WebGPU and silently produces all-zero logits; the fused contrib op
  reduces in fp32. Recipe + forensics: the kohra repo's `reference/MDLM-algorithm.md`.
- Tokenizer files (Qwen3 ChatML).

## Notes

- **No KV cache** — bidirectional attention, one full forward per denoise step. Cost is
  steps × forward, not tokens; parallel block denoising is what plays to WebGPU.
- A block-diffusion sibling with stronger reasoning is at
  [`naklitechie/Qwen3-0.6B-diffusion-bd3lm-ONNX`](https://huggingface.co/naklitechie/Qwen3-0.6B-diffusion-bd3lm-ONNX).

## Attribution & license

Derivative of `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` (built on `Qwen/Qwen3-0.6B`, Apache-2.0;
MDLM adaptation by [dLLM](https://github.com/ZHZisZZ/dllm)). See the source repos for exact terms.
"""

BD3LM_CARD = """---
license: apache-2.0
base_model: dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1
library_name: onnxruntime-web
tags:
- text-diffusion
- block-diffusion
- diffusion-lm
- webgpu
- onnx
- browser
- kohra
pipeline_tag: text-generation
---

# Qwen3-0.6B-diffusion-bd3lm-ONNX (block diffusion, fused fp16, for the browser)

ONNX export of [`dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1)
— **BD3LM**, a *block-diffusion* language model — packaged to run **client-side in a web browser**
on WebGPU via [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/), driven by the
[kohra](https://github.com/NakliTechie/kohra) denoising loop (`kohra.js`).

Block diffusion sits between autoregressive and fully-masked (MDLM) generation: the sequence is
split into blocks of 32 tokens with a **block-causal** attention pattern — a block attends
bidirectionally within itself and causally to all earlier blocks — and is denoised block-by-block,
left to right. Same Tiny-A2D Qwen3-0.6B architecture as the MDLM sibling, but a separately-trained
checkpoint with **stronger reasoning**: GSM8K 46.3 vs 29.3, HumanEval 46.3 vs 30.5.

## Use it

Same `kohra.js` loader as the MDLM repo — block diffusion is one extra generate flag,
`blockCausal: true` (the graph takes a 2nd input, a block-causal attention mask, built for you):

```js
import { pipeline } from './kohra.js';

const generate = await pipeline('text-diffusion', {
  model: 'https://huggingface.co/naklitechie/Qwen3-0.6B-diffusion-bd3lm-ONNX/resolve/main/onnx/model_fp16_fused.onnx',
  tokenizer: 'naklitechie/Qwen3-0.6B-diffusion-bd3lm-ONNX',
});

const { text } = await generate('Lily runs 12 km/h for 4 hours. How far in 8 hours?', { blockCausal: true });
console.log(text);   // -> "...48 * 2 = 96 km. Thus, Lily runs \\boxed{96} km in 8 hours."
```

Requires a WebGPU browser (Chrome/Edge 121+) over https or localhost. ~1.5 GB download
(fp16, cached after first load); ~2.8 tok/s on an M-series Mac at 128 denoise steps.

## Files

- `onnx/model_fp16_fused.onnx` (+ `.onnx.data`) — **2-input** graph: `input_ids` and a
  `[1,1,T,T]` additive **block-causal** `attention_mask` (0 = attend, -1e9 = block, on the
  `pos // 32` grid). RMSNorm is fused to `SimplifiedLayerNormalization` (the WebGPU fp16 fix);
  attention stays decomposed so the 4D mask is honored.
- `onnx/model_q4f16_rtn_sym.onnx` (+ `.onnx.data`) — 4-bit (RTN, symmetric) q4f16, ~680 MB.
  Coherent on WebGPU; fp16 stays the default at 0.6B (q4's dequant overhead and a small quality
  dip aren't worth it until the model is too big for fp16).
- Tokenizer files (Qwen3 ChatML).

## Notes

- **No KV cache** — one full forward per denoise step over the whole canvas. Block-causality is
  what makes the cache-free forward correct: a still-masked future block can't influence an
  earlier one, so the current block's logits are right regardless of what's downstream.
- The loop relies on the model's default `arange` positions (the graph has no `position_ids`
  input), so the prompt sits at `[0,P)` with no padding and the first generated tokens complete
  the prompt's last partial block — exactly the configuration the ONNX-vs-torch parity verified.

## Attribution & license

Derivative of `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1` (built on `Qwen/Qwen3-0.6B`, Apache-2.0;
block-diffusion adaptation by [dLLM](https://github.com/ZHZisZZ/dllm)). See the source repos.
"""

CONFIGS = {
    "mdlm": {
        "repo_id": "naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX",
        "models_dir": os.path.join(ROOT, "models", "qwen3-0.6b-mdlm-onnx"),
        "cache_glob": os.path.expanduser(
            "~/.cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1/snapshots/*/"
        ),
        "card": MDLM_CARD,
        "q4": None,
    },
    "bd3lm": {
        "repo_id": "naklitechie/Qwen3-0.6B-diffusion-bd3lm-ONNX",
        "models_dir": os.path.join(ROOT, "models", "qwen3-0.6b-bd3lm-onnx"),
        "cache_glob": os.path.expanduser(
            "~/.cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-bd3lm-v0.1/snapshots/*/"
        ),
        "card": BD3LM_CARD,
        "q4": "model_q4f16_rtn_sym.onnx",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CONFIGS), default="mdlm")
    ap.add_argument("--stage", choices=["meta", "weights", "q4", "space"], required=True)
    args = ap.parse_args()

    api = HfApi()
    cfg = CONFIGS[args.model]
    repo_id = cfg["repo_id"]
    models_dir = cfg["models_dir"]
    graph = os.path.join(models_dir, "model_fp16_fused.onnx")
    data = graph + ".data"
    snap_dirs = glob.glob(cfg["cache_glob"])
    snap = snap_dirs[0] if snap_dirs else None

    if args.stage == "meta":
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
        print(f"repo ready: {repo_id}")

        upload_file(path_or_fileobj=cfg["card"].encode(), path_in_repo="README.md",
                    repo_id=repo_id, commit_message="model card")
        print("uploaded README.md (model card)")

        for name in TOKENIZER_FILES:
            src = os.path.join(snap, name) if snap else None
            if src and os.path.exists(src):
                upload_file(path_or_fileobj=src, path_in_repo=name, repo_id=repo_id,
                            commit_message=f"tokenizer: {name}")
                print(f"uploaded {name}")
            else:
                print(f"SKIP {name} (not in cache)")

        upload_file(path_or_fileobj=graph, path_in_repo="onnx/model_fp16_fused.onnx",
                    repo_id=repo_id, commit_message="onnx graph (fused fp16)")
        print("uploaded onnx/model_fp16_fused.onnx")

        # bundle the loader so the repo is self-contained (copy this file into your app)
        kohra_js = os.path.join(ROOT, "kohra.js")
        upload_file(path_or_fileobj=kohra_js, path_in_repo="kohra.js",
                    repo_id=repo_id, commit_message="bundle the kohra.js loader")
        print("uploaded kohra.js")
        print("META DONE — now run: --stage weights")

    elif args.stage == "weights":
        sz = os.path.getsize(data) / 1e9
        print(f"uploading external data ({sz:.1f} GB) — this takes a while…")
        upload_file(path_or_fileobj=data, path_in_repo="onnx/model_fp16_fused.onnx.data",
                    repo_id=repo_id, commit_message="onnx external data (fused fp16)")
        print("WEIGHTS DONE")
        print(f"https://huggingface.co/{repo_id}")

    elif args.stage == "q4":
        q4 = cfg["q4"]
        if not q4:
            print(f"no q4 graph configured for {args.model}"); return
        q4_graph = os.path.join(models_dir, q4)
        q4_data = q4_graph + ".data"
        sz = os.path.getsize(q4_data) / 1e6
        print(f"uploading q4 graph + external data ({sz:.0f} MB)…")
        upload_file(path_or_fileobj=q4_graph, path_in_repo=f"onnx/{q4}",
                    repo_id=repo_id, commit_message="onnx graph (q4f16 RTN)")
        upload_file(path_or_fileobj=q4_data, path_in_repo=f"onnx/{q4}.data",
                    repo_id=repo_id, commit_message="onnx external data (q4f16 RTN)")
        print("Q4 DONE")
        print(f"https://huggingface.co/{repo_id}")

    elif args.stage == "space":
        # Static demo Space. The GitHub Action keeps index.html + kohra.js in sync after this;
        # we seed it here so the Space works immediately.
        api.create_repo(SPACE_ID, repo_type="space", space_sdk="static",
                        exist_ok=True, private=False)
        print(f"space ready: {SPACE_ID}")
        upload_file(path_or_fileobj=SPACE_README.encode(), path_in_repo="README.md",
                    repo_id=SPACE_ID, repo_type="space", commit_message="space card")
        for name in ("index.html", "kohra.js"):
            upload_file(path_or_fileobj=os.path.join(ROOT, name), path_in_repo=name,
                        repo_id=SPACE_ID, repo_type="space", commit_message=f"seed {name}")
            print(f"uploaded {name}")
        print(f"SPACE DONE — https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
