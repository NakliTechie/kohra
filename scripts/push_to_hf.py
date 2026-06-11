"""Publish the fused-fp16 ONNX export (+ tokenizer + model card) to the Hugging Face Hub.

The fused-fp16 graph is the in-browser deliverable: it runs on WebGPU at ~9.8 tok/s
(the unfused fp16 graph silently zeros on WebGPU — see reference/MDLM-algorithm.md).

Usage:
  .venv/bin/python scripts/push_to_hf.py --stage meta     # repo + tokenizer + card + graph
  .venv/bin/python scripts/push_to_hf.py --stage weights  # the 1.4GB external-data blob
"""

import argparse
import glob
import os

from huggingface_hub import HfApi, upload_file

REPO_ID = "naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX"
HERE = os.path.dirname(__file__)
MODELS = os.path.join(HERE, "..", "models", "qwen3-0.6b-mdlm-onnx")
GRAPH = os.path.join(MODELS, "model_fp16_fused.onnx")
DATA = os.path.join(MODELS, "model_fp16_fused.onnx.data")

# tokenizer files from the source model's local HF cache snapshot
CACHE_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1/snapshots/*/"
)
TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "chat_template.jinja",
]

MODEL_CARD = """---
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
(`web/kohra.js`), a small transformers.js-style module.

## Use it

Grab the loader (`kohra.js`, one file — it's in this repo, or in the
[kohra project](https://github.com/NakliTechie/kohra) at `web/kohra.js`) and import it locally;
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
- q4f16 (~500 MB) exists and is correct on CPU, but ORT-web's WebGPU `MatMulNBits` kernel
  miscomputes it on Apple GPUs — parked pending an upstream fix.

## Attribution & license

Derivative of `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` (built on `Qwen/Qwen3-0.6B`, Apache-2.0;
MDLM adaptation by [dLLM](https://github.com/ZHZisZZ/dllm)). See the source repos for exact terms.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["meta", "weights"], required=True)
    args = ap.parse_args()

    api = HfApi()
    snap_dirs = glob.glob(CACHE_GLOB)
    snap = snap_dirs[0] if snap_dirs else None

    if args.stage == "meta":
        api.create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)
        print(f"repo ready: {REPO_ID}")

        upload_file(path_or_fileobj=MODEL_CARD.encode(), path_in_repo="README.md",
                    repo_id=REPO_ID, commit_message="model card")
        print("uploaded README.md (model card)")

        for name in TOKENIZER_FILES:
            src = os.path.join(snap, name) if snap else None
            if src and os.path.exists(src):
                upload_file(path_or_fileobj=src, path_in_repo=name, repo_id=REPO_ID,
                            commit_message=f"tokenizer: {name}")
                print(f"uploaded {name}")
            else:
                print(f"SKIP {name} (not in cache)")

        upload_file(path_or_fileobj=GRAPH, path_in_repo="onnx/model_fp16_fused.onnx",
                    repo_id=REPO_ID, commit_message="onnx graph (fused fp16)")
        print("uploaded onnx/model_fp16_fused.onnx")

        # bundle the loader so the repo is self-contained (copy this file into your app)
        kohra_js = os.path.join(HERE, "..", "web", "kohra.js")
        upload_file(path_or_fileobj=kohra_js, path_in_repo="kohra.js",
                    repo_id=REPO_ID, commit_message="bundle the kohra.js loader")
        print("uploaded kohra.js")
        print("META DONE — now run: --stage weights")

    elif args.stage == "weights":
        sz = os.path.getsize(DATA) / 1e9
        print(f"uploading external data ({sz:.1f} GB) — this takes a while…")
        upload_file(path_or_fileobj=DATA, path_in_repo="onnx/model_fp16_fused.onnx.data",
                    repo_id=REPO_ID, commit_message="onnx external data (fused fp16)")
        print("WEIGHTS DONE")
        print(f"https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
