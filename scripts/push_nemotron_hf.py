"""Publish the Nemotron-Labs-Diffusion-3B q4f16 block-KV graph to the Hugging Face Hub.

The repo carries the NVIDIA Nemotron Open Model License (redistribution §3a: recipients get a
copy), a NOTICE with the §3c attribution statement, the tokenizer and config from the source
checkpoint, the q4f16 graph with its two external-data files, and kohra.js.

  .venv-nemotron/bin/python scripts/push_nemotron_hf.py
"""

import os

from huggingface_hub import HfApi

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
REPO_ID = "naklitechie/Nemotron-Labs-Diffusion-3B-ONNX"
SRC = os.environ.get("NEMOTRON_SRC", "/Volumes/Models2/Models/diffusion/nemotron-labs-diffusion-3b")
ONNX = os.environ.get("NEMOTRON_OUT", "/Volumes/Models2/Models/diffusion/nemotron-labs-diffusion-3b-onnx")
GRAPH = "model_kv_q4f16.onnx"
SOURCE_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "chat_template.jinja", "config.json", "generation_config.json"]

NOTICE = """This repository contains a derivative work of nvidia/Nemotron-Labs-Diffusion-3B
(https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B): the model exported to ONNX and quantised
to 4-bit weights (q4f16) by NakliTechie for kohra (https://github.com/NakliTechie/kohra).

Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License.

The full licence text is in LICENSE.
"""

CARD = """---
license: other
license_name: nvidia-nemotron-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
base_model: nvidia/Nemotron-Labs-Diffusion-3B
library_name: onnxruntime-web
pipeline_tag: text-generation
tags:
- text-diffusion
- diffusion-lm
- block-diffusion
- webgpu
- onnx
- browser
---

# Nemotron-Labs-Diffusion-3B · ONNX q4f16 for the browser

[nvidia/Nemotron-Labs-Diffusion-3B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B) exported to one
ONNX graph that runs on WebGPU with [kohra](https://github.com/NakliTechie/kohra), in-browser text diffusion.
Try it in the [kohra demo](https://naklitechie.github.io/kohra/?arch=nemotron-q4).

**Size:** 2.73 GB of weights in two files (`onnx/model_kv_q4f16.onnx_data` 2.00 GB, `_data_1` 0.74 GB).
4-bit RTN weights (MatMulNBits, symmetric, block 32), fp16 elsewhere, embeddings fp16.

## How the graph decodes

The model writes 32-token blocks against a KV cache, as its own `generate` does. One graph serves both passes,
selected by an additive `attention_mask` input of shape `[1, 1, L, past + L]`:

- **Denoise** a block: an all-zero mask, so the block sees the whole cache and itself.
- **Commit** a finished block: a causal mask; keep `present_*` as the new cache. The argmax of the last
  logit seeds the first token of the next block.

Inputs: `input_ids [1, L]`, `position_ids [1, L]` (absolute), `attention_mask`, `past_key_i` / `past_value_i`
`[1, 8, past, 128]` for 26 layers. Outputs: `logits [1, L, 131072]` fp32, `present_key_i` / `present_value_i`.
Mask token id 100, end-of-turn id 11.

```js
import { DiffusionLM } from 'https://cdn.jsdelivr.net/gh/NakliTechie/kohra@main/kohra.js';
const R = 'https://huggingface.co/naklitechie/Nemotron-Labs-Diffusion-3B-ONNX/resolve/main';
const lm = await DiffusionLM.from_pretrained({
  model: `${R}/onnx/model_kv_q4f16.onnx`, tokenizer: 'naklitechie/Nemotron-Labs-Diffusion-3B-ONNX',
  kvCache: true, maskId: 100n, eosId: 11n,
  chatFormat: (t) => `<|im_start|>system\\n<|im_end|>\\n<|im_start|>user\\n${t}<|im_end|>\\n<|im_start|>assistant\\n<think></think>`,
});
const { text } = await lm.generate('List three prime numbers greater than 50.', { maxNewTokens: 128, threshold: 0.9 });
```

## Checks

- Export wrapper vs the upstream PyTorch forward: identical logits and cache in both modes (bf16, max difference 0.0).
- On a 66-token real text, the q4 graph picks the same next token as the bf16 model at 83% of positions; next-token
  accuracy 0.431 (bf16: 0.385).
- WebGPU in Chrome on an M4 Pro: six generations coherent; "three primes greater than 50" → 53, 59, 61 in 34
  forward passes with `threshold: 0.9`.
- Not yet compared token for token with upstream `generate`.

Export script: [`scripts/export_nemotron_kv.py`](https://github.com/NakliTechie/kohra/blob/main/scripts/export_nemotron_kv.py).

## Licence

Use is governed by the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/);
a copy is in `LICENSE`. Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License.
"""


def main():
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    ops = [("README.md", CARD.encode()), ("NOTICE", NOTICE.encode())]
    for path_in_repo, data in ops:
        api.upload_file(path_or_fileobj=data, path_in_repo=path_in_repo, repo_id=REPO_ID)
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "reference", "nemotron", "LICENSE"),
                    path_in_repo="LICENSE", repo_id=REPO_ID)
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "kohra.js"), path_in_repo="kohra.js", repo_id=REPO_ID)
    for f in SOURCE_FILES:
        api.upload_file(path_or_fileobj=os.path.join(SRC, f), path_in_repo=f, repo_id=REPO_ID)
    graph_files = sorted(f for f in os.listdir(ONNX) if f.startswith(GRAPH))
    for f in graph_files:
        print(f"uploading onnx/{f} ({os.path.getsize(os.path.join(ONNX, f)) / 1e9:.2f} GB)…")
        api.upload_file(path_or_fileobj=os.path.join(ONNX, f), path_in_repo=f"onnx/{f}", repo_id=REPO_ID)
    print(f"done: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
