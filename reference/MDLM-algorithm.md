# MDLM sampling — the spec the JS loop is written against

Source: `vendor/dllm` (`dllm/core/samplers/mdlm.py`, Apache-2.0), which itself follows
[LLaDA's generate.py](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py). Verified against
`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` (= `dllm-collection/…`, same weights).

## Model facts (Tiny-A2D Qwen3-0.6B MDLM)

| Fact | Value |
|---|---|
| Architecture | `A2DQwen3LMHeadModel` — stock Qwen3 0.6B, **one delta**: causal mask replaced by padding-only bidirectional mask (`_prepare_4d_attention_mask`) |
| Layers | 28, all `full_attention` (no sliding) · hidden 1024 · head_dim 128 · GQA 16/8 |
| KV cache | **None** — every denoise step is a fresh full forward |
| Mask token | `<|mask|>` = **151669** (`tokenizer.mask_token_id`; fits in Qwen3's padded vocab, no resize) |
| BOS / PAD | `<|endoftext|>` = 151643 |
| EOS | `<|im_end|>` = 151645 |
| Chat template | stock Qwen3 ChatML (`<|im_start|>role\n…<|im_end|>`) |
| Weights | bf16 safetensors, ~1.4GB; vocab 151936 |
| Training | SFT-only (tulu-3 + smoltalk + OpenCoder-python), 1024 max_length, from `Qwen/Qwen3-0.6B` non-causal conversion |
| Quality bar | GSM8K 29.3 · MMLU 40.0 · HumanEval 30.5 (BD3LM sibling: GSM8K **46.3**, HumanEval **46.3**) |

Loading note: the HF repo bundles `modeling_qwen3.py` via `auto_map` (`AutoModelForMaskedLM`);
the dllm package also registers the classes natively — either path works.

## Reference sampler defaults (examples/a2d/mdlm/sample.py)

```
steps=128  max_new_tokens=128  block_size=32  temperature=0.0
remasking="low_confidence"  right_shift_logits=False  cfg_scale=0.0
```
Eval harness used: `steps=256 max_new_tokens=256 block_size=256` (single block).

## The algorithm (sample(), batch B; B=1 in the browser)

### Setup

1. Tokenize prompt with chat template + generation prompt → `prompt` (length `P`).
2. Canvas `x`: shape `[B, T]`, `T = P + max_new_tokens`. Fill with **EOS (151645)**;
   copy prompt into `x[:P]`; set `x[P : P+max_new_tokens] = MASK (151669)`.
3. `attention_mask = 1` over `[0, P+max_new_tokens)` (all of it, both prompt and masked tail).
   With B=1 and no padding the mask is all-ones → can be omitted entirely in the export.

### Block schedule

```
num_blocks       = ceil(max_new_tokens / block_size)
steps_per_block  = ceil(steps / num_blocks)
```
Blocks are processed **left to right**; block `b` covers canvas positions
`[P + b·block_size, P + (b+1)·block_size)`.

### Per-block transfer budget — get_num_transfer_tokens

With the default **LinearAlphaScheduler** (α(t) = 1−t), the reverse-transfer probability for
step i of S is `1/(S−i)` … which works out to **near-equal splits**: `n_masked/S` tokens
revealed per step, deterministically rounded (`round()`), clamped by remaining masks. Rows of
zero transfers are compacted out (steps with nothing to do are skipped — "no time conditioning"
means skipping steps is sound).

JS port: precompute the per-step reveal counts `[k_0 … k_{S−1}]` for each block with the same
round-and-subtract loop; skip zero entries.

### Per-step loop (inside block b, step i)

1. `logits = model(x)` — **full forward over the whole canvas** (no attention_mask needed at B=1).
2. (skip CFG, suppress_tokens, right_shift_logits — all off at defaults)
3. Temperature: `t=0` → plain argmax. Else **Gumbel-max in float64**:
   `noise = rand_like(logits); sampled = exp(logits) / (−ln(noise))^t; argmax(sampled)`.
   (Reference insists on f64 for quality — JS numbers are f64 natively, fine; but f32 logits
   from WebGPU must be upcast before exp.)
4. `x0 = argmax(...)` over vocab at every position.
5. Confidence (`remasking="low_confidence"`): `conf = softmax(logits)[x0]` — probability of the
   argmax token, per position. (`"random"`: conf = U(0,1).)
6. Restrict: positions beyond the current block's end get conf = −∞;
   positions not currently masked get conf = −∞ (and keep their token).
7. Commit: take top-`k_i` positions by confidence, write `x0` there (unmask them).
8. Repeat until block exhausted; move to next block.

### Termination / trimming

No early stopping in the loop. After all blocks: find first **EOS at/after P**, cut there,
decode skipping specials (`sample_trim`). The canvas tail past EOS is typically EOS-filled
(model learned to pad with EOS).

UX note for the browser: histories (canvas snapshot per step) make a great
"fog lifting" visualization — the reference TerminalVisualizer does exactly this.

## ONNX export implications

- Export = **stock Qwen3 forward with a bidirectional (or no) attention mask, no KV cache,
  logits over full sequence**. Inputs: `input_ids [1, T]` (+ optional `attention_mask`).
  Output: `logits [1, T, vocab]`.
- Dynamic `T` (canvas width varies with prompt length); position_ids implicit (0..T−1).
- The whole denoising loop lives **outside** the graph, in JS — model graph is loop-free.
- Only logits at currently-masked positions are consumed, but the graph returns all
  (a gather-at-masked-positions optimization can come later; vocab 151936 × T floats is the
  transfer cost — at T≈200 ≈ 120MB f32, so **do** consider returning only masked-position
  logits or top-k in a v2 graph).
- bf16 weights → fp16 or q4f16 for ORT-web. No MoE in this model — the symmetric-QMoE
  constraint only bites at G3 (LLaDA-MoE); dense MatMulNBits may use asymmetric.

### Export pipeline results (2026-06-10, scripts/export_onnx.py)

- **fp32**: legacy TorchScript exporter, eager attention, `use_cache=False`, opset 17,
  dynamic axes verified at a non-trace length. Parity: max|Δlogit|=0.0001, argmax 1.0000.
- **fp16**: `onnxruntime.transformers.OnnxModel.convert_float_to_float16(keep_io_types=True)`.
  (`onnxconverter-common` emits type-broken graphs on this model — don't use it.)
  Parity: max|Δlogit|=0.098, masked-argmax 0.969. **The G1 artifact** (~1.5GB).
- **Gotcha:** ORT's `SimplifiedLayerNormFusion` crashes at session init on the converter's
  inserted casts → create sessions with graph optimizations **disabled** (Python:
  `ORT_DISABLE_ALL`; ORT-web: `graphOptimizationLevel: 'disabled'`).
- **q4f16: DEFERRED.** MatMulNBits over the cast-littered fp16 graph only shrinks 1.5→1.3GB
  and wrecks parity (masked-argmax 0.78 asym / 0.31 sym). Right recipe to try next:
  quantize the *fp32* graph first, then fp16-convert the remainder; or reuse
  onnx-community's conversion scripts; consider excluding `lm_head`.
- **Full-generation check (the real gate):** `scripts/sample_onnx.py` (numpy minimal loop,
  fp32 ONNX, CPU EP) on the math prompt → coherent AND arithmetically correct
  (`\boxed{96}`), 128 forwards in 24.9s (0.19 s/forward — ORT CPU fp32 is ~13× faster than
  PyTorch CPU bf16 here).
- Cosmetic: outputs open with an empty Qwen3 `<think>\n\n</think>` block (template artifact) —
  strip in UI or pass `enable_thinking: false` if supported.

## Perf model

Cost ≈ `steps × full-forward(T)`. No KV cache to win back; WebGPU batch throughput is the
whole game. Defaults: 128 forwards of T≈160–200 for 128 generated tokens — i.e., ~1 forward
per token, but each forward is parallel over all positions (vs AR's sequential 128 forwards
of width 1 + cache). Step-skipping (zero-transfer compaction) and fewer-steps configs
(steps < max_new_tokens) are the speed lever: steps=64/block to halve forwards at some
quality cost. Fast-dLLM-style confidence-threshold decoding is the G2 upgrade.

## Observed reference run (this machine, CPU bf16, 2026-06-10)

`examples/a2d/mdlm/sample.py` at defaults (steps=128, max_new=128, block=32, temp=0, batch=2),
~6 min wall on CPU (≈2–3 s/forward at batch 2 — CPU bf16 is slow; this is what WebGPU must beat).

- **Math prompt** ("Lily runs 12 km/h for 4 hours. How far in 8 hours?"): fluent, perfectly
  formatted chain-of-thought ending in `\boxed{344}` — structurally coherent, arithmetically
  wrong (twice). Consistent with GSM8K 29.3 at 0.6B.
- **Code prompt** ("write an educational python function"): clean correct `square()` function
  with docstring-style prose + usage example. Genuinely good.

**G1 quality bar set:** fluent multi-sentence instruct-following with correct format =
"coherent". Arithmetic correctness is NOT the bar (use the bd3lm sibling or G3 model for that).
