# Flux 2 Klein Training Status

Date: 2026-06-05

## Fixed in repository

- Config generator now emits `use_text_encoder_2: false` for Flux 2 Klein so ai-toolkit does not use the default second text encoder path.
- DB9Studio UI resolves `black-forest-labs/FLUX.2-klein-base-9B` to `models/FLUX.2-klein-base-9B` when a local snapshot exists.
- DB9Studio UI accordion now has a separate `2.1c Size-degrade pairs` panel title; the Train panel is no longer shifted into the wrong title.
- Verification now checks the Flux 2 Klein local model fallback and the 2.1c UI title.

## Runtime fixes tested on the 5090 machine

These edits were made directly in `D:\BBB\DB9-Local-Trainer\ai-toolkit\toolkit\stable_diffusion_model.py` during debugging. They must be kept if ai-toolkit is refreshed:

- Guarded T5/text_encoder_2 loading behind `self.model_config.use_text_encoder_2`.
- Flux 2 Klein path uses Qwen tokenizer/text encoder instead of CLIP:
  - `Qwen2TokenizerFast`
  - `Qwen3ForCausalLM`
  - `Flux2KleinPipeline`
  - `AutoencoderKLFlux2`
- Guarded optional `text_encoder_2` preparation with `if text_encoder[1] is not None`.

## Error timeline

1. Hugging Face gated repo 401 on `scheduler/scheduler_config.json`.
   - Cause: remote HF access/token not visible to train process.
   - Fix: use full local model snapshot at `models/FLUX.2-klein-base-9B`.

2. ai-toolkit attempted to load `text_encoder_2`.
   - Cause: ai-toolkit default `use_text_encoder_2=True` and Flux loader ignored Flux 2 Klein layout.
   - Fix: YAML emits `use_text_encoder_2: false`; loader patched to respect it.

3. ai-toolkit attempted CLIP tokenizer/text encoder.
   - Cause: Flux 2 Klein model index uses Qwen, not CLIP.
   - Fix: loader patched toward Qwen/Flux2Klein classes.

4. Current remaining risk.
   - If the next traceback references transformer class/config/shape, patch the transformer loader from `FluxTransformer2DModel` to `Flux2Transformer2DModel` for Flux 2 Klein.

## Required local model rule

Always download and use the local Flux 2 Klein snapshot before training:

```text
models/FLUX.2-klein-base-9B/
  model_index.json
  scheduler/
  text_encoder/
  tokenizer/
  transformer/
  vae/
```

Generated configs should use a local `name_or_path` when that folder exists.
## Latest progress update

- Added repository-side local model resolution so UI-generated configs prefer `models/FLUX.2-klein-base-9B` when present.
- Added dataset filename validation with `BAD_NAME` logs for missing captions, missing controls, orphan files, and unsafe stems.
- Added a UI `Validate Dataset` button and automatic validation before config generation.
- Added README install/Jupyter/upscale optimization notes.

## Current active runtime issue

Training advanced past model loading and failed at Qwen3 prompt encoding with CUDA OOM on a 32GB GPU. Recommended next run settings:

- Tile/resolution: 1024 or 1536 before returning to 2048.
- Batch size: 1.
- Gradient accumulation: 4-8.
- Keep gradient checkpointing enabled.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before training.

If OOM persists, patch prompt/text encoder handling so Qwen prompt encoding is cached/offloaded rather than kept fully on GPU during training.
