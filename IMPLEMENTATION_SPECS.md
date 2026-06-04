# DB9-Toolkit-Trainner - Implementation Specifications

## Purpose

Implement and maintain a local notebook-based Flux LoRA training toolkit for DB9 workflows.

## Canonical Names

- Project display name: `DB9-Toolkit-Trainner`
- Config/project id: `db9_toolkit_trainner`
- Repository: `https://github.com/bemapa3/DB9-TrainLORA`
- Former folder/name: `DB9-Local-Trainer`

## Default Model

`black-forest-labs/FLUX.2-klein-base-9B`

## Core Scripts

### `scripts/config_generator.py`

Generates ai-toolkit YAML from notebook parameters.

Expected config properties:

- `job: extension`
- process type `sd_trainer`
- Flux model flag `is_flux: true`
- dataset image path `{dataset_path}/img`
- caption path `{dataset_path}/captions`
- `noise_scheduler: flowmatch`
- sample sampler `flowmatch`
- sample prompts stored as a YAML list

### `scripts/data_clean.py`

Cleans image datasets by validating images, checking min/max resolution, converting to RGB, and writing to `datasets/processed/img`.

### `scripts/caption_gemini.py`

Generates captions with Gemini. It must require either `--api-key` or `GEMINI_API_KEY` from the environment.

### `scripts/caption_florence.py`

Generates local captions with Florence-2. It should remain optional because it may download a large model and require GPU memory.

### `scripts/caption_utils.py`

Post-processes `.txt` caption files. The expected input directory is `datasets/processed/captions`.

### `scripts/degrade_images.py`

Creates degraded control images for paired upscale/img2img training. It validates degradation parameters before processing.

## Notebooks

### `1_Setup.ipynb`

Must be safe to rerun. It should not fail just because `ai-toolkit` already exists.

### `2_Prepare_Dataset.ipynb`

Must fail early with a clear message when Gemini captioning is selected but `GEMINI_API_KEY` is missing.

### `3_Train_Config.ipynb`

Must generate `configs/db9_toolkit_trainner.yaml` using Flux 2 Klein defaults.

### `4_Train.ipynb`

Must check for the generated config and `ai-toolkit/run.py` before invoking training.

### `5_Test_LoRA.ipynb`

Must discover the newest `.safetensors` checkpoint instead of assuming a single hard-coded filename.

## Verification

Run:

```bash
python -m compileall -q scripts verify.py create_notebooks.py
python create_notebooks.py
python verify.py
```

These commands validate Python syntax, regenerate notebooks, and check project/model/workflow conventions. They do not prove training succeeds on GPU.

## Agent Notes

Agents should preserve the canonical name `DB9-Toolkit-Trainner` and keep generated notebooks in sync with `create_notebooks.py`.
