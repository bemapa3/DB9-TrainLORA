# DB9-Toolkit-Trainner - Architecture Notes

## Goal

DB9-Toolkit-Trainner is a local Jupyter workflow for training DB9 Flux LoRA checkpoints with `ai-toolkit`.

The current implementation keeps the notebook-first workflow and focuses on:

- Local setup for Windows or desktop Python environments
- Dataset cleaning and caption generation
- ai-toolkit YAML generation
- Flux 2 Klein training defaults
- Simple checkpoint testing from the newest `.safetensors` output

## Current Repository Layout

```text
SDVN-Local-Trainer/
  notebooks/
    1_Setup.ipynb
    2_Prepare_Dataset.ipynb
    3_Train_Config.ipynb
    4_Train.ipynb
    5_Test_LoRA.ipynb
  scripts/
    caption_florence.py
    caption_gemini.py
    caption_utils.py
    config_generator.py
    data_clean.py
    degrade_images.py
  AGENTS.md
  README.md
  IMPLEMENTATION_SPECS.md
  create_notebooks.py
  requirements.txt
  verify.py
```

Runtime folders are created by the notebooks or user workflow:

```text
ai-toolkit/              # cloned by notebook 1
datasets/raw/           # user-provided source images
datasets/processed/img/
datasets/processed/captions/
configs/                # generated YAML configs
outputs/                # training outputs and tests
```

## Workflow

1. `1_Setup.ipynb`
   - Installs root requirements.
   - Clones `https://github.com/ostris/ai-toolkit.git` only if missing.
   - Updates ai-toolkit submodules.
   - Installs ai-toolkit requirements if present.

2. `2_Prepare_Dataset.ipynb`
   - Cleans images from `datasets/raw` into `datasets/processed/img`.
   - Requires `GEMINI_API_KEY` in `.env` before Gemini captioning.
   - Writes captions into `datasets/processed/captions`.

3. `3_Train_Config.ipynb`
   - Generates `configs/db9_toolkit_trainner.yaml`.
   - Default base model: `black-forest-labs/FLUX.2-klein-base-9B`.

4. `4_Train.ipynb`
   - Verifies config and `ai-toolkit/run.py` exist.
   - Runs `python run.py ../configs/db9_toolkit_trainner.yaml` inside `ai-toolkit`.

5. `5_Test_LoRA.ipynb`
   - Finds the newest `.safetensors` under `outputs/db9_toolkit_trainner`.
   - Loads it with `FluxPipeline` against the Flux 2 Klein base model.
   - Saves a test image to `outputs/test/test_result.png`.

## Config Defaults

- Project display name: `DB9-Toolkit-Trainner`
- Project id: `db9_toolkit_trainner`
- Base model: `black-forest-labs/FLUX.2-klein-base-9B`
- Trainer type: `sd_trainer`
- LoRA rank/alpha: `64/64`
- Batch size: `6`
- Resolution: `1536 x 1536`
- Bucket step: `64`
- Bucket range: `512` to `2048`
- Optimizer: `adamw8bit`
- Scheduler: `constant_with_warmup`
- Flux scheduler/sampler: `flowmatch`

## Known Boundaries

- `verify.py` is a smoke check, not a GPU training test.
- Gemini captioning requires a valid API key.
- Flux model access may require Hugging Face authentication.
- Python 3.10 or 3.11 is recommended for CUDA package compatibility.
