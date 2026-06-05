# DB9-Toolkit-Trainner

Local Jupyter workflow for training DB9 Flux LoRA checkpoints with ai-toolkit.

## Requirements

- Python 3.10 or 3.11 recommended
- CUDA-capable NVIDIA GPU with at least 12 GB VRAM
- Git
- Hugging Face access for the selected base model
- Gemini API key only if using Gemini auto-captioning

## Setup

```bash
git clone https://github.com/bemapa3/DB9-TrainLORA.git
cd DB9-Local-Trainer
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/1_Setup.ipynb` first. It installs local dependencies, clones or updates `ai-toolkit`, initializes submodules, and installs `ai-toolkit` requirements.

## Workflow

1. `notebooks/1_Setup.ipynb` - environment and ai-toolkit setup
2. `notebooks/2_Prepare_Dataset.ipynb` - clean dataset and generate captions
3. `notebooks/3_Train_Config.ipynb` - generate `configs/db9_toolkit_trainner.yaml`
4. `notebooks/4_Train.ipynb` - run ai-toolkit training
5. `notebooks/5_Test_LoRA.ipynb` - load the newest `.safetensors` checkpoint and test it

## Dataset Layout

```text
datasets/
  raw/                 # original images
  processed/
    img/               # cleaned training images
    captions/          # one .txt caption per image
```

## Defaults

- Project display name: `DB9-Toolkit-Trainner`
- Config project id: `db9_toolkit_trainner`
- Base model: `black-forest-labs/FLUX.2-klein-base-9B`
- Batch size: 6
- LoRA rank/alpha: 64/64
- Resolution: 1536 x 1536
- Optimizer: `adamw8bit`

## Environment

Copy `.env.example` to `.env` and set values as needed:

```bash
GEMINI_API_KEY=your_api_key_here
HF_TOKEN=your_token_here
```

`GEMINI_API_KEY` is required only for Gemini captioning. `HF_TOKEN` may be required by Hugging Face depending on model access.

## Verification

```bash
python verify.py
```

This checks notebook JSON, generated config shape, expected project/model defaults, and key workflow path conventions. It does not run GPU training.

## Links

- ai-toolkit: https://github.com/ostris/ai-toolkit
- Flux 2 Klein: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B

## One-click RTX 5090 setup

On Windows, double-click `setup_5090.bat` from the repository folder. It will:

- pull the latest code
- detect NVIDIA GPU information
- create `.venv`
- install PyTorch CUDA 12.8 wheels for RTX 5090-class systems
- install project requirements
- clone/update `ai-toolkit`
- create runtime folders
- run `python verify.py`

After it finishes, put training images in `datasets/raw`, edit `.env` if Gemini/Hugging Face tokens are needed, then run:

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook
```

## Launcher app

For a simple clickable control panel, double-click `launcher_app.bat`. The launcher shows common commands and buttons for:

- full RTX 5090 setup
- verification
- opening Jupyter
- opening data/output folders
- editing `.env`
- cleaning images
- Gemini captioning
- training
- CUDA check

Long-running jobs open in separate PowerShell windows so their logs remain visible.

## DB9Studio-style local training notebook

If you prefer the familiar DB9Studio workflow form layout, open `notebooks/0_DB9Studio_UI.ipynb` or click `DB9Studio-style UI` in `launcher_app.bat`. It opens a code-hidden widget UI for setup, data processing, upscale tiling, config generation, and training.

## Install Notes

- Prefer the local model snapshot for Flux 2 Klein. Put the full snapshot under `models/FLUX.2-klein-base-9B` before training so generated configs avoid gated Hugging Face downloads.
- The local model folder must contain `model_index.json`, `scheduler`, `text_encoder`, `tokenizer`, `transformer`, and `vae`.
- After `git pull`, restart the Jupyter kernel or reopen the launcher so the UI imports the latest scripts.
- If training from PowerShell, use the project virtualenv explicitly: `\.venv\Scripts\python.exe`.
- For RTX 5090/32GB, start conservative: `batch_size: 1`, gradient accumulation 4-8, resolution 1024-1536 for Flux 2 Klein img2img/upscale.

## Open Jupyter

From the repository folder:

```powershell
Set-Location D:\BBB\DB9-Local-Trainer
.\.venv\Scripts\Activate.ps1
jupyter notebook
```

Then open `notebooks/0_DB9_Toolkit_Training_Local.ipynb` for the DB9Studio-style UI. If browser does not open automatically, copy the localhost URL printed in PowerShell into Chrome/Edge.

## Dataset Name Validation

Before generating config or training, validate processed filenames:

```powershell
.\.venv\Scripts\python.exe scripts\validate_dataset.py --processed datasets\processed --mode img2img_upscale --control datasets\processed\control
```

The validator logs `BAD_NAME` lines for unsafe names, missing captions, missing controls, or orphan files. Target images, captions, and controls must share the same stem.

## Img2Img / Upscale Training Tips

- Use `2.1b Upscale detail` when starting from large sharp source images; it creates target/control/caption tile pairs.
- Use `2.1c Size-degrade pairs` when you already have separate sharp outputs and degraded inputs.
- For images larger than 2048px, tile them first. Do not train directly on oversized originals.
- If VRAM OOM happens during prompt encoding, lower tile/resolution to 1024 or 1536, set batch size to 1, and keep gradient checkpointing enabled.
- Clean `datasets/processed/img`, `datasets/processed/control`, and `datasets/processed/captions` before rebuilding tiles to avoid mixing old and new pairs.
