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
cd SDVN-Local-Trainer
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
