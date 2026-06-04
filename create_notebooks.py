import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_DISPLAY_NAME = "DB9-Toolkit-Trainner"
PROJECT_NAME = "db9_toolkit_trainner"
MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"


def save_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    with open(NOTEBOOKS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code_cell(source):
    if isinstance(source, str):
        source = [line + "\n" for line in source.split("\n")]
        if source and source[-1].endswith("\n"):
            source[-1] = source[-1][:-1]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cwd_cell_code = '''import os
from pathlib import Path
if Path.cwd().name == "notebooks":
    os.chdir("..")
print(f"Current working directory: {Path.cwd()}")'''

setup_ai_toolkit_code = '''from pathlib import Path
import subprocess
import sys

toolkit_dir = Path("ai-toolkit")
if toolkit_dir.exists():
    print("ai-toolkit already exists; updating submodules.")
else:
    subprocess.check_call([
        "git", "clone", "--recurse-submodules",
        "https://github.com/ostris/ai-toolkit.git",
        str(toolkit_dir),
    ])

subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=toolkit_dir)
requirements = toolkit_dir / "requirements.txt"
if requirements.exists():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
else:
    print("ai-toolkit requirements.txt not found; skipped dependency install.")'''

save_notebook("1_Setup.ipynb", [
    md_cell(f"# 1. Setup Environment\n\nCai dat moi truong va dependencies cho {PROJECT_DISPLAY_NAME}."),
    code_cell(cwd_cell_code),
    code_cell("import sys\n!{sys.executable} -m pip install -r requirements.txt"),
    code_cell(setup_ai_toolkit_code),
])

caption_gemini_code = '''import os
from dotenv import load_dotenv

load_dotenv('.env')
api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY in .env before running Gemini captioning.")

!python scripts/caption_gemini.py --input datasets/processed/img --output datasets/processed/captions --prompt "Mo ta chi tiet hinh anh nay"'''

save_notebook("2_Prepare_Dataset.ipynb", [
    md_cell("# 2. Prepare Dataset\n\nXu ly dataset (resize, convert) va tao caption bang Gemini."),
    code_cell(cwd_cell_code),
    md_cell("## 2.1. Clean Dataset"),
    code_cell("!python scripts/data_clean.py --input datasets/raw --output datasets/processed/img --min-res 512 --max-res 4096 --format jpg"),
    md_cell("## 2.2. Auto Caption (Gemini Flash)"),
    code_cell(caption_gemini_code),
])

code_config = f'''import sys
sys.path.append('scripts')
from config_generator import generate_config
import os
from pathlib import Path

project_name = "{PROJECT_NAME}"
model_path = "{MODEL_ID}"

base_dir = Path.cwd()
dataset_path = str(base_dir / "datasets" / "processed").replace("\\", "/")
output_path = str(base_dir / "outputs").replace("\\", "/")

lora_rank = 64
lora_alpha = 64
batch_size = 6
learning_rate = 4e-4
train_steps = 2000
gradient_accumulation = 1
resolution = 1536
enable_bucketing = True
bucket_step = 64
min_bucket_reso = 512
max_bucket_reso = 2048
flip_aug = False
color_aug = False
optimizer = "adamw8bit"
lr_scheduler = "constant_with_warmup"
warmup_steps = 100
gradient_checkpointing = True
quantize = False
save_every = 500
sample_every = 500
sample_prompts = [
    "A portrait of a person in DB9 style, highly detailed"
]

config_yaml = generate_config(
    project_name=project_name,
    model_path=model_path,
    dataset_path=dataset_path,
    output_path=output_path,
    lora_rank=lora_rank,
    lora_alpha=lora_alpha,
    batch_size=batch_size,
    learning_rate=learning_rate,
    train_steps=train_steps,
    gradient_accumulation=gradient_accumulation,
    resolution=resolution,
    enable_bucketing=enable_bucketing,
    bucket_step=bucket_step,
    min_bucket_reso=min_bucket_reso,
    max_bucket_reso=max_bucket_reso,
    flip_aug=flip_aug,
    color_aug=color_aug,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    warmup_steps=warmup_steps,
    gradient_checkpointing=gradient_checkpointing,
    quantize=quantize,
    save_every=save_every,
    sample_every=sample_every,
    sample_prompts=sample_prompts,
)

os.makedirs("configs", exist_ok=True)
config_path = f"configs/{{project_name}}.yaml"
with open(config_path, "w", encoding="utf-8") as f:
    f.write(config_yaml)

print(f"Config saved to {{config_path}}")'''

save_notebook("3_Train_Config.ipynb", [
    md_cell("# 3. Train Config\n\nTao file YAML cau hinh cho ai-toolkit."),
    code_cell(cwd_cell_code),
    code_cell(code_config),
])

save_notebook("4_Train.ipynb", [
    md_cell("# 4. Train\n\nBat dau qua trinh training."),
    code_cell(cwd_cell_code),
    code_cell(f'''from pathlib import Path

config_path = Path("configs/{PROJECT_NAME}.yaml")
run_path = Path("ai-toolkit/run.py")
if not config_path.exists():
    raise FileNotFoundError(f"Missing config: {{config_path}}. Run notebook 3 first.")
if not run_path.exists():
    raise FileNotFoundError("Missing ai-toolkit/run.py. Run notebook 1 first.")

!cd ai-toolkit && python run.py ../configs/{PROJECT_NAME}.yaml'''),
])

code_test = f'''import torch
from diffusers import FluxPipeline
from pathlib import Path
import os

model_id = "{MODEL_ID}"
project_name = "{PROJECT_NAME}"
checkpoint_dir = Path("outputs") / project_name
checkpoints = sorted(checkpoint_dir.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
if not checkpoints:
    raise FileNotFoundError(f"No .safetensors checkpoints found under {{checkpoint_dir}}. Run training first.")
lora_path = str(checkpoints[-1])
print(f"Loading LoRA checkpoint: {{lora_path}}")

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(lora_path)
pipe.to("cuda")

prompt = "A portrait of a person in DB9 style, highly detailed"
image = pipe(
    prompt,
    num_inference_steps=28,
    guidance_scale=3.5,
    height=1536,
    width=1536,
).images[0]

os.makedirs("outputs/test", exist_ok=True)
image.save("outputs/test/test_result.png")
print("Saved image to outputs/test/test_result.png")
image'''

save_notebook("5_Test_LoRA.ipynb", [
    md_cell("# 5. Test LoRA\n\nKiem tra checkpoint da train."),
    code_cell(cwd_cell_code),
    code_cell(code_test),
])

print("OK")
