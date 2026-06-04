import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def save_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(NOTEBOOKS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    }

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
        "source": source
    }

cwd_cell_code = '''import os
from pathlib import Path
if Path.cwd().name == "notebooks":
    os.chdir("..")
print(f"Current working directory: {Path.cwd()}")'''

# 1_Setup.ipynb
save_notebook("1_Setup.ipynb", [
    md_cell("# 1. Setup Environment\n\nCài đặt môi trường và các dependencies cần thiết."),
    code_cell(cwd_cell_code),
    code_cell("!git clone https://github.com/ostris/ai-toolkit.git\n!cd ai-toolkit && git submodule update --init --recursive"),
    code_cell("!pip install -r requirements.txt")
])

# 2_Prepare_Dataset.ipynb
save_notebook("2_Prepare_Dataset.ipynb", [
    md_cell("# 2. Prepare Dataset\n\nXử lý dataset (resize, convert) và tự động tạo caption bằng Gemini."),
    code_cell(cwd_cell_code),
    code_cell("import os\nfrom dotenv import load_dotenv\n\nload_dotenv('.env')"),
    md_cell("## 2.1. Clean Dataset"),
    code_cell("!python scripts/data_clean.py --input datasets/raw --output datasets/processed/img --min-res 512 --max-res 4096 --format jpg"),
    md_cell("## 2.2. Auto Caption (Gemini Flash)"),
    code_cell("!python scripts/caption_gemini.py --input datasets/processed/img --output datasets/processed/captions --prompt \"Mô tả chi tiết hình ảnh này\"")
])

# 3_Train_Config.ipynb
code_config = '''import sys
sys.path.append('scripts')
from config_generator import generate_config
import os
from pathlib import Path

project_name = "sdvn_flux_lora"
model_path = "black-forest-labs/FLUX.1-dev"

# Get absolute paths to pass to config
base_dir = Path.cwd()
dataset_path = str(base_dir / "datasets" / "processed").replace("\\", "/")
output_path = str(base_dir / "outputs").replace("\\", "/")

# 🎯 Optimized cho RTX 5090
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
    "A portrait of a person in sdvn style, highly detailed"
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
    sample_prompts=sample_prompts
)

os.makedirs("configs", exist_ok=True)
config_path = f"configs/{project_name}.yaml"
with open(config_path, "w", encoding="utf-8") as f:
    f.write(config_yaml)
    
print(f"✅ Config saved to {config_path}")'''

save_notebook("3_Train_Config.ipynb", [
    md_cell("# 3. Train Config\n\nTạo file YAML cấu hình cho ai-toolkit."),
    code_cell(cwd_cell_code),
    code_cell(code_config)
])

# 4_Train.ipynb
save_notebook("4_Train.ipynb", [
    md_cell("# 4. Train\n\nBắt đầu quá trình training."),
    code_cell(cwd_cell_code),
    code_cell("!cd ai-toolkit && python run.py ../configs/sdvn_flux_lora.yaml")
])

# 5_Test_LoRA.ipynb
code_test = '''import torch
from diffusers import FluxPipeline
import os

model_id = "black-forest-labs/FLUX.1-dev"
lora_path = "outputs/sdvn_flux_lora/sdvn_flux_lora.safetensors"

print("Đang tải model...")
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(lora_path)
pipe.to("cuda")

prompt = "A portrait of a person in sdvn style, highly detailed"

print("Đang tạo ảnh...")
image = pipe(
    prompt,
    num_inference_steps=28,
    guidance_scale=3.5,
    height=1536,
    width=1536
).images[0]

os.makedirs("outputs/test", exist_ok=True)
image.save("outputs/test/test_result.png")
print("✅ Đã lưu ảnh tại outputs/test/test_result.png")
image'''

save_notebook("5_Test_LoRA.ipynb", [
    md_cell("# 5. Test LoRA\n\nKiểm tra model đã train."),
    code_cell(cwd_cell_code),
    code_cell(code_test)
])

print("OK")
