# 🏗️ SDVN Local Trainer — Architecture Design

## 🎯 Mục tiêu

Chuyển SDVN Colab notebook thành local app với:
- ✅ Giữ nguyên UI/UX workflow của SDVN Colab
- ✅ Jupyter notebook interface (quen thuộc)
- ✅ Tối ưu cho RTX 5090 (32GB VRAM)
- ✅ Support Flux 2 Klein 9B
- ✅ Auto caption với Gemini API
- ✅ Config templates tiếng Việt

---

## 📁 Cấu trúc thư mục

```
SDVN-Local-Trainer/
├── notebooks/
│   ├── 1_Setup.ipynb                    # Cài đặt dependencies
│   ├── 2_Prepare_Dataset.ipynb          # Xử lý dataset + caption
│   ├── 3_Train_Config.ipynb             # Cấu hình training
│   ├── 4_Train.ipynb                    # Training execution
│   └── 5_Test_LoRA.ipynb                # Test checkpoints
│
├── scripts/
│   ├── setup.py                         # Setup environment
│   ├── caption_gemini.py                # Auto caption với Gemini
│   ├── caption_florence.py              # Auto caption với Florence
│   ├── data_clean.py                    # Clean dataset
│   ├── config_generator.py              # Generate YAML từ form
│   └── test_lora.py                     # Test LoRA script
│
├── configs/
│   ├── templates/
│   │   ├── flux2_klein_default.yaml     # Template mặc định
│   │   ├── flux2_klein_fast.yaml        # Fast training
│   │   ├── flux2_klein_quality.yaml     # Quality focus
│   │   └── flux2_klein_5090.yaml        # Optimized cho 5090
│   └── generated/                       # Configs được generate
│
├── ai-toolkit/                          # Submodule: ostris/ai-toolkit
│
├── models/                              # Model storage
│   └── flux2-klein-9b/
│
├── datasets/                            # Dataset storage
│   ├── raw/                             # Ảnh gốc
│   ├── processed/                       # Ảnh đã xử lý
│   │   ├── img/
│   │   └── captions/
│   └── metadata.json                    # Dataset info
│
├── outputs/                             # Training outputs
│   ├── checkpoints/
│   ├── samples/
│   └── logs/
│
├── requirements.txt                     # Python dependencies
├── environment.yml                      # Conda environment (optional)
├── README.md                            # Hướng dẫn sử dụng
└── .env.example                         # Environment variables template

```

---

## 🎨 UI Design — Jupyter Notebooks

### 1. Setup Notebook (`1_Setup.ipynb`)

**Cells:**

```python
# Cell 1: Header
"""
# 🚀 SDVN Local Trainer — Setup
Cài đặt môi trường training Flux LoRA trên máy local
"""

# Cell 2: System Info
#@title 📊 Thông tin hệ thống
import torch
import subprocess

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Cell 3: Install Dependencies
#@title ⚙️ Cài đặt dependencies
#@markdown Chọn môi trường cài đặt:
install_mode = "pip"  #@param ["pip", "conda"]

if install_mode == "pip":
    !pip install -r requirements.txt
else:
    !conda env create -f environment.yml

# Cell 4: Clone ai-toolkit
#@title 📦 Clone ai-toolkit
#@markdown ai-toolkit là backend training engine

import os
if not os.path.exists("ai-toolkit"):
    !git clone https://github.com/ostris/ai-toolkit.git
    !cd ai-toolkit && pip install -r requirements.txt

# Cell 5: Download Model
#@title 🤖 Download Flux 2 Klein 9B
#@markdown Model sẽ được lưu vào `models/flux2-klein-9b/`

model_name = "black-forest-labs/FLUX.2-klein-base-9B"  #@param {type:"string"}
force_redownload = False  #@param {type:"boolean"}

import os
model_dir = "models/flux2-klein-9b"

if not os.path.exists(model_dir) or force_redownload:
    !huggingface-cli download {model_name} --local-dir {model_dir}
    print(f"✅ Model downloaded to {model_dir}")
else:
    print(f"⏭️  Model already exists at {model_dir}")

# Cell 6: Verify Setup
#@title ✅ Kiểm tra cài đặt

checks = {
    "PyTorch": torch.cuda.is_available(),
    "ai-toolkit": os.path.exists("ai-toolkit"),
    "Model": os.path.exists("models/flux2-klein-9b"),
    "VRAM >= 12GB": torch.cuda.get_device_properties(0).total_memory / 1024**3 >= 12
}

for name, status in checks.items():
    icon = "✅" if status else "❌"
    print(f"{icon} {name}")

if all(checks.values()):
    print("\n🎉 Setup hoàn tất! Chuyển sang notebook tiếp theo.")
else:
    print("\n⚠️  Có lỗi trong quá trình setup. Kiểm tra lại.")
```

---

### 2. Prepare Dataset Notebook (`2_Prepare_Dataset.ipynb`)

**Cells:**

```python
# Cell 1: Header
"""
# 📁 SDVN Local Trainer — Prepare Dataset
Xử lý dataset: clean, resize, caption
"""

# Cell 2: Dataset Path
#@title 📂 Đường dẫn dataset
#@markdown Chọn thư mục chứa ảnh training

dataset_raw_path = "datasets/raw"  #@param {type:"string"}
dataset_output_path = "datasets/processed"  #@param {type:"string"}

import os
os.makedirs(dataset_raw_path, exist_ok=True)
os.makedirs(f"{dataset_output_path}/img", exist_ok=True)
os.makedirs(f"{dataset_output_path}/captions", exist_ok=True)

print(f"📂 Raw dataset: {dataset_raw_path}")
print(f"📂 Output: {dataset_output_path}")

# Cell 3: Data Clean
#@title 🧹 Làm sạch dataset
#@markdown Xóa ảnh lỗi, resize, convert format

enable_clean = True  #@param {type:"boolean"}
min_resolution = 512  #@param {type:"integer"}
max_resolution = 4096  #@param {type:"integer"}
target_format = "jpg"  #@param ["jpg", "png", "webp"]

if enable_clean:
    !python scripts/data_clean.py \
        --input {dataset_raw_path} \
        --output {dataset_output_path}/img \
        --min-res {min_resolution} \
        --max-res {max_resolution} \
        --format {target_format}

# Cell 4: Auto Caption
#@title 🤖 Tự động tạo caption
#@markdown Chọn phương pháp caption

caption_method = "Gemini API"  #@param ["Gemini API", "Florence", "Manual Template"]
gemini_api_key = ""  #@param {type:"string"}

if caption_method == "Gemini API":
    import os
    os.environ["GEMINI_API_KEY"] = gemini_api_key
    
    !python scripts/caption_gemini.py \
        --input {dataset_output_path}/img \
        --output {dataset_output_path}/captions \
        --prompt "Describe this architectural visualization image in detail"

elif caption_method == "Florence":
    !python scripts/caption_florence.py \
        --input {dataset_output_path}/img \
        --output {dataset_output_path}/captions

elif caption_method == "Manual Template":
    # Generate template captions
    from pathlib import Path
    img_dir = Path(f"{dataset_output_path}/img")
    cap_dir = Path(f"{dataset_output_path}/captions")
    
    for img in img_dir.glob("*.jpg"):
        cap_file = cap_dir / f"{img.stem}.txt"
        if not cap_file.exists():
            cap_file.write_text("photoreal architectural visualization, [EDIT THIS], high detail, sharp focus")
    
    print(f"✅ Created {len(list(cap_dir.glob('*.txt')))} template captions")
    print("⚠️  Hãy edit các file .txt trong datasets/processed/captions/")

# Cell 5: Verify Dataset
#@title ✅ Kiểm tra dataset

from pathlib import Path

img_dir = Path(f"{dataset_output_path}/img")
cap_dir = Path(f"{dataset_output_path}/captions")

images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
captions = list(cap_dir.glob("*.txt"))

print(f"📊 Dataset Statistics:")
print(f"  Images: {len(images)}")
print(f"  Captions: {len(captions)}")

# Check missing captions
missing = []
for img in images:
    cap_file = cap_dir / f"{img.stem}.txt"
    if not cap_file.exists():
        missing.append(img.name)

if missing:
    print(f"\n⚠️  Missing captions ({len(missing)}):")
    for name in missing[:10]:
        print(f"    - {name}")
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more")
else:
    print("\n✅ All images have captions!")

# Show sample
if captions:
    sample = captions[0]
    print(f"\n📝 Sample caption ({sample.name}):")
    print(f"  {sample.read_text()}")
```

---

### 3. Train Config Notebook (`3_Train_Config.ipynb`)

**Cells:**

```python
# Cell 1: Header
"""
# ⚙️ SDVN Local Trainer — Training Configuration
Cấu hình training parameters
"""

# Cell 2: Basic Settings
#@title 🎯 Cài đặt cơ bản

project_name = "flux2_klein_architectural"  #@param {type:"string"}
#@markdown Tên project (dùng cho output folder)

model_path = "models/flux2-klein-9b"  #@param {type:"string"}
dataset_path = "datasets/processed"  #@param {type:"string"}
output_path = "outputs"  #@param {type:"string"}

# Cell 3: LoRA Settings
#@title 🔧 Cấu hình LoRA

lora_rank = 64  #@param {type:"slider", min:16, max:128, step:16}
lora_alpha = 64  #@param {type:"slider", min:16, max:128, step:16}

#@markdown **Khuyến nghị:**
#@markdown - Rank 32-64: Standard quality
#@markdown - Rank 64-128: High quality (RTX 5090)
#@markdown - Alpha = Rank hoặc Rank * 0.75

# Cell 4: Training Settings
#@title 🚀 Cấu hình training

batch_size = 6  #@param {type:"slider", min:1, max:12, step:1}
#@markdown RTX 5090 khuyến nghị: 6-8

learning_rate = 0.0001  #@param {type:"number"}
#@markdown Standard: 1e-4 (0.0001)

train_steps = 2000  #@param {type:"integer"}
#@markdown Số steps training

gradient_accumulation = 1  #@param {type:"integer"}
#@markdown Tăng nếu batch_size nhỏ

# Cell 5: Dataset Settings
#@title 📊 Cấu hình dataset

resolution = 1536  #@param {type:"slider", min:512, max:2048, step:64}
#@markdown Resolution training (match DB9 tile size: 1536)

enable_bucketing = True  #@param {type:"boolean"}
#@markdown Cho phép train nhiều aspect ratios

bucket_step = 64  #@param {type:"integer"}
min_bucket_reso = 1024  #@param {type:"integer"}
max_bucket_reso = 2048  #@param {type:"integer"}

flip_augmentation = True  #@param {type:"boolean"}
color_augmentation = False  #@param {type:"boolean"}
#@markdown ⚠️ Tắt color aug để tránh conflict với color lock

# Cell 6: Advanced Settings
#@title 🔬 Cài đặt nâng cao

optimizer = "adamw"  #@param ["adamw", "adamw8bit", "lion"]
lr_scheduler = "constant_with_warmup"  #@param ["constant", "constant_with_warmup", "cosine", "cosine_with_restarts"]
warmup_steps = 100  #@param {type:"integer"}

gradient_checkpointing = False  #@param {type:"boolean"}
#@markdown RTX 5090 không cần (đủ VRAM)

quantize_model = False  #@param {type:"boolean"}
#@markdown RTX 5090 không cần (đủ VRAM)

# Cell 7: Saving Settings
#@title 💾 Cài đặt lưu checkpoint

save_every = 250  #@param {type:"integer"}
#@markdown Lưu checkpoint mỗi N steps

sample_every = 100  #@param {type:"integer"}
#@markdown Generate sample mỗi N steps

sample_prompts = [
    "photoreal architectural visualization, modern villa exterior, concrete and glass, soft daylight, high detail",
    "photoreal architectural visualization, luxury interior living room, marble floor, warm lighting, high detail"
]  #@param {type:"raw"}

# Cell 8: Generate Config
#@title 📝 Tạo config file

import sys
sys.path.append('scripts')
from config_generator import generate_config

config = generate_config(
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
    flip_aug=flip_augmentation,
    color_aug=color_augmentation,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    warmup_steps=warmup_steps,
    gradient_checkpointing=gradient_checkpointing,
    quantize=quantize_model,
    save_every=save_every,
    sample_every=sample_every,
    sample_prompts=sample_prompts
)

config_path = f"configs/generated/{project_name}.yaml"
import os
os.makedirs("configs/generated", exist_ok=True)

with open(config_path, 'w') as f:
    f.write(config)

print(f"✅ Config saved to: {config_path}")
print("\n📄 Config preview:")
print(config[:500] + "...")

# Cell 9: Estimate VRAM
#@title 📊 Ước tính VRAM usage

def estimate_vram(batch_size, rank, resolution, quantize):
    base = 8 if quantize else 12  # Klein 9B base
    batch_overhead = batch_size * 1.5
    rank_overhead = rank / 64 * 2
    reso_overhead = (resolution / 1536) ** 2 * 2
    
    total = base + batch_overhead + rank_overhead + reso_overhead
    return total

vram_estimate = estimate_vram(batch_size, lora_rank, resolution, quantize_model)

print(f"📊 Estimated VRAM usage: {vram_estimate:.1f} GB")
print(f"📊 Available VRAM: 32 GB (RTX 5090)")

if vram_estimate > 32:
    print("⚠️  WARNING: Có thể OOM! Giảm batch_size hoặc resolution")
elif vram_estimate > 28:
    print("⚠️  VRAM usage cao. Monitor trong quá trình training")
else:
    print("✅ VRAM usage OK")
```

---

### 4. Train Notebook (`4_Train.ipynb`)

```python
# Cell 1: Header
"""
# 🚀 SDVN Local Trainer — Training
Bắt đầu training LoRA
"""

# Cell 2: Load Config
#@title 📂 Load config

import os
from pathlib import Path

configs = list(Path("configs/generated").glob("*.yaml"))
config_names = [c.stem for c in configs]

if not config_names:
    print("❌ Không tìm thấy config. Chạy notebook 3_Train_Config.ipynb trước")
else:
    print("📋 Available configs:")
    for i, name in enumerate(config_names):
        print(f"  {i+1}. {name}")

selected_config = config_names[0] if config_names else ""  #@param {type:"string"}
config_path = f"configs/generated/{selected_config}.yaml"

# Cell 3: Pre-flight Check
#@title ✅ Kiểm tra trước khi train

import torch
import yaml

# Load config
with open(config_path) as f:
    config = yaml.safe_load(f)

# Checks
checks = {
    "CUDA available": torch.cuda.is_available(),
    "Model exists": os.path.exists(config['config']['process'][0]['model']['name_or_path']),
    "Dataset exists": os.path.exists(config['config']['process'][0]['datasets'][0]['folder_path']),
    "Output dir writable": os.access("outputs", os.W_OK)
}

for name, status in checks.items():
    icon = "✅" if status else "❌"
    print(f"{icon} {name}")

if not all(checks.values()):
    print("\n❌ Pre-flight check failed. Fix errors above.")
else:
    print("\n✅ All checks passed. Ready to train!")

# Cell 4: Start Training
#@title 🚀 Bắt đầu training

#@markdown ⚠️ Training sẽ chạy trong cell này. Không đóng notebook.

import subprocess
import sys

cmd = [
    sys.executable,
    "ai-toolkit/run.py",
    config_path
]

print(f"🚀 Starting training...")
print(f"📝 Config: {config_path}")
print(f"📊 Monitor: tensorboard --logdir outputs/{selected_config}")
print("\n" + "="*60 + "\n")

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in process.stdout:
    print(line, end='')

process.wait()

if process.returncode == 0:
    print("\n" + "="*60)
    print("✅ Training completed successfully!")
else:
    print("\n" + "="*60)
    print(f"❌ Training failed with code {process.returncode}")

# Cell 5: Monitor (optional)
#@title 📊 Monitor training (TensorBoard)

#@markdown Chạy cell này trong terminal riêng hoặc background

%load_ext tensorboard
%tensorboard --logdir outputs/{selected_config}
```

---

### 5. Test LoRA Notebook (`5_Test_LoRA.ipynb`)

```python
# Cell 1: Header
"""
# 🧪 SDVN Local Trainer — Test LoRA
Test checkpoints và export
"""

# Cell 2: List Checkpoints
#@title 📋 Danh sách checkpoints

from pathlib import Path

project_name = "flux2_klein_architectural"  #@param {type:"string"}
checkpoint_dir = Path(f"outputs/{project_name}")

if not checkpoint_dir.exists():
    print(f"❌ Project not found: {project_name}")
else:
    checkpoints = sorted(checkpoint_dir.glob("*.safetensors"))
    print(f"📊 Found {len(checkpoints)} checkpoints:\n")
    for i, ckpt in enumerate(checkpoints):
        size_mb = ckpt.stat().st_size / 1024 / 1024
        print(f"  {i+1}. {ckpt.name} ({size_mb:.1f} MB)")

# Cell 3: Quick Test
#@title 🧪 Test nhanh với ComfyUI

#@markdown Copy checkpoint vào ComfyUI models/loras/

selected_checkpoint = ""  #@param {type:"string"}
comfyui_lora_dir = ""  #@param {type:"string"}

if selected_checkpoint and comfyui_lora_dir:
    import shutil
    src = checkpoint_dir / selected_checkpoint
    dst = Path(comfyui_lora_dir) / selected_checkpoint
    
    shutil.copy(src, dst)
    print(f"✅ Copied to: {dst}")
    print("\n📝 Test trong ComfyUI:")
    print("  1. Load Checkpoint: FLUX.2-klein-base-9B")
    print(f"  2. Load LoRA: {selected_checkpoint}")
    print("  3. LoRA strength: 0.5")
    print("  4. Generate test image")

# Cell 4: Batch Test
#@title 🔬 Test nhiều strengths

test_strengths = [0.3, 0.4, 0.5, 0.6, 0.7]  #@param {type:"raw"}
test_prompt = "photoreal architectural visualization, modern villa, high detail"  #@param {type:"string"}

#@markdown Requires ComfyUI API

print("🔬 Batch test với strengths:", test_strengths)
print("⚠️  Cần ComfyUI API running")
print("\nScript: scripts/test_lora.py")

# Cell 5: Export Best Checkpoint
#@title 📦 Export checkpoint tốt nhất

best_checkpoint = ""  #@param {type:"string"}
export_name = "flux2_klein_arch_v1"  #@param {type:"string"}
export_dir = "outputs/exports"  #@param {type:"string"}

if best_checkpoint:
    import shutil
    import os
    
    os.makedirs(export_dir, exist_ok=True)
    
    src = checkpoint_dir / best_checkpoint
    dst = Path(export_dir) / f"{export_name}.safetensors"
    
    shutil.copy(src, dst)
    print(f"✅ Exported to: {dst}")
    
    # Create metadata
    metadata = {
        "name": export_name,
        "base_model": "FLUX.2-klein-base-9B",
        "trained_on": project_name,
        "checkpoint": best_checkpoint,
        "recommended_strength": "0.5-0.6",
        "notes": "Optimized for DB9 Flux Locked Upscale"
    }
    
    import json
    meta_path = Path(export_dir) / f"{export_name}.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to: {meta_path}")
```

---

## 📝 Supporting Scripts

Tao sẽ tạo các script hỗ trợ trong message tiếp theo.

Mày OK với kiến trúc này không? Có cần adjust gì không?
