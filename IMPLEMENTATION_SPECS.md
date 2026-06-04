# 📋 SDVN Local Trainer — Implementation Specifications

## 🎯 Cho Gemini Flash (Code Implementation)

### Task: Implement supporting scripts

---

## 1. `scripts/config_generator.py`

**Purpose:** Generate ai-toolkit YAML config from notebook parameters

**Input:** Python function parameters from notebook
**Output:** YAML string

**Spec:**

```python
def generate_config(
    project_name: str,
    model_path: str,
    dataset_path: str,
    output_path: str,
    lora_rank: int,
    lora_alpha: int,
    batch_size: int,
    learning_rate: float,
    train_steps: int,
    gradient_accumulation: int,
    resolution: int,
    enable_bucketing: bool,
    bucket_step: int,
    min_bucket_reso: int,
    max_bucket_reso: int,
    flip_aug: bool,
    color_aug: bool,
    optimizer: str,
    lr_scheduler: str,
    warmup_steps: int,
    gradient_checkpointing: bool,
    quantize: bool,
    save_every: int,
    sample_every: int,
    sample_prompts: list
) -> str:
    """
    Generate ai-toolkit YAML config
    
    Returns:
        YAML string ready to write to file
    """
    pass
```

**Template structure:**

```yaml
---
job: extension
config:
  name: {project_name}
  process:
    - type: sd_trainer
      training_folder: {output_path}/{project_name}
      device: cuda:0
      
      model:
        name_or_path: {model_path}
        is_flux: true
        quantize: {quantize}
      
      network:
        type: lora
        linear: {lora_rank}
        linear_alpha: {lora_alpha}
      
      datasets:
        - folder_path: {dataset_path}/img
          caption_ext: txt
          caption_folder: {dataset_path}/captions
          resolution: [{resolution}, {resolution}]
          enable_bucket: {enable_bucketing}
          bucket_step_size: {bucket_step}
          min_bucket_reso: {min_bucket_reso}
          max_bucket_reso: {max_bucket_reso}
          random_crop: false
          flip_aug: {flip_aug}
          color_aug: {color_aug}
      
      train:
        batch_size: {batch_size}
        steps: {train_steps}
        gradient_accumulation_steps: {gradient_accumulation}
        train_unet: true
        train_text_encoder: false
        lr: {learning_rate}
        lr_scheduler: {lr_scheduler}
        lr_warmup_steps: {warmup_steps}
        optimizer: {optimizer}
        max_grad_norm: 1.0
        gradient_checkpointing: {gradient_checkpointing}
        noise_scheduler: flowmatch
        save_every: {save_every}
        sample_every: {sample_every}
        sample_prompts: {sample_prompts}
      
      logging:
        log_every: 10
        use_wandb: false
      
      sample:
        sampler: flowmatch
        sample_steps: 20
        cfg_scale: 1.0
        width: {resolution}
        height: {resolution}

meta:
  name: {project_name}
  version: '1.0'
```

---

## 2. `scripts/caption_gemini.py`

**Purpose:** Auto-generate captions using Gemini API

**CLI:**
```bash
python scripts/caption_gemini.py \
  --input datasets/processed/img \
  --output datasets/processed/captions \
  --prompt "Describe this architectural image" \
  --api-key $GEMINI_API_KEY
```

**Spec:**

```python
import argparse
import os
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input image directory')
    parser.add_argument('--output', required=True, help='Output caption directory')
    parser.add_argument('--prompt', default='Describe this image', help='Caption prompt')
    parser.add_argument('--api-key', help='Gemini API key (or use GEMINI_API_KEY env)')
    parser.add_argument('--model', default='gemini-1.5-flash', help='Gemini model')
    parser.add_argument('--rate-limit', type=float, default=1.0, help='Seconds between requests')
    args = parser.parse_args()
    
    # Setup
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("API key required")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)
    
    img_dir = Path(args.input)
    cap_dir = Path(args.output)
    cap_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    
    for i, img_path in enumerate(images):
        cap_path = cap_dir / f"{img_path.stem}.txt"
        
        if cap_path.exists():
            print(f"[{i+1}/{len(images)}] ⏭️  {img_path.name} (exists)")
            continue
        
        try:
            img = Image.open(img_path)
            response = model.generate_content([args.prompt, img])
            
            cap_path.write_text(response.text.strip())
            print(f"[{i+1}/{len(images)}] ✅ {img_path.name}")
            
            time.sleep(args.rate_limit)
        except Exception as e:
            print(f"[{i+1}/{len(images)}] ❌ {img_path.name}: {e}")
    
    print(f"\n✅ Completed: {len(list(cap_dir.glob('*.txt')))} captions")

if __name__ == '__main__':
    main()
```

---

## 3. `scripts/data_clean.py`

**Purpose:** Clean dataset (remove corrupt, resize, convert format)

**CLI:**
```bash
python scripts/data_clean.py \
  --input datasets/raw \
  --output datasets/processed/img \
  --min-res 512 \
  --max-res 4096 \
  --format jpg
```

**Spec:**

```python
import argparse
from pathlib import Path
from PIL import Image
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-res', type=int, default=512)
    parser.add_argument('--max-res', type=int, default=4096)
    parser.add_argument('--format', default='jpg', choices=['jpg', 'png', 'webp'])
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')) + list(input_dir.glob('*.webp'))
    
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    
    for img_path in images:
        try:
            img = Image.open(img_path)
            w, h = img.size
            
            # Check resolution
            if w < args.min_res or h < args.min_res:
                print(f"⏭️  {img_path.name}: too small ({w}×{h})")
                stats['skipped'] += 1
                continue
            
            if w > args.max_res or h > args.max_res:
                print(f"⏭️  {img_path.name}: too large ({w}×{h})")
                stats['skipped'] += 1
                continue
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save
            output_path = output_dir / f"{img_path.stem}.{args.format}"
            img.save(output_path, quality=95)
            
            print(f"✅ {img_path.name} → {output_path.name}")
            stats['processed'] += 1
            
        except Exception as e:
            print(f"❌ {img_path.name}: {e}")
            stats['errors'] += 1
    
    print(f"\n📊 Stats: {stats['processed']} processed, {stats['skipped']} skipped, {stats['errors']} errors")

if __name__ == '__main__':
    main()
```

---

## 4. `requirements.txt`

```
torch>=2.0.0
torchvision
transformers
diffusers
accelerate
safetensors
huggingface-hub
google-generativeai
pillow
pyyaml
tensorboard
tqdm
```

---

## 5. `README.md`

**Content:**

```markdown
# 🚀 SDVN Local Trainer

Train Flux LoRA trên máy local với Jupyter notebooks.

## 📋 Requirements

- Python 3.10+
- CUDA 11.8+
- GPU với ≥12GB VRAM (khuyến nghị: RTX 5090 32GB)

## 🔧 Setup

1. Clone repo:
```bash
git clone <repo-url>
cd SDVN-Local-Trainer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Jupyter:
```bash
jupyter notebook
```

4. Mở `notebooks/1_Setup.ipynb` và làm theo hướng dẫn

## 📁 Workflow

1. **Setup** (`1_Setup.ipynb`) — Cài đặt môi trường
2. **Prepare Dataset** (`2_Prepare_Dataset.ipynb`) — Xử lý dataset
3. **Train Config** (`3_Train_Config.ipynb`) — Cấu hình training
4. **Train** (`4_Train.ipynb`) — Training
5. **Test LoRA** (`5_Test_LoRA.ipynb`) — Test checkpoints

## 🎯 Optimized cho RTX 5090

- Batch size: 6-8
- LoRA rank: 64-128
- Resolution: 1536×1536
- No quantization needed
- Full precision training

## 📊 Estimated Training Time

- 2000 steps, batch 6, rank 64: ~12-15 minutes
- Dataset 100 images: ~2000-2500 steps recommended

## 🔗 Links

- ai-toolkit: https://github.com/ostris/ai-toolkit
- Flux 2 Klein: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B
```

---

## 6. `.env.example`

```bash
# Gemini API Key (for auto-captioning)
GEMINI_API_KEY=your_api_key_here

# HuggingFace Token (for model download)
HF_TOKEN=your_token_here

# ComfyUI Path (for testing)
COMFYUI_PATH=/path/to/ComfyUI
```

---

## 🎯 Implementation Priority

### Phase 1 (Core — Gemini Flash)
1. ✅ `config_generator.py` — YAML generation
2. ✅ `caption_gemini.py` — Auto caption
3. ✅ `data_clean.py` — Dataset cleaning
4. ✅ `requirements.txt`
5. ✅ `README.md`

### Phase 2 (Notebooks — Gemini Flash)
1. ✅ `1_Setup.ipynb`
2. ✅ `2_Prepare_Dataset.ipynb`
3. ✅ `3_Train_Config.ipynb`
4. ✅ `4_Train.ipynb`
5. ✅ `5_Test_LoRA.ipynb`

### Phase 3 (Optional — Later)
- `caption_florence.py` — Florence captioning
- `test_lora.py` — Automated testing
- Config templates (YAML files)

---

## 📝 Notes for Gemini

- Use Python 3.10+ syntax
- Follow PEP 8 style
- Add type hints where appropriate
- Include error handling
- Add progress indicators (tqdm, print)
- Keep code simple and readable
- Match SDVN Colab UX (form-based, Vietnamese labels)

---

## 🔍 For Codex (Review)

Check:
- ✅ YAML syntax correct
- ✅ File paths consistent
- ✅ Error handling robust
- ✅ CLI arguments complete
- ✅ Dependencies listed
- ✅ Notebook cells executable
- ✅ RTX 5090 optimizations applied
- ✅ Matches ai-toolkit API

