# 🚀 SDVN Local Trainer

Train Flux LoRA trên máy local với Jupyter notebooks.

## 📋 Requirements

- Python 3.10+
- CUDA 11.8+
- GPU với ≥12GB VRAM (khuyến nghị: RTX 5090 32GB)

## 🔧 Setup

1. Clone repo:
```bash
git clone https://github.com/your-username/SDVN-Local-Trainer
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
