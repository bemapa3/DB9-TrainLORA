from pathlib import Path
import os
import subprocess
import sys

import ipywidgets as W
from IPython.display import display, HTML

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    os.chdir(PROJECT_ROOT.parent)
PROJECT_ROOT = Path.cwd()

style = {"description_width": "160px"}
wide = W.Layout(width="100%")
third = W.Layout(width="32%")
half = W.Layout(width="49%")
log = W.Output(layout=W.Layout(border="1px solid #444", height="280px", overflow="auto", width="100%"))


def py():
    candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return str(candidate) if candidate.exists() else sys.executable


def run_cmd(title, cmd, cwd=None):
    cwd = Path(cwd or PROJECT_ROOT)
    with log:
        print(f"\n==> {title}")
        print(f"> {cmd}")
        p = subprocess.Popen(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="")
        code = p.wait()
        print(("DONE" if code == 0 else "FAILED") + f": {title}")


def note(text):
    return W.HTML(f'<span style="background:#2b2f33;border:1px solid #555;border-radius:6px;padding:5px 8px;color:#eee;font-family:monospace;display:inline-block;margin:4px 0">{text}</span>')


def open_folder(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(path)
    else:
        run_cmd("Open folder", f'xdg-open "{path}"')

# 1 Setup
ui_train = W.Dropdown(description="UI_Train", options=["DB9Studio", "Toolkit_UI"], value="DB9Studio", style=style, layout=wide)
run_install = W.Checkbox(description="Run_Install", value=False, indent=False)
btn_setup = W.Button(description="Run setup / verify", button_style="primary", icon="check")
btn_setup.on_click(lambda _: run_cmd("Full setup" if run_install.value else "Verify", "setup_5090.bat" if run_install.value else f'"{py()}" verify.py'))
setup_box = W.VBox([ui_train, run_install, btn_setup])

# 2.1 Data
train_folder = W.Text(description="TrainFolder", value="datasets/raw", style=style, layout=wide)
processed_folder = W.Text(description="ProcessedFolder", value="datasets/processed", style=style, layout=wide)
control_folder = W.Text(description="ControlFolder", value="", style=style, layout=wide)
data_clean = W.Checkbox(description="DataClean", value=True, indent=False)
no_caption = W.Checkbox(description="No_gen_caption", value=False, indent=False)
caption = W.Dropdown(description="Caption", options=["None", "APIGemini | 2.5 Flash", "APIGemini | 2.5 Pro", "APIGemini | 2.5 Flash Lite", "Florence"], value="None", style=style, layout=third)
caption_length = W.Dropdown(description="Caption_Length", options=["Short", "Medium", "Long"], value="Medium", style=style, layout=third)
api_prompt = W.Text(description="API_Prompt", value="Mo ta chi tiet hinh anh nay", style=style, layout=wide)
custom_caption = W.Text(description="Custom_Caption", value="", style=style, layout=half)
append_caption = W.Checkbox(description="Append", value=False, indent=False)
overwrite_caption = W.Checkbox(description="Overwrite_Caption", value=False, indent=False)
min_res = W.IntText(description="Min_Resolution", value=512, style=style, layout=third)
max_res = W.IntText(description="Max_Resolution", value=4096, style=style, layout=third)
img_format = W.Dropdown(description="Image_Format", options=["jpg", "png", "webp"], value="jpg", style=style, layout=third)
btn_open_data = W.Button(description="Open Data Folder", icon="folder-open")
btn_clean = W.Button(description="Run DataClean", button_style="info", icon="image")
btn_caption = W.Button(description="Run Caption", button_style="info", icon="comment")
btn_open_data.on_click(lambda _: open_folder(PROJECT_ROOT / train_folder.value))
btn_clean.on_click(lambda _: run_cmd("Data clean", f'"{py()}" scripts/data_clean.py --input "{train_folder.value}" --output "{processed_folder.value}/img" --min-res {min_res.value} --max-res {max_res.value} --format {img_format.value}'))

def run_caption(_):
    if no_caption.value or caption.value == "None":
        with log: print("Caption skipped.")
        return
    model_map = {"APIGemini | 2.5 Flash":"gemini-2.5-flash", "APIGemini | 2.5 Pro":"gemini-2.5-pro", "APIGemini | 2.5 Flash Lite":"gemini-2.5-flash-lite"}
    if caption.value.startswith("APIGemini"):
        cmd = f'"{py()}" scripts/caption_gemini.py --input "{processed_folder.value}/img" --output "{processed_folder.value}/captions" --model {model_map[caption.value]} --length {caption_length.value} --prompt "{api_prompt.value}"'
        if custom_caption.value: cmd += f' --custom-caption "{custom_caption.value}"'
        if append_caption.value: cmd += ' --append'
        if overwrite_caption.value: cmd += ' --overwrite'
    else:
        cmd = f'"{py()}" scripts/caption_florence.py --input "{processed_folder.value}/img" --output "{processed_folder.value}/captions" --length {caption_length.value}'
    run_cmd("Caption", cmd)
btn_caption.on_click(run_caption)
data_box = W.VBox([note("Train nhieu lora bang cach nhap nhieu thu muc, cach nhau bang dau phay (,)") , train_folder, note("ControlFolder dung cho img2img/upscale; filename phai khop voi anh target"), control_folder, W.HBox([data_clean, no_caption]), note("OCR Prompt / API Caption"), W.HBox([caption, W.Text(description="APIkey", value="From .env only", disabled=True, style=style, layout=third), caption_length]), api_prompt, W.HBox([custom_caption, append_caption, overwrite_caption]), W.HBox([min_res, max_res, img_format]), W.HBox([btn_open_data, btn_clean, btn_caption])])

# 2.1b Upscale
tile_enable = W.Checkbox(description="Use_Upscale_Detail_Tiles", value=False, indent=False)
tile_size = W.IntText(description="Tile_Size", value=2048, style=style, layout=third)
tile_overlap = W.IntText(description="Tile_Overlap", value=256, style=style, layout=third)
max_tiles = W.IntText(description="Max_Tiles_Per_Image", value=0, style=style, layout=third)
min_detail = W.FloatText(description="Min_Detail_Score", value=3.0, style=style, layout=third)
degrade_scale = W.FloatText(description="Degrade_Scale", value=2.0, style=style, layout=third)
blur_radius = W.FloatText(description="Blur_Radius", value=1.2, style=style, layout=third)
jpeg_quality = W.IntText(description="Jpeg_Quality", value=45, style=style, layout=third)
tile_caption = W.Text(description="Tile_Caption", value="high detail upscale restoration, sharp architectural detail", style=style, layout=wide)
btn_tiles = W.Button(description="Prepare Upscale Tiles", button_style="warning", icon="th-large")
def run_tiles(_):
    if not tile_enable.value:
        with log: print("Enable Use_Upscale_Detail_Tiles first.")
        return
    cmd = f'"{py()}" scripts/prepare_upscale_detail_dataset.py --input "{train_folder.value}" --output "{processed_folder.value}" --tile-size {tile_size.value} --overlap {tile_overlap.value} --max-tiles-per-image {max_tiles.value} --min-detail-score {min_detail.value} --degrade-scale {degrade_scale.value} --blur-radius {blur_radius.value} --jpeg-quality {jpeg_quality.value} --caption "{tile_caption.value}" --overwrite'
    run_cmd("Prepare upscale detail tiles", cmd)
    control_folder.value = str(Path(processed_folder.value) / "control")
    training_mode.value = "img2img_upscale"
btn_tiles.on_click(run_tiles)
tile_box = W.VBox([tile_enable, W.HBox([tile_size, tile_overlap, max_tiles]), W.HBox([min_detail, degrade_scale, blur_radius, jpeg_quality]), tile_caption, btn_tiles])


# 2.1c User-provided size degrade pairs
pair_input = W.Text(description="InputBlurFolder", value="datasets/input_blur", style=style, layout=wide)
pair_output = W.Text(description="SharpOutputFolder", value="datasets/output_sharp", style=style, layout=wide)
pair_processed = W.Text(description="PairProcessedFolder", value="datasets/processed", style=style, layout=wide)
pair_caption = W.Text(description="Fallback_Caption", value="upscale restoration, sharp high quality result", style=style, layout=wide)
pair_overwrite = W.Checkbox(description="Overwrite_Pairs", value=True, indent=False)
btn_open_pair_input = W.Button(description="Open Input Blur", icon="folder-open")
btn_open_pair_output = W.Button(description="Open Sharp Output", icon="folder-open")
btn_pairs = W.Button(description="Build Size-Degrade Pairs", button_style="warning", icon="exchange")
btn_open_pair_input.on_click(lambda _: open_folder(PROJECT_ROOT / pair_input.value))
btn_open_pair_output.on_click(lambda _: open_folder(PROJECT_ROOT / pair_output.value))
def run_pairs(_):
    cmd = f'"{py()}" scripts/build_size_degrade_pairs.py --input "{pair_input.value}" --output "{pair_output.value}" --processed "{pair_processed.value}" --caption "{pair_caption.value}"'
    if pair_overwrite.value:
        cmd += ' --overwrite'
    run_cmd("Build size-degrade pairs", cmd)
    processed_folder.value = pair_processed.value
    control_folder.value = str(Path(pair_processed.value) / "control")
    training_mode.value = "img2img_upscale"
btn_pairs.on_click(run_pairs)
pair_box = W.VBox([
    note("1 folder output sharp co image+caption theo ten goc; 1 folder input blur co file dang tenanh_200, tenanh_2048"),
    pair_output,
    pair_input,
    pair_processed,
    pair_caption,
    pair_overwrite,
    W.HBox([btn_open_pair_output, btn_open_pair_input, btn_pairs]),
])

# 2.2 Config
type_train = W.Dropdown(description="TypeTrain", options=["FLUX.2-klein-base-9B", "Flux"], value="FLUX.2-klein-base-9B", style=style, layout=wide)
training_mode = W.Dropdown(description="Training_Mode", options=["text2img", "img2img_upscale"], value="text2img", style=style, layout=third)
low_vram = W.Checkbox(description="Low_VRAM", value=False, indent=False)
lora_name = W.Text(description="Lora_name", value="db9_toolkit_trainner", style=style, layout=half)
output_folder = W.Text(description="OutputFolder", value="outputs", style=style, layout=half)
steps = W.IntText(description="Steps", value=2000, style=style, layout=third)
save_steps = W.IntText(description="Save_steps", value=500, style=style, layout=third)
sample_steps = W.IntText(description="Sample_steps", value=500, style=style, layout=third)
resolution = W.Text(description="Resolution", value="1536", style=style, layout=wide)
batch_size = W.IntText(description="Batch_size", value=6, style=style, layout=third)
lr = W.FloatText(description="Lr", value=4e-4, style=style, layout=third)
lr_scheduler = W.Dropdown(description="Lr_Scheduler", options=["constant", "constant_with_warmup", "cosine", "linear", "cosine_with_restarts", "polynomial"], value="constant_with_warmup", style=style, layout=third)
dim = W.IntText(description="Dim", value=64, style=style, layout=third)
alpha = W.IntText(description="Alpha", value=64, style=style, layout=third)
optimizer = W.Dropdown(description="Optimizer", options=["adamw8bit", "adam", "adamw"], value="adamw8bit", style=style, layout=third)
grad_ckpt = W.Checkbox(description="Gradient_Checkpointing", value=True, indent=False)
bucketing = W.Checkbox(description="Enable_Bucketing", value=True, indent=False)
min_bucket = W.IntText(description="Min_Bucket_Reso", value=512, style=style, layout=third)
max_bucket = W.IntText(description="Max_Bucket_Reso", value=2048, style=style, layout=third)
bucket_step = W.IntText(description="Bucket_step", value=64, style=style, layout=third)
sample_prompt = W.Text(description="Sampler_Prompt", value="A portrait of a person in DB9 style, highly detailed", style=style, layout=wide)
btn_config = W.Button(description="Generate Config", button_style="success", icon="cog")
def gen_config(_):
    from scripts.config_generator import generate_config
    res = int(str(resolution.value).split(',')[0].strip())
    bs = min(batch_size.value, 1) if low_vram.value else batch_size.value
    model_path = "black-forest-labs/FLUX.2-klein-base-9B" if type_train.value == "FLUX.2-klein-base-9B" else "black-forest-labs/FLUX.1-dev"
    control_path = ""
    if training_mode.value != "text2img":
        if not control_folder.value: raise ValueError("ControlFolder is required for img2img_upscale")
        control_path = str((PROJECT_ROOT / control_folder.value).resolve()).replace('\\', '/')
    yml = generate_config(project_name=lora_name.value, model_path=model_path, dataset_path=str((PROJECT_ROOT / processed_folder.value).resolve()).replace('\\', '/'), output_path=str((PROJECT_ROOT / output_folder.value).resolve()).replace('\\', '/'), lora_rank=dim.value, lora_alpha=alpha.value, batch_size=bs, learning_rate=lr.value, train_steps=steps.value, gradient_accumulation=1, resolution=res, enable_bucketing=bucketing.value, bucket_step=bucket_step.value, min_bucket_reso=min_bucket.value, max_bucket_reso=max_bucket.value, flip_aug=False, color_aug=False, optimizer=optimizer.value, lr_scheduler=lr_scheduler.value, warmup_steps=100, gradient_checkpointing=grad_ckpt.value, quantize=False, save_every=save_steps.value, sample_every=sample_steps.value, sample_prompts=[sample_prompt.value] if sample_prompt.value else [], training_mode=training_mode.value, control_folder_path=control_path)
    Path('configs').mkdir(exist_ok=True)
    path = Path('configs') / f'{lora_name.value}.yaml'
    path.write_text(yml, encoding='utf-8')
    with log: print(f"Config saved: {path} | Resolution: {res}px | Mode: {training_mode.value}")
btn_config.on_click(gen_config)
config_box = W.VBox([type_train, note("Nen dung Low_VRAM neu bi OOM"), W.HBox([training_mode, low_vram, grad_ckpt, bucketing]), W.HBox([lora_name, output_folder]), W.HBox([steps, save_steps, sample_steps]), resolution, W.HBox([batch_size, lr, lr_scheduler]), W.HBox([dim, alpha, optimizer]), W.HBox([min_bucket, max_bucket, bucket_step]), sample_prompt, btn_config])

# Train
run_train_flag = W.Checkbox(description="RunTrain", value=False, indent=False)
btn_train = W.Button(description="Run Lora Train", button_style="danger", icon="play")
def train(_):
    cfg = Path('configs') / f'{lora_name.value}.yaml'
    if not cfg.exists():
        with log: print("Missing config. Click Generate Config first.")
        return
    if not run_train_flag.value:
        with log: print("Set RunTrain=True before starting training.")
        return
    run_cmd("Train", f'cd ai-toolkit; "{py()}" run.py ../{cfg.as_posix()}')
btn_train.on_click(train)
train_box = W.VBox([W.HBox([run_train_flag, btn_train])])

accordion = W.Accordion(children=[setup_box, data_box, tile_box, pair_box, config_box, train_box])
for i, title in enumerate(["☕ 1. Cai dat", "✨ 2.1 Xu ly du lieu", "🧩 2.1b Upscale detail", "⚙️ 2.2 Cai dat train", "🧪 3. Train"]):
    accordion.set_title(i, title)

display(HTML('''<style>div.input{display:none;} div.prompt{display:none;} .widget-label{font-weight:600;}</style><h2>DB9Studio Toolkit Training</h2><div style="background:#2b2f33;border:1px solid #555;border-radius:6px;padding:6px 10px;color:#eee;font-family:monospace;display:inline-block;margin-bottom:8px">Code hidden - use controls below - Flux 2 Klein max 2048px</div>'''))
display(accordion)
display(HTML('<h3>Log</h3>'))
display(log)
