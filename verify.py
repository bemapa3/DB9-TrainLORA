import json
import sys
from pathlib import Path

import yaml

sys.path.append("scripts")
from config_generator import FLUX2_KLEIN_MAX_RESOLUTION, generate_config

PROJECT_DISPLAY_NAME = "DB9-Toolkit-Trainner"
PROJECT_NAME = "db9_toolkit_trainner"
MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    "1_Setup.ipynb",
    "2_Prepare_Dataset.ipynb",
    "3_Train_Config.ipynb",
    "4_Train.ipynb",
    "5_Test_LoRA.ipynb",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def read_notebook(name: str) -> str:
    path = ROOT / "notebooks" / name
    if not path.exists():
        fail(f"Missing notebook: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("nbformat") != 4:
        fail(f"Unexpected nbformat in {name}: {data.get('nbformat')}")
    return "\n".join("".join(cell.get("source", [])) for cell in data.get("cells", []))


def make_config(resolution: int = 1024, max_bucket_reso: int = 2048) -> str:
    return generate_config(
        project_name=PROJECT_NAME,
        model_path=MODEL_ID,
        dataset_path="datasets/processed",
        output_path="outputs",
        lora_rank=64,
        lora_alpha=64,
        batch_size=6,
        learning_rate=4e-4,
        train_steps=100,
        gradient_accumulation=1,
        resolution=resolution,
        enable_bucketing=True,
        bucket_step=64,
        min_bucket_reso=512,
        max_bucket_reso=max_bucket_reso,
        flip_aug=False,
        color_aug=False,
        optimizer="adamw8bit",
        lr_scheduler="constant_with_warmup",
        warmup_steps=10,
        gradient_checkpointing=True,
        quantize=False,
        save_every=50,
        sample_every=50,
        sample_prompts=['A prompt with "quotes"', "And : colons", "[brackets]"],
    )


def verify_notebooks() -> None:
    sources = {name: read_notebook(name) for name in NOTEBOOKS}
    local_training = read_notebook("0_DB9_Toolkit_Training_Local.ipynb")
    checks = {
        "DB9Studio-style local notebook": "Prepare Upscale Detail Dataset" in local_training,
        "upscale detail mode param": "img2img_upscale" in local_training,
        "project name in setup": PROJECT_DISPLAY_NAME in sources["1_Setup.ipynb"],
        "ai-toolkit idempotent setup": "toolkit_dir.exists()" in sources["1_Setup.ipynb"],
        "Gemini key guard": "GEMINI_API_KEY" in sources["2_Prepare_Dataset.ipynb"],
        "project id in config": PROJECT_NAME in sources["3_Train_Config.ipynb"],
        "Flux 2 Klein model in config": MODEL_ID in sources["3_Train_Config.ipynb"],
        "Flux 2 Klein warning cell": "Flux 2 Klein Validation" in sources["3_Train_Config.ipynb"],
        "Flux 2 Klein 2048 notebook guard": "FLUX2_KLEIN_MAX_RESOLUTION = 2048" in sources["3_Train_Config.ipynb"],
        "train config guard": "Missing config" in sources["4_Train.ipynb"],
        "test checkpoint discovery": "rglob(\"*.safetensors\")" in sources["5_Test_LoRA.ipynb"],
        "Flux 2 Klein model in test": MODEL_ID in sources["5_Test_LoRA.ipynb"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        fail("Notebook checks failed: " + ", ".join(failed))


def verify_caption_gemini() -> None:
    source = (ROOT / "scripts" / "caption_gemini.py").read_text(encoding="utf-8")
    checks = {
        "no api-key cli flag": "--api-key" not in source,
        "env-only key": "os.environ.get(\"GEMINI_API_KEY\"" in source,
        "retry helper": "generate_with_retry" in source,
        "exponential backoff": "2 ** attempt" in source,
        "safety fallback": "SAFETY_FALLBACK_CAPTION" in source,
        "fallback status": "FALLBACK" in source,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        fail("Gemini caption checks failed: " + ", ".join(failed))


def verify_config_generator() -> None:
    yml = make_config(resolution=1024)
    parsed = yaml.safe_load(yml)
    process = parsed["config"]["process"][0]
    dataset = process["datasets"][0]
    train = process["train"]

    checks = {
        "project id": parsed["config"]["name"] == PROJECT_NAME,
        "model id": process["model"]["name_or_path"] == MODEL_ID,
        "dataset image path": dataset["folder_path"] == "datasets/processed/img",
        "dataset caption path": dataset["caption_folder"] == "datasets/processed/captions",
        "flux flag": process["model"].get("is_flux") is True,
        "flowmatch scheduler": train.get("noise_scheduler") == "flowmatch",
        "sample prompts list": isinstance(train.get("sample_prompts"), list),
        "max resolution meta": parsed["meta"].get("flux2_klein_max_resolution") == FLUX2_KLEIN_MAX_RESOLUTION,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        fail("Config checks failed: " + ", ".join(failed))


def verify_flux2_resolution_limits() -> None:
    parsed = yaml.safe_load(make_config(resolution=2048, max_bucket_reso=2048))
    sample = parsed["config"]["process"][0]["sample"]
    if sample["width"] != 2048 or sample["height"] != 2048:
        fail("2048px config did not preserve sample dimensions")

    try:
        make_config(resolution=2560, max_bucket_reso=2560)
    except ValueError as exc:
        if "2048" not in str(exc):
            fail(f"2560px failure did not mention 2048px limit: {exc}")
    else:
        fail("2560px Flux 2 Klein config should fail")


def verify_upscale_detail_script() -> None:
    path = ROOT / "scripts" / "prepare_upscale_detail_dataset.py"
    if not path.exists():
        fail("Missing prepare_upscale_detail_dataset.py")
    source = path.read_text(encoding="utf-8")
    checks = {
        "tile size guard": "--tile-size must be between 1 and 2048" in source,
        "control output": "control_dir" in source,
        "caption output": "caption_path.write_text" in source,
        "detail filter": "min_detail_score" in source,
    }
    pair_path = ROOT / "scripts" / "build_size_degrade_pairs.py"
    if not pair_path.exists():
        fail("Missing build_size_degrade_pairs.py")
    pair_source = pair_path.read_text(encoding="utf-8")
    checks.update({
        "size suffix parser": "SIZE_SUFFIX_RE" in pair_source,
        "sharp output matcher": "find_sharp" in pair_source,
        "pair control output": "control_dir" in pair_source,
        "pair caption copy": "copy_caption" in pair_source,
    })
    ui_source = (ROOT / "scripts" / "db9studio_ui.py").read_text(encoding="utf-8")
    checks.update({
        "DB9Studio hidden UI": "Build Size-Degrade Pairs" in ui_source,
    })
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        fail("Upscale/pair script checks failed: " + ", ".join(failed))


def main() -> None:
    verify_notebooks()
    verify_caption_gemini()
    verify_config_generator()
    verify_flux2_resolution_limits()
    verify_upscale_detail_script()
    print("OK: DB9-Toolkit-Trainner workflow smoke checks passed.")


if __name__ == "__main__":
    main()
