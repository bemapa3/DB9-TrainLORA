import json
import sys
from pathlib import Path

import yaml

sys.path.append("scripts")
from config_generator import generate_config

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


def verify_notebooks() -> None:
    sources = {name: read_notebook(name) for name in NOTEBOOKS}
    checks = {
        "project name in setup": PROJECT_DISPLAY_NAME in sources["1_Setup.ipynb"],
        "ai-toolkit idempotent setup": "toolkit_dir.exists()" in sources["1_Setup.ipynb"],
        "Gemini key guard": "GEMINI_API_KEY" in sources["2_Prepare_Dataset.ipynb"],
        "project id in config": PROJECT_NAME in sources["3_Train_Config.ipynb"],
        "Flux 2 Klein model in config": MODEL_ID in sources["3_Train_Config.ipynb"],
        "train config guard": "Missing config" in sources["4_Train.ipynb"],
        "test checkpoint discovery": "rglob(\"*.safetensors\")" in sources["5_Test_LoRA.ipynb"],
        "Flux 2 Klein model in test": MODEL_ID in sources["5_Test_LoRA.ipynb"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        fail("Notebook checks failed: " + ", ".join(failed))


def verify_config_generator() -> None:
    yml = generate_config(
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
        resolution=1024,
        enable_bucketing=True,
        bucket_step=64,
        min_bucket_reso=512,
        max_bucket_reso=2048,
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
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        fail("Config checks failed: " + ", ".join(failed))


def main() -> None:
    verify_notebooks()
    verify_config_generator()
    print("OK: DB9-Toolkit-Trainner workflow smoke checks passed.")


if __name__ == "__main__":
    main()
