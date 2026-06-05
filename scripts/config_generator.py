import yaml

FLUX2_KLEIN_MAX_RESOLUTION = 2048

# DB9-compatible bucket ratios (matching db9_flux_locked_upscale tile sizes)
# Tile sizes: 512-2048 step 64, supports non-square tiles
DB9_BUCKET_RATIOS = [
    (1, 1),
    (3, 4),
    (4, 3),
    (2, 3),
    (3, 2),
    (9, 16),
    (16, 9),
    (1, 2),
    (2, 1),
]


def _is_flux2_klein(model_path: str) -> bool:
    normalized = model_path.lower()
    return "flux.2-klein" in normalized or "flux2-klein" in normalized


def validate_training_config(
    model_path: str,
    resolution: int,
    enable_bucketing: bool,
    bucket_step: int,
    min_bucket_reso: int,
    max_bucket_reso: int,
) -> None:
    if resolution <= 0:
        raise ValueError("resolution must be greater than 0")

    if _is_flux2_klein(model_path) and resolution > FLUX2_KLEIN_MAX_RESOLUTION:
        raise ValueError(
            "Flux 2 Klein supports a maximum training/sample resolution of "
            f"{FLUX2_KLEIN_MAX_RESOLUTION}px; got {resolution}px"
        )

    if min_bucket_reso <= 0 or max_bucket_reso <= 0:
        raise ValueError("bucket resolutions must be greater than 0")
    if min_bucket_reso > max_bucket_reso:
        raise ValueError("min_bucket_reso cannot be greater than max_bucket_reso")
    if bucket_step <= 0:
        raise ValueError("bucket_step must be greater than 0")
    if min_bucket_reso % bucket_step != 0:
        raise ValueError("min_bucket_reso must be divisible by bucket_step")
    if max_bucket_reso % bucket_step != 0:
        raise ValueError("max_bucket_reso must be divisible by bucket_step")

    if enable_bucketing:
        if resolution < min_bucket_reso or resolution > max_bucket_reso:
            raise ValueError(
                "resolution must be within [min_bucket_reso, max_bucket_reso] "
                "when bucketing is enabled"
            )

    if _is_flux2_klein(model_path) and max_bucket_reso > FLUX2_KLEIN_MAX_RESOLUTION:
        raise ValueError(
            "Flux 2 Klein max_bucket_reso cannot exceed "
            f"{FLUX2_KLEIN_MAX_RESOLUTION}px; got {max_bucket_reso}px"
        )


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
    sample_prompts: list,
    training_mode: str = "text2img",
    control_folder_path: str = "",
    trigger_word: str = "",
    model_type: str = "flux",
) -> str:
    """Generate ai-toolkit YAML config."""
    valid_modes = ("text2img", "img2img_upscale", "img2img_kontext")
    if training_mode not in valid_modes:
        raise ValueError(f"training_mode must be one of {valid_modes}, got '{training_mode}'")

    if training_mode != "text2img" and not control_folder_path:
        raise ValueError(f"control_folder_path is required for training_mode='{training_mode}'")

    validate_training_config(
        model_path=model_path,
        resolution=resolution,
        enable_bucketing=enable_bucketing,
        bucket_step=bucket_step,
        min_bucket_reso=min_bucket_reso,
        max_bucket_reso=max_bucket_reso,
    )

    dataset_config = {
        "folder_path": f"{dataset_path}/img",
        "caption_ext": "txt",
        "caption_folder": f"{dataset_path}/captions",
        "resolution": [resolution, resolution],
        "enable_bucket": enable_bucketing,
        "bucket_step_size": bucket_step,
        "min_bucket_reso": min_bucket_reso,
        "max_bucket_reso": max_bucket_reso,
        "random_crop": False,
        "flip_aug": flip_aug,
        "color_aug": color_aug,
    }

    if training_mode in ("img2img_upscale", "img2img_kontext"):
        dataset_config["control_path"] = control_folder_path
        if training_mode == "img2img_upscale":
            dataset_config["flip_aug"] = False
            dataset_config["random_crop"] = False

    if trigger_word:
        dataset_config["default_caption"] = trigger_word

    trainer_type = "sd_trainer"

    model_config = {
        "name_or_path": model_path,
        "quantize": quantize,
    }
    if model_type.lower() == "flux":
        model_config["is_flux"] = True
        if _is_flux2_klein(model_path):
            model_config["use_text_encoder_2"] = False
    elif model_type.lower() == "sdxl":
        model_config["is_xl"] = True
    elif model_type.lower() == "sd15":
        model_config["is_v2"] = False

    train_config = {
        "batch_size": batch_size,
        "steps": train_steps,
        "gradient_accumulation_steps": gradient_accumulation,
        "train_unet": True,
        "train_text_encoder": False,
        "lr": learning_rate,
        "lr_scheduler": lr_scheduler,
        "lr_warmup_steps": warmup_steps,
        "optimizer": optimizer,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": gradient_checkpointing,
        "save_every": save_every,
        "sample_every": sample_every,
        "sample_prompts": sample_prompts if sample_prompts else [],
    }

    if model_type.lower() == "flux":
        train_config["noise_scheduler"] = "flowmatch"

    sample_config = {
        "sampler": "flowmatch" if model_type.lower() == "flux" else "euler_a",
        "sample_steps": 20,
        "cfg_scale": 1.0 if model_type.lower() == "flux" else 7.0,
        "width": resolution,
        "height": resolution,
    }

    config_dict = {
        "job": "extension",
        "config": {
            "name": project_name,
            "process": [
                {
                    "type": trainer_type,
                    "training_folder": f"{output_path}/{project_name}",
                    "device": "cuda:0",
                    "model": model_config,
                    "network": {
                        "type": "lora",
                        "linear": lora_rank,
                        "linear_alpha": lora_alpha,
                    },
                    "datasets": [dataset_config],
                    "train": train_config,
                    "logging": {
                        "log_every": 10,
                        "use_wandb": False,
                    },
                    "sample": sample_config,
                }
            ],
        },
        "meta": {
            "name": project_name,
            "version": "1.0",
            "training_mode": training_mode,
            "db9_compatible": True,
            "db9_bucket_step": bucket_step,
            "db9_min_reso": min_bucket_reso,
            "db9_max_reso": max_bucket_reso,
            "flux2_klein_max_resolution": FLUX2_KLEIN_MAX_RESOLUTION,
        },
    }

    yaml_str = yaml.safe_dump(config_dict, sort_keys=False, allow_unicode=True)
    return "---\n" + yaml_str
