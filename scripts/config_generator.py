import yaml


# DB9-compatible bucket ratios (matching db9_flux_locked_upscale tile sizes)
# Tile sizes: 512-2048 step 64, supports non-square tiles
DB9_BUCKET_RATIOS = [
    (1, 1),    # 1:1   square
    (3, 4),    # 3:4   portrait
    (4, 3),    # 4:3   landscape
    (2, 3),    # 2:3   portrait
    (3, 2),    # 3:2   landscape
    (9, 16),   # 9:16  tall portrait
    (16, 9),   # 16:9  wide landscape
    (1, 2),    # 1:2   tall
    (2, 1),    # 2:1   wide
]


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
    # --- New parameters for img2img / upscale training ---
    training_mode: str = "text2img",
    control_folder_path: str = "",
    trigger_word: str = "",
) -> str:
    """
    Generate ai-toolkit YAML config.
    
    training_mode:
      - "text2img"        : Standard LoRA (text prompt -> image)
      - "img2img_upscale" : Upscale LoRA (degraded input -> sharp output)
      - "img2img_kontext" : Kontext/Edit LoRA (control image -> result image)
    
    control_folder_path:
      Required for img2img modes. Path to the folder containing
      condition/control images (e.g., degraded/blurred images).
      File names must match the training images.
    
    Returns:
        YAML string ready to write to file
    """
    
    # Validate training mode
    valid_modes = ("text2img", "img2img_upscale", "img2img_kontext")
    if training_mode not in valid_modes:
        raise ValueError(f"training_mode must be one of {valid_modes}, got '{training_mode}'")
    
    if training_mode != "text2img" and not control_folder_path:
        raise ValueError(f"control_folder_path is required for training_mode='{training_mode}'")
    
    # Build dataset config
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
    
    # Add control/condition folder for img2img modes
    if training_mode in ("img2img_upscale", "img2img_kontext"):
        dataset_config["control_path"] = control_folder_path
        # For upscale training, disable flip_aug to keep pairs aligned
        if training_mode == "img2img_upscale":
            dataset_config["flip_aug"] = False
            dataset_config["random_crop"] = False
    
    # Add trigger word to caption if specified
    if trigger_word:
        dataset_config["default_caption"] = trigger_word
    
    # Determine trainer type
    trainer_type = "sd_trainer"
    if training_mode == "img2img_kontext":
        trainer_type = "sd_trainer"  # ai-toolkit uses same trainer with control_path
    
    # Build model config
    model_config = {
        "name_or_path": model_path,
        "is_flux": True,
        "quantize": quantize,
    }
    
    # Build train config
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
        "noise_scheduler": "flowmatch",
        "save_every": save_every,
        "sample_every": sample_every,
        "sample_prompts": sample_prompts if sample_prompts else [],
    }
    
    # Build sample config
    sample_config = {
        "sampler": "flowmatch",
        "sample_steps": 20,
        "cfg_scale": 1.0,
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
        },
    }
    
    yaml_str = yaml.safe_dump(config_dict, sort_keys=False, allow_unicode=True)
    return "---\n" + yaml_str
