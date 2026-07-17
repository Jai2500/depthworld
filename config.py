import os
from dataclasses import dataclass

import torch


@dataclass
class wm_args:
    ########################### training args ##############################
    # Model paths. Point these at local snapshots of the frozen backbones —
    # via env vars, the CLI flags (--svd_model_path / --clip_model_path /
    # --ckpt_path), or by placing them under checkpoints/ (see
    # checkpoints/README.md).
    pretrained_model_path = os.environ.get(
        "CTRLWORLD_SVD_MODEL_PATH", "checkpoints/stable-video-diffusion-img2vid"
    )
    clip_model_path = os.environ.get(
        "CTRLWORLD_CLIP_MODEL_PATH", "checkpoints/clip-vit-base-patch32"
    )
    # Optional base world-model checkpoint to initialize from. Training
    # entrypoints normally leave this unset (--no_ckpt) and warm-start via
    # the CTRLWORLD_WARM_START_UNET_CKPT env hook instead.
    ckpt_path = os.environ.get("CTRLWORLD_BASE_CKPT") or None

    # dataset parameters
    # raw data
    dataset_root_path = "dataset_example"
    dataset_names = "droid_subset"
    split = "val"
    # meta info
    dataset_meta_info_path = "dataset_meta_info"
    dataset_cfgs = dataset_names
    prob = [1.0]
    annotation_name = "annotation"  #'annotation_all_skip1'
    num_workers = 4
    down_sample = 3  # downsample 15hz to 5hz
    skip_step = 1
    read_folder_for_id = False
    dataset_start_idx = -1
    dataset_end_idx = -1

    # logs parameters
    debug = False
    tag = "doird_subset"
    output_dir = f"model_ckpt/{tag}"
    wandb_run_name = tag
    wandb_project_name = "droid_example"

    # training parameters
    # Global seed for torch + numpy + python random + CUDA. Setting this
    # makes runs reproducible across submissions (same shuffle order, same
    # init noise); unseeded runs cannot do meaningful skip-first-batches
    # on warm-start.
    seed = 42
    learning_rate = 1e-5  # 5e-6
    gradient_accumulation_steps = 1
    mixed_precision = "fp16"
    train_batch_size = 4
    shuffle = True
    num_train_epochs = 100
    max_train_steps = 500000
    checkpointing_steps = 2000
    validation_steps = 2500
    max_grad_norm = 1.0
    # for val
    # Number of validation videos generated per fire of validate_video_generation.
    # Each video costs ~50 diffusion steps × dual-UNet forward + VAE decode ≈ 2-3 min
    # in production. With NCCL watchdog at 10 min, video_num × per-video cost MUST
    # stay well under 600 s — otherwise non-main ranks hit barrier timeout while
    # rank 0 is still sampling. Default reduced from 10 → 2 (2026-05-17 fix).
    video_num = 2

    ############################ model args ##############################

    # model parameters
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 2  # 7.5 #7.5 #7.5 #3.0
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320
    height = 192
    # num history and num future predictions
    num_frames = 5  # NOTE:
    num_history = 6  # NOTE: The history is actually 6 and not 7.
    action_dim = 7
    # cartesian: 6D EE pose + gripper; joint_position: 7D Franka qpos + gripper.
    action_space = "cartesian"
    text_cond = True
    frame_level_cond = True
    his_cond_zero = False
    dtype = torch.bfloat16  # [torch.float32, torch.bfloat16] # during inference, we can use bfloat16 to accelerate the inference speed and save memory

    ########################### prediction args #########################
    pred_step = 5  # predict 5 steps (1s) of action per chunk
    # Sparse-history frame offsets used by autoregressive rollout.
    history_idx = [0, 0, -12, -9, -6, -3]
    # Checkpoint used by validate_video_generation during training.
    val_model_path = ckpt_path
