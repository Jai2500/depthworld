# Training entry for CrtlWorld (horiz layout) + optional pointmap aux loss
# through a latent-fed DPT head.
#
# Thin wrapper around `scripts/train_wm.py`. Parses the pointmap CLI flags,
# monkey-patches the model class to `CrtlWorldPointmap` (with the decoder
# attached in __init__), patches the horiz dataset to the pointmap-GT
# variant, then forwards everything else to `train_wm.main(args)`.
#
# Without `--pointmap_enabled` this is equivalent to running train_wm.py
# directly with `--dataset_class rgb_depth_horiz` (plain horiz training).
#
# Compatible with the env-var warm-start hook in train_wm.py: set
# CTRLWORLD_WARM_START_UNET_CKPT=<path>_dcp_merged.pt to bootstrap from a
# plain CrtlWorld checkpoint (e.g. horiz-40k). Pointmap params stay at
# fresh init; the strict=False load handles that.

from __future__ import annotations

import argparse
import os
import sys

# insert(0), not append: this env ships an unrelated top-level `scripts`
# package in site-packages that would otherwise shadow ours.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_pointmap_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--pointmap_enabled", action="store_true")
    p.add_argument(
        "--pointmap_dpt_init",
        choices=["random", "vggt"],
        default="random",
        help=(
            "Initialization for the DPT pointmap head. `vggt` copies VGGT's "
            "direct XYZ+confidence point head."
        ),
    )
    p.add_argument(
        "--pointmap_vggt_ckpt",
        type=str,
        default="",
        help="Optional path to VGGT model.pt for --pointmap_dpt_init=vggt.",
    )
    p.add_argument("--pointmap_loss_weight", type=float, default=0.05)
    p.add_argument("--pointmap_drop_top_pct", type=float, default=0.05)
    p.add_argument("--pointmap_no_log_space", action="store_true")
    p.add_argument("--pointmap_min_snr_gamma", type=float, default=5.0)
    p.add_argument("--pointmap_conf_alpha", type=float, default=0.2)
    p.add_argument("--pointmap_robust_c", type=float, default=0.05)
    p.add_argument("--pointmap_robust_alpha", type=float, default=0.5)
    p.add_argument("--pointmap_dpt_base_ch", type=int, default=64)
    p.add_argument("--pointmap_dpt_features", type=int, default=256)

    args, remaining = p.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    return args


def main():
    pm_cli = _parse_pointmap_args()

    import scripts.train_wm as train_wm
    from models.ctrl_world_pointmap import CrtlWorldPointmap
    import dataset.dataset_droid as _horiz_mod

    class CrtlWorldPointmapInitialized(CrtlWorldPointmap):
        def __init__(self_inner, args, num_history=None):
            super().__init__(args, num_history=num_history)
            if pm_cli.pointmap_enabled:
                from models.pointmap_decoder import DPTPointMapDecoder

                if pm_cli.pointmap_dpt_init == "vggt":
                    decoder = DPTPointMapDecoder.from_vggt(
                        base_ch=pm_cli.pointmap_dpt_base_ch,
                        features=pm_cli.pointmap_dpt_features,
                        vggt_ckpt_path=pm_cli.pointmap_vggt_ckpt or None,
                    )
                else:
                    decoder = DPTPointMapDecoder(
                        base_ch=pm_cli.pointmap_dpt_base_ch,
                        features=pm_cli.pointmap_dpt_features,
                    )
                self_inner.attach_pointmap_decoder(
                    decoder,
                    loss_weight=pm_cli.pointmap_loss_weight,
                    drop_top_pct=pm_cli.pointmap_drop_top_pct,
                    loss_in_log=not pm_cli.pointmap_no_log_space,
                    min_snr_gamma=pm_cli.pointmap_min_snr_gamma,
                    conf_alpha=pm_cli.pointmap_conf_alpha,
                    robust_c=pm_cli.pointmap_robust_c,
                    robust_alpha=pm_cli.pointmap_robust_alpha,
                )
                n_pm = sum(p.numel() for p in decoder.parameters())
                print(
                    f"[pointmap_decoder] params={n_pm/1e6:.2f}M "
                    f"dpt_init={pm_cli.pointmap_dpt_init} "
                    f"loss_weight={pm_cli.pointmap_loss_weight}",
                    flush=True,
                )

    # Monkey patches. `train_wm.main()` resolves CrtlWorld and the horiz
    # dataset at call time, so these stick.
    train_wm.CrtlWorld = CrtlWorldPointmapInitialized
    if pm_cli.pointmap_enabled:
        from dataset.dataset_droid_pointmap import (
            Dataset_mix_horiz_with_pointmap,
        )

        _horiz_mod.Dataset_mix_horiz = Dataset_mix_horiz_with_pointmap

    # Argparse — copy from train_wm.py's bottom-of-file block. Stable surface.
    from argparse import ArgumentParser
    from config import wm_args

    parser = ArgumentParser()
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_on_cpu", action="store_true")
    parser.add_argument("--resume_from_dir", type=str, default=None)
    parser.add_argument("--svd_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--no_ckpt", action="store_true")
    parser.add_argument("--dataset_root_path", type=str, default=None)
    parser.add_argument("--dataset_meta_info_path", type=str, default=None)
    parser.add_argument("--dataset_names", type=str, default=None)
    parser.add_argument(
        "--action_space",
        type=str,
        default=None,
        choices=["cartesian", "cartesian_position", "ee", "eef", "end_effector", "joint_position", "joint", "joints", "joint_pos", "qpos"],
        help=(
            "Action conditioning source. Default cartesian = normalized "
            "6D EE pose + gripper. joint_position = normalized Franka "
            "7D joint positions + gripper using robot joint limits."
        ),
    )
    parser.add_argument("--dataset_cfgs", type=str, default=None)
    parser.add_argument(
        "--dataset_class", type=str, default="rgb_depth_horiz",
        choices=["rgb_only", "rgb_depth_horiz"],
    )
    parser.add_argument("--down_sample", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--validation_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--wandb_project_name", type=str, default=None)
    parser.add_argument("--gripper_filter_mode", type=str, default=None)
    parser.add_argument("--gripper_filter_sample_prob", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    args_new = parser.parse_args()
    if args_new.dataset_cfgs is None:
        args_new.dataset_cfgs = args_new.dataset_names

    args = wm_args()
    args.ckpt_path = None  # disable upstream default; warm-start hook handles loading
    args.wandb_project_name = "depth_wm"

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None and v is not False:
                args.__dict__[k] = v
        return args

    args = merge_args(args, args_new)
    train_wm._apply_action_space_config(args)
    if args_new.svd_model_path is not None:
        args.pretrained_model_path = args_new.svd_model_path
    if args_new.clip_model_path is not None:
        args.clip_model_path = args_new.clip_model_path
    args.use_swanlab = False
    if args_new.no_ckpt:
        args.ckpt_path = None
    if args_new.tag is not None:
        args.output_dir = f"model_ckpt/{args.tag}"
        args.wandb_run_name = args.tag
    args.dataset_class = args_new.dataset_class

    args.pointmap_cli = vars(pm_cli)
    print(
        f"[train_wm_horiz_pointmap] pointmap={pm_cli.pointmap_enabled} "
        f"dpt_init={pm_cli.pointmap_dpt_init} "
        f"loss_weight={pm_cli.pointmap_loss_weight}",
        flush=True,
    )

    train_wm.main(args)


if __name__ == "__main__":
    main()
