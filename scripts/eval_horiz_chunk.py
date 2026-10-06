"""Single-chunk evaluation for the horiz (RGB+depth horizontal-stack) checkpoint.

  * One `CrtlWorld` UNet. Sampling via `CtrlWorldDiffusionPipeline`.
  * Latents are (T, 4, V*H_lat, 2*W_lat) — RGB and depth packed side-by-side
    along width. Width must be set to 640.
  * After VAE decode, the right half of width is depth-as-grayscale; per-view
    inverse log+range gives metric depth.
  * Checkpoints with a latent-fed DPT pointmap head are auto-detected by
    `build_horiz_model`; the decoder's direct XYZ readout feeds the
    pointmap metrics alongside the depth-unprojection ones.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import wm_args
from scripts.eval_horiz_utils import (
    Dataset_mix_horiz_with_meta,
    PointmapHelper,
    apply_action_space_config,
    build_horiz_model,
    compute_depth_metrics,
    compute_depth_psnr_ssim,
    compute_pointmap_metrics,
    compute_rgb_metrics,
    decode_horiz_latents,
    save_per_sample_videos,
    write_metrics_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument(
        "--svd_model_path",
        default=None,
        help="Override cfg.pretrained_model_path for the frozen SVD VAE/CLIP modules.",
    )
    p.add_argument(
        "--clip_model_path",
        default=None,
        help="Override cfg.clip_model_path for the frozen CLIP text/tokenizer modules.",
    )

    p.add_argument("--dataset_root_path", default="data")
    p.add_argument("--dataset_meta_info_path", default="dataset_meta_info")
    p.add_argument("--dataset_names", default="droid_raw_ctrl")
    p.add_argument("--dataset_cfgs", default="droid_raw_ctrl")
    p.add_argument("--annotation_name", default="annotation")
    p.add_argument(
        "--action_space",
        default="cartesian",
        choices=["cartesian", "cartesian_position", "ee", "eef", "end_effector", "joint_position", "joint", "joints", "joint_pos", "qpos"],
        help=(
            "Action conditioning source. Use joint_position for checkpoints "
            "trained with 7D Franka joint angles + gripper."
        ),
    )
    p.add_argument("--split", default="val", choices=["train", "val"])

    p.add_argument("--num_val_samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_guidance_scale", type=float, default=1.0)
    p.add_argument("--max_guidance_scale", type=float, default=2.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--width", type=int, default=640,
                   help="Horiz latent width (2 * per-view RGB latent width).")

    p.add_argument("--output_dir", default="eval_results_horiz_chunk")
    p.add_argument("--save_videos", action="store_true")

    # Sharding for multi-GPU / SLURM-array parallelism. With default
    # num_shards=1 the behaviour is identical to the single-process case
    # (writes `per_sample.csv` + `aggregate.json`). With num_shards>1,
    # each worker processes ~1/N of the seeded sample list and writes
    # `per_sample_shard{idx:02d}.csv`; run `scripts/aggregate_eval_shards.py`
    # afterwards to produce the canonical merged CSV + aggregate.
    p.add_argument(
        "--shard_idx", type=int, default=0,
        help="This worker's shard index in [0, num_shards).",
    )
    p.add_argument(
        "--num_shards", type=int, default=1,
        help="Total number of shards. Default 1 = no parallelism.",
    )
    p.add_argument(
        "--shard_partition", default="stripe",
        choices=["stripe", "contiguous"],
        help="How to split the seeded indices across shards. 'stripe' picks "
             "every num_shards-th item starting at shard_idx (resilient to "
             "per-sample cost variance); 'contiguous' takes a contiguous "
             "slice.",
    )
    p.add_argument(
        "--save_depth_arrays",
        action="store_true",
        help="Save generated, GT decoded-latent, and raw-disparity metric depth arrays.",
    )

    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    return p.parse_args()


def _summarize_per_view(metric_arr: np.ndarray) -> dict:
    out = {}
    roles = ["ext1", "ext2", "wrist"]
    means_per_view = []
    for v_idx, role in enumerate(roles):
        v_vals = metric_arr[v_idx]
        v_vals = v_vals[~np.isnan(v_vals)]
        m = float(v_vals.mean()) if v_vals.size > 0 else float("nan")
        out[role] = m
        means_per_view.append(m)
    out["ext_mean"] = (
        float(np.nanmean([means_per_view[0], means_per_view[1]]))
        if not (np.isnan(means_per_view[0]) and np.isnan(means_per_view[1]))
        else float("nan")
    )
    flat = metric_arr.flatten()
    flat = flat[~np.isnan(flat)]
    out["all_mean"] = float(flat.mean()) if flat.size > 0 else float("nan")
    return out


def main() -> None:
    args_cli = parse_args()

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    import random as _py_random
    _py_random.seed(args_cli.seed)

    device = torch.device(
        args_cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args_cli.dtype
    ]

    cfg = wm_args()
    for field in (
        "dataset_root_path",
        "dataset_meta_info_path",
        "dataset_names",
        "dataset_cfgs",
        "annotation_name",
    ):
        val = getattr(args_cli, field, None)
        if val is not None:
            setattr(cfg, field, val)
    cfg.action_space = args_cli.action_space
    apply_action_space_config(cfg)
    cfg.ckpt_path = args_cli.ckpt_path
    if args_cli.svd_model_path:
        cfg.pretrained_model_path = args_cli.svd_model_path
    if args_cli.clip_model_path:
        cfg.clip_model_path = args_cli.clip_model_path
    cfg.num_inference_steps = int(args_cli.num_inference_steps)
    cfg.width = int(args_cli.width)              # horiz: doubled width
    cfg.guidance_scale = float(args_cli.max_guidance_scale)

    # ── Checkpoint load ──
    print(f"[eval_horiz] Loading checkpoint: {args_cli.ckpt_path}", flush=True)
    state_dict = torch.load(args_cli.ckpt_path, map_location="cpu", weights_only=False)

    model = build_horiz_model(cfg, state_dict)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    model.vae.eval()
    model.image_encoder.eval()
    model.text_encoder.eval()
    model.pipeline.set_progress_bar_config(disable=True)

    # ── Dataset ──
    val_dataset = Dataset_mix_horiz_with_meta(
        cfg, mode=args_cli.split, his_dropout=False, sparse_history=False,
    )
    n = min(int(args_cli.num_val_samples), len(val_dataset))
    indices = torch.randperm(
        len(val_dataset), generator=torch.Generator().manual_seed(args_cli.seed)
    )[:n].tolist()

    # ── Sharding: pick this worker's slice of (sample_id, dataset_idx) pairs ──
    # sample_id is GLOBAL ([0, n)) so per-sample artifacts (depth_arrays/)
    # keyed by sample_id don't collide across
    # shards. CSV is per-shard.
    shard_idx = int(args_cli.shard_idx)
    num_shards = int(args_cli.num_shards)
    if num_shards < 1 or not (0 <= shard_idx < num_shards):
        raise SystemExit(
            f"invalid --shard_idx={shard_idx} / --num_shards={num_shards}; "
            f"need 0 <= shard_idx < num_shards and num_shards >= 1"
        )
    pairs = list(enumerate(indices))   # (global_sample_id, dataset_idx)
    if num_shards == 1:
        my_pairs = pairs
    elif args_cli.shard_partition == "stripe":
        my_pairs = pairs[shard_idx::num_shards]
    else:  # contiguous
        import math as _math
        per = _math.ceil(n / num_shards)
        start = shard_idx * per
        end = min(start + per, n)
        my_pairs = pairs[start:end]
    print(
        f"[eval_horiz] shard {shard_idx}/{num_shards} ({args_cli.shard_partition}): "
        f"{len(my_pairs)}/{n} samples (dataset size={len(val_dataset)})",
        flush=True,
    )

    import lpips as lpips_lib
    lpips_model = lpips_lib.LPIPS(net="alex").to(device).eval()
    pmh = PointmapHelper()
    ckpt_label = Path(args_cli.ckpt_path).stem
    out_dir = os.path.join(args_cli.output_dir, ckpt_label)
    os.makedirs(out_dir, exist_ok=True)

    rows: list[dict] = []
    for local_i, (sample_id, idx) in enumerate(my_pairs):
        try:
            sample = val_dataset[idx]
            row = _eval_one(
                sample, model, cfg, args_cli, pmh, lpips_model, device, dtype,
                out_dir, sample_id,
            )
            row["sample_id"] = sample_id
            row["dataset_index"] = int(idx)
            row["episode_id"] = int(sample.get("episode_id", -1))
            rows.append(row)
            print(
                f"[eval_horiz][shard {shard_idx}] {local_i+1}/{len(my_pairs)} "
                f"(global_id={sample_id} ep={row['episode_id']}): "
                f"lpips={row.get('lpips_all_mean', float('nan')):.3f} "
                f"abs_rel_vae={row.get('abs_rel_vae_all_mean', float('nan')):.3f} "
                f"pm_l1_vae={row.get('pm_l1_vae_unproj_all_mean', float('nan')):.3f}",
                flush=True,
            )
        except Exception as e:
            import traceback
            print(f"[eval_horiz] skip idx={idx}: {e}", flush=True)
            traceback.print_exc()

    # Per-shard CSV when sharded; canonical name when not.
    csv_name = (
        "per_sample.csv" if num_shards == 1
        else f"per_sample_shard{shard_idx:02d}.csv"
    )
    write_metrics_csv(os.path.join(out_dir, csv_name), rows)
    if rows:
        keys = sorted({k for r in rows for k in r.keys() if k.endswith("_all_mean")})
        summary = {
            k: float(np.mean([r[k] for r in rows if k in r and not np.isnan(r[k])]))
            if any(k in r and not np.isnan(r[k]) for r in rows)
            else float("nan")
            for k in keys
        }
        label = "Aggregate" if num_shards == 1 else f"Shard-{shard_idx} aggregate (partial)"
        print(f"\n[eval_horiz] {label} (mean over samples):", flush=True)
        for k in keys:
            print(f"  {k:<35s}  {summary[k]:.4f}", flush=True)
        import json
        agg_name = (
            "aggregate.json" if num_shards == 1
            else f"aggregate_shard{shard_idx:02d}.json"
        )
        with open(os.path.join(out_dir, agg_name), "w") as f:
            json.dump(summary, f, indent=2)
        if num_shards > 1:
            print(
                f"[eval_horiz] shard done. Run scripts/aggregate_eval_shards.py "
                f"on {out_dir} after all shards complete.",
                flush=True,
            )


@torch.no_grad()
def _eval_one(
    sample: dict,
    model,
    cfg,
    args_cli: argparse.Namespace,
    pmh: PointmapHelper,
    lpips_model,
    device: torch.device,
    dtype: torch.dtype,
    out_dir: str,
    sample_id: int,
) -> dict:
    from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

    num_his = int(cfg.num_history)
    num_frames = int(cfg.num_frames)

    latent = sample["latent"].unsqueeze(0).to(device=device, dtype=dtype)
    action = sample["action"].unsqueeze(0).to(device=device, dtype=dtype)
    text = [sample["text"]]

    his_latent_gt = latent[:, :num_his]
    current_latent = latent[:, num_his]
    gt_lat_full = latent[:, num_his : num_his + num_frames]

    action_latent = model.action_encoder(
        action, text, model.tokenizer, model.text_encoder, cfg.frame_level_cond,
    )

    V_per_view_h = int(cfg.height) // 8
    num_views = current_latent.shape[2] // V_per_view_h
    stacked_image_height = current_latent.shape[2] * 8

    _, pred_lat_full = CtrlWorldDiffusionPipeline.__call__(
        model.pipeline,
        image=current_latent,
        text=action_latent,
        width=int(cfg.width),
        height=stacked_image_height,
        num_frames=num_frames,
        history=his_latent_gt,
        num_inference_steps=int(cfg.num_inference_steps),
        decode_chunk_size=int(getattr(cfg, "decode_chunk_size", 8)),
        min_guidance_scale=float(args_cli.min_guidance_scale),
        max_guidance_scale=float(args_cli.max_guidance_scale),
        fps=int(getattr(cfg, "fps", 5)),
        motion_bucket_id=int(getattr(cfg, "motion_bucket_id", 127)),
        mask=None,
        output_type="latent",
        return_dict=False,
        frame_level_cond=bool(getattr(cfg, "frame_level_cond", True)),
        his_cond_zero=bool(getattr(cfg, "his_cond_zero", False)),
    )

    # ── Decode + width-split: pred + gt-VAE ──
    pred_rgb_u8, pred_dep_m_t, pred_dep_u8 = decode_horiz_latents(model, pred_lat_full)
    gt_rgb_u8_vae, gt_dep_m_t_vae, gt_dep_u8_vae = decode_horiz_latents(model, gt_lat_full)
    # Move depth tensors to GPU for metric ops.
    pred_dep_m = pred_dep_m_t.to(device).float()
    gt_dep_m_vae = gt_dep_m_t_vae.to(device).float()

    rgb_m = compute_rgb_metrics(pred_rgb_u8, gt_rgb_u8_vae, lpips_model, device)
    depth_m_vae = compute_depth_metrics(pred_dep_m, gt_dep_m_vae)
    depth_img_m = compute_depth_psnr_ssim(pred_dep_u8, gt_dep_u8_vae)

    row: dict = {}
    for name, m in {
        "lpips": rgb_m["lpips"],
        "psnr": rgb_m["psnr"],
        "ssim": rgb_m["ssim"],
        "abs_rel_vae": depth_m_vae["abs_rel"],
        "rmse_vae": depth_m_vae["rmse"],
        "rmse_log_vae": depth_m_vae["rmse_log"],
        "d1_vae": depth_m_vae["d1"],
        "depth_psnr_vae": depth_img_m["depth_psnr"],
        "depth_ssim_vae": depth_img_m["depth_ssim"],
    }.items():
        s = _summarize_per_view(m)
        for k, v in s.items():
            row[f"{name}_{k}"] = v

    # ── Pointmap from decoded GT depth latents (primary geometry metric) ──
    episode_id = int(sample.get("episode_id", -1))
    if episode_id >= 0 and pmh.has_traj(episode_id):
        from depth_extras.disparity_to_depth import get_camera_native_intrinsics
        from depth_extras.pointmap import apply_extrinsics, unproject_depth_to_cam_xyz
        from dataset_example.extract_latent_droid_raw import VIEW_ORDER as _VO

        V_total, Fr = pred_dep_m.shape[:2]
        gt_xyz_w = torch.zeros(
            V_total, Fr, 3, *pred_dep_m.shape[-2:],
            dtype=torch.float32, device=pred_dep_m.device,
        )
        pred_xyz_w = torch.zeros_like(gt_xyz_w)
        intr_rec = pmh._intr_by_traj.get(episode_id)
        pm_valid_total = torch.zeros(
            V_total, Fr, *pred_dep_m.shape[-2:],
            dtype=torch.bool, device=pred_dep_m.device,
        )
        for v_idx in range(V_total):
            try:
                native = get_camera_native_intrinsics(intr_rec, _VO[v_idx])
            except (KeyError, ValueError, TypeError):
                continue
            T_wc = pmh.get_world_from_cam(episode_id, v_idx)
            if T_wc is None:
                continue
            T_wc = T_wc.to(pred_dep_m.device)
            gt_cam = unproject_depth_to_cam_xyz(gt_dep_m_vae[v_idx].float(), native)
            gt_xyz_w[v_idx] = apply_extrinsics(gt_cam, T_wc)
            pred_cam = unproject_depth_to_cam_xyz(pred_dep_m[v_idx].float(), native)
            pred_xyz_w[v_idx] = apply_extrinsics(pred_cam, T_wc)
            pm_valid_total[v_idx] = (
                torch.isfinite(gt_dep_m_vae[v_idx])
                & torch.isfinite(pred_dep_m[v_idx])
                & (gt_dep_m_vae[v_idx] > 0)
                & (pred_dep_m[v_idx] > 0)
            )
        pm_m_vae_unproj = compute_pointmap_metrics(
            pred_xyz_w, gt_xyz_w, valid_mask=pm_valid_total,
        )
        for name, m in {
            "pm_l1_vae_unproj": pm_m_vae_unproj["pm_l1"],
            "pm_log_l1_vae_unproj": pm_m_vae_unproj["pm_log_l1"],
        }.items():
            s = _summarize_per_view(m)
            for k, v in s.items():
                row[f"{name}_{k}"] = v

    if args_cli.save_depth_arrays:
        depth_dir = os.path.join(out_dir, "depth_arrays")
        os.makedirs(depth_dir, exist_ok=True)
        np.save(
            os.path.join(depth_dir, f"sample_{sample_id:06d}_pred_depth_m.npy"),
            pred_dep_m.detach().cpu().numpy().astype(np.float32),
        )
        np.save(
            os.path.join(depth_dir, f"sample_{sample_id:06d}_gtvae_depth_m.npy"),
            gt_dep_m_vae.detach().cpu().numpy().astype(np.float32),
        )

    if args_cli.save_videos:
        save_per_sample_videos(
            out_dir=out_dir,
            sample_id=sample_id,
            pred_rgb_vfhwc=pred_rgb_u8,
            gt_rgb_vfhwc=gt_rgb_u8_vae,
            pred_dep_vfh=pred_dep_u8,
            gt_dep_vfh=gt_dep_u8_vae,
            fps=int(getattr(cfg, "fps", 5)),
        )

    return row


def _depth_metric_to_uint8(depth_m: torch.Tensor) -> np.ndarray:
    """(V, F, H, W) metric depth -> (V, F, H, W) uint8 grayscale (encoder side)."""
    from scripts.eval_horiz_utils import depth_to_uint8_for_video
    return depth_to_uint8_for_video(depth_m.float())


if __name__ == "__main__":
    main()
