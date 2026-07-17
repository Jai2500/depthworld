#!/usr/bin/env python
"""Extract per-episode RGB and depth SVD latents from the raw DROID tree.

Reads dataset_meta_info/droid_raw_ctrl/raw_index.jsonl produced by
scripts/build_droid_raw_index.py. For each episode and each of the 3 views
(ext1 -> view 0, ext2 -> view 1, wrist -> view 2):

  - decode recordings/MP4/<serial>.mp4 (1280x720 left view, 15 Hz)
  - load recordings/s2m2_v2/<serial>/stereo_s2m2.npz['disparity_px'] (15 Hz)
  - convert disparity -> metric depth = baseline*fx/disparity
  - log-map depth (per-role range) to gray (close=bright), replicate to 3 ch
  - decimate [::3] to 5 Hz, resize to (192, 320), normalize to [-1, 1]
  - encode through SVD VAE -> save .pt latent

Also writes an annotation/<split>/<traj_id>.json matching the droid_hf schema
so Dataset_mix / Dataset_mix_dual load it unchanged.

Multi-process parallelism is via --num_workers / --worker_idx (the script is
launched once per GPU/process). Each worker handles rows[worker_idx::num_workers]
and skips episodes that already have all outputs on disk.
"""

import argparse
import gc
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import h5py
import mediapy
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import (
    AutoencoderKLTemporalDecoder,
)

# Locked-in encoding choices (see depth_ranges.json + log-vs-linear-vs-inverse analysis).
RANGES_M = {
    "ext1":  (0.30, 3.30),
    "ext2":  (0.30, 3.30),
    "wrist": (0.07, 2.00),
}
# View index ordering in the annotation/latent_videos array. Matches existing
# droid_hf convention: 0 = exterior 1, 1 = exterior 2, 2 = wrist.
VIEW_ORDER = ["ext1", "ext2", "wrist"]

# Spatial / temporal preprocessing (same as droid_hf RGB pipeline).
TARGET_SIZE = (192, 320)
RGB_SKIP = 3                       # 15 Hz -> 5 Hz
VAE_BATCH = 64


def log_depth_to_gray_u8(depth_m: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    """Log map depth_m -> uint8 gray. close=bright (255), far=dark (0)."""
    d = np.clip(depth_m, d_min, d_max)
    log_min = np.log(d_min)
    log_max = np.log(d_max)
    g = (np.log(d) - log_min) / (log_max - log_min)   # [0, 1], close=0
    g = 1.0 - g                                         # close=1, far=0
    return (g * 255.0).astype(np.uint8)


def read_jsonl_rows(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_meta" in obj:
                continue
            rows.append(obj)
    return rows


def trajectory_states(h5_path: str) -> dict:
    """Read the per-frame state/action arrays from trajectory.h5 (15 Hz)."""
    with h5py.File(h5_path, "r") as f:
        # Note: observation uses 'joint_positions' (plural), action uses 'joint_position' (singular).
        out = {
            "observation.state.cartesian_position": f["observation/robot_state/cartesian_position"][...].tolist(),
            "observation.state.joint_position":     f["observation/robot_state/joint_positions"][...].tolist(),
            "observation.state.gripper_position":   f["observation/robot_state/gripper_position"][...].tolist(),
            "action.cartesian_position":            f["action/cartesian_position"][...].tolist(),
            "action.joint_position":                f["action/joint_position"][...].tolist(),
            "action.gripper_position":              f["action/gripper_position"][...].tolist(),
            "action.joint_velocity":                f["action/joint_velocity"][...].tolist(),
        }
    return out


def frames_uint8_to_latent(
    frames_u8: np.ndarray, vae: AutoencoderKLTemporalDecoder, device: torch.device,
) -> torch.Tensor:
    """frames_u8: (T, H, W, 3) uint8 -> (T, 4, H/8, W/8) fp32 latent on CPU."""
    t = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float()        # (T, 3, H, W)
    t = t / 255.0 * 2.0 - 1.0
    t = F.interpolate(t, size=TARGET_SIZE, mode="bilinear", align_corners=False)
    latents = []
    with torch.no_grad():
        for i in range(0, len(t), VAE_BATCH):
            chunk = t[i:i + VAE_BATCH].to(device, dtype=vae.dtype)
            z = vae.encode(chunk).latent_dist.sample()
            z = z * vae.config.scaling_factor
            latents.append(z.detach().to(torch.float32).cpu())
    return torch.cat(latents, dim=0)


def write_qa_video(frames_u8_5hz: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mediapy.write_video(path, frames_u8_5hz, fps=5)


def outputs_already_complete(out_root: Path, split: str, traj_id: int) -> bool:
    ann = out_root / "annotation" / split / f"{traj_id}.json"
    if not ann.is_file():
        return False
    for v in range(3):
        for sub in ("latent_videos", "latent_videos_depth"):
            if not (out_root / sub / split / str(traj_id) / f"{v}.pt").is_file():
                return False
    return True


def process_episode(
    row: dict,
    out_root: Path,
    vae: AutoencoderKLTemporalDecoder,
    device: torch.device,
    write_qa_videos: bool,
) -> tuple[bool, str]:
    traj_id = row["traj_id"]
    split = row["split"]
    uuid = row["episode_uuid"]
    raw_dir = row["raw_dir"]

    if outputs_already_complete(out_root, split, traj_id):
        return True, "skip-complete"

    # 1. Load h5 states (sets the canonical 15 Hz length).
    h5_path = os.path.join(raw_dir, "trajectory.h5")
    if not os.path.isfile(h5_path):
        return False, f"no_h5:{h5_path}"
    states = trajectory_states(h5_path)
    T_traj = len(states["observation.state.cartesian_position"])

    # 2. For each view: load RGB + depth, align, decimate, encode. Latents are
    #    tiny (~860 KB each) so we hold them in memory across views and truncate
    #    all 3 to a common T_5hz at the end (handles 1-frame off-by-ones between
    #    cameras within the same episode).
    rgb_latents: list[torch.Tensor] = []
    depth_latents: list[torch.Tensor] = []
    rgb_previews: list[np.ndarray] = []      # only filled when write_qa_videos
    depth_previews: list[np.ndarray] = []

    for v_idx, role in enumerate(VIEW_ORDER):
        cam = row["cameras"][role]

        rgb_full = np.asarray(mediapy.read_video(cam["mp4_path"]))  # (T_mp4, 720, 1280, 3) uint8
        T_mp4 = rgb_full.shape[0]

        with np.load(cam["npz_path"]) as z:
            disp = z["disparity_px"][...]                # (T_disp, 720, 1280) float16
        T_disp = disp.shape[0]

        N = min(T_mp4, T_disp, T_traj)
        if N < RGB_SKIP * 2:
            return False, f"too_short:{uuid}:N={N}"

        rgb_full = rgb_full[:N]
        disp = disp[:N]

        # Disparity -> metric depth -> log-gray -> 3-channel uint8. Keep disp
        # in float16 until we divide; release intermediate buffers ASAP.
        depth = (cam["baseline"] * cam["fx"]) / np.maximum(disp.astype(np.float32), 1e-3)
        del disp
        d_min, d_max = RANGES_M[role]
        gray_u8 = log_depth_to_gray_u8(depth, d_min, d_max)     # (N, 720, 1280) uint8
        del depth
        depth_rgb = np.repeat(gray_u8[..., None], 3, axis=-1)   # (N, 720, 1280, 3) uint8
        del gray_u8

        # Decimate 15 Hz -> 5 Hz. .copy() detaches from the larger backing array
        # so the (N, 720, 1280, ·) buffers can be freed below.
        rgb_5hz = rgb_full[::RGB_SKIP].copy()
        depth_5hz = depth_rgb[::RGB_SKIP].copy()
        del rgb_full, depth_rgb

        rgb_latents.append(frames_uint8_to_latent(rgb_5hz, vae, device))
        depth_latents.append(frames_uint8_to_latent(depth_5hz, vae, device))
        if write_qa_videos:
            rgb_previews.append(rgb_5hz)
            depth_previews.append(depth_5hz)
        else:
            del rgb_5hz, depth_5hz

        gc.collect()

    # 3. Truncate all views to common T_5hz, then save.
    T_5hz = int(min(lat.shape[0] for lat in rgb_latents))
    video_length = T_5hz

    for v_idx in range(3):
        rgb_lat = rgb_latents[v_idx][:T_5hz]
        depth_lat = depth_latents[v_idx][:T_5hz]
        rgb_out = out_root / "latent_videos" / split / str(traj_id) / f"{v_idx}.pt"
        dep_out = out_root / "latent_videos_depth" / split / str(traj_id) / f"{v_idx}.pt"
        rgb_out.parent.mkdir(parents=True, exist_ok=True)
        dep_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rgb_lat, rgb_out)
        torch.save(depth_lat, dep_out)
        if write_qa_videos:
            write_qa_video(
                rgb_previews[v_idx][:T_5hz],
                str(out_root / "videos" / split / str(traj_id) / f"{v_idx}.mp4"),
            )
            write_qa_video(
                depth_previews[v_idx][:T_5hz],
                str(out_root / "videos_depth" / split / str(traj_id) / f"{v_idx}.mp4"),
            )

    # Drop big buffers before annotation/next-episode.
    del rgb_latents, depth_latents, rgb_previews, depth_previews
    gc.collect()

    # 3. Pick text. Prefer aggregated annotations, fall back to current_task, then "".
    if row.get("language_instructions"):
        texts = list(row["language_instructions"])
    elif row.get("current_task"):
        texts = [row["current_task"]]
    else:
        texts = [""]

    # 4. Build annotation. Match droid_hf schema; add `source` for traceability.
    ann = {
        "texts": texts,
        "episode_id": traj_id,
        "success": int(bool(row.get("success", False))),
        "video_length": video_length,
        "state_length": video_length,  # legacy; existing code only reads observation.state.*
        "raw_length": T_traj,
        "videos": [
            {"video_path": f"videos/{split}/{traj_id}/{i}.mp4"} for i in range(3)
        ],
        "latent_videos": [
            {"latent_video_path": f"latent_videos/{split}/{traj_id}/{i}.pt"} for i in range(3)
        ],
        "states": [
            (states["observation.state.cartesian_position"][i] + [states["observation.state.gripper_position"][i]])
            for i in range(0, T_traj, RGB_SKIP)
        ][:video_length],
        "observation.state.cartesian_position": states["observation.state.cartesian_position"],
        "observation.state.joint_position":     states["observation.state.joint_position"],
        "observation.state.gripper_position":   states["observation.state.gripper_position"],
        "action.cartesian_position":            states["action.cartesian_position"],
        "action.joint_position":                states["action.joint_position"],
        "action.gripper_position":              states["action.gripper_position"],
        "action.joint_velocity":                states["action.joint_velocity"],
        "source": {
            "episode_uuid": uuid,
            "raw_dir": raw_dir,
            "lab": row.get("lab"),
            "role_to_serial": {r: row["cameras"][r]["serial"] for r in VIEW_ORDER},
            "baselines_m": {r: row["cameras"][r]["baseline"] for r in VIEW_ORDER},
            "fx_px": {r: row["cameras"][r]["fx"] for r in VIEW_ORDER},
            "depth_ranges_m": RANGES_M,
            "depth_encoding": "log_close_bright",
        },
    }
    ann_path = out_root / "annotation" / split / f"{traj_id}.json"
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(ann_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ann, f, indent=2)
    os.replace(tmp, ann_path)

    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", type=str,
                    default="dataset_meta_info/droid_raw_ctrl/raw_index.jsonl")
    ap.add_argument("--output_root", type=str, default="data/droid_raw_ctrl")
    ap.add_argument("--svd_path", type=str,
                    default="checkpoints/stable-video-diffusion-img2vid")
    ap.add_argument("--worker_idx", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--limit_to_date", type=str, default="",
                    help="if set (e.g. 2023-10-21), only process raw_dirs containing this substring")
    ap.add_argument("--max_episodes", type=int, default=0, help="0 = no limit")
    ap.add_argument("--write_qa_videos", action="store_true",
                    help="also save 5 Hz MP4s under videos/ and videos_depth/ for visual QA")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    t0 = time.time()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Loading raw_index from {args.index_path}", flush=True)
    rows = read_jsonl_rows(args.index_path)
    print(f"[{time.strftime('%H:%M:%S')}]   {len(rows)} indexed episodes", flush=True)

    if args.limit_to_date:
        rows = [r for r in rows if args.limit_to_date in r["raw_dir"]]
        print(f"[{time.strftime('%H:%M:%S')}]   {len(rows)} after limit_to_date={args.limit_to_date!r}", flush=True)

    # Worker shard: each process takes rows[worker_idx::num_workers].
    rows = rows[args.worker_idx::args.num_workers]
    if args.max_episodes > 0:
        rows = rows[:args.max_episodes]
    print(f"[{time.strftime('%H:%M:%S')}] Worker {args.worker_idx}/{args.num_workers}: {len(rows)} episodes to process", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[{time.strftime('%H:%M:%S')}] Loading SVD VAE from {args.svd_path}", flush=True)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_path, subfolder="vae")
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    print(f"[{time.strftime('%H:%M:%S')}]   vae on {device}, dtype={vae.dtype}", flush=True)

    def _on_term(signum, frame):
        print(f"[{time.strftime('%H:%M:%S')}] caught signal {signum}; exiting", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    signal.signal(signal.SIGUSR1, _on_term)

    n_ok = 0
    n_skip = 0
    n_err = 0
    t1 = time.time()
    for i, row in enumerate(rows):
        try:
            ok, reason = process_episode(row, out_root, vae, device, args.write_qa_videos)
        except Exception as e:
            ok = False
            reason = f"exception:{type(e).__name__}:{e}"
            traceback.print_exc()

        if ok and reason == "skip-complete":
            n_skip += 1
        elif ok:
            n_ok += 1
        else:
            n_err += 1
            if n_err <= 20:
                print(f"  [err] traj_id={row.get('traj_id')} uuid={row.get('episode_uuid')} : {reason}", flush=True)

        if (i + 1) % 10 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - t1
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (len(rows) - (i + 1)) / max(rate, 1e-3)
            print(
                f"[{time.strftime('%H:%M:%S')}]  {i + 1}/{len(rows)} eps "
                f"(ok={n_ok}, skip={n_skip}, err={n_err}, "
                f"{rate:.2f} ep/s, {elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)",
                flush=True,
            )

    print(
        f"\n[{time.strftime('%H:%M:%S')}] Worker {args.worker_idx} done: "
        f"ok={n_ok}, skip={n_skip}, err={n_err}, total wall={time.time()-t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
