#!/usr/bin/env python
"""Export raw downsampled stereo disparity (disparity_lo).

Reads dataset_meta_info/droid_raw_ctrl/raw_index.jsonl. For each episode and
each of the 3 views:

  - skip MP4 decode entirely (RGB/depth latents come from
    extract_latent_droid_raw.py and are unchanged)
  - read recordings/s2m2_v2/<serial>/stereo_s2m2.npz['disparity_px']
  - temporally decimate disparity [::3] (matches the 15Hz -> 5Hz convention)
  - spatially downsample disparity via Z-BUFFER MAX-POOL (scatter_reduce amax):
      mathematically equivalent to "unproject every high-res pixel to a 3D
      point, project into the low-res camera with scaled K, z-buffer" for our
      same-camera-different-resolution case.
  - save raw downsampled disparity (fp16, UNCLIPPED) to disparity_lo/<split>/<id>/<view>.pt

disparity_lo is the raw downsampled stereo signal, used to build world-frame
pointmap ground truth for pointmap training and depth/pointmap eval metrics.
Metric depth is recoverable as baseline*fx/disparity using fields stored in
annotation/.../source. It is saved unclipped at full numerical precision
(subject to fp16) so downstream consumers keep flexibility on what range to
clip / how to map.
"""

import argparse
import gc
import json
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

VIEW_ORDER = ["ext1", "ext2", "wrist"]

TARGET_SIZE = (192, 320)   # (Ho, Wo)
RGB_SKIP = 3               # 15 Hz -> 5 Hz

# ------------------------------------------------------------------
# Z-buffer max-pool primitives
# ------------------------------------------------------------------

# Per-worker cache of pixel-grid scatter indices. Same shape conversion for
# every episode, so we compute the (H*W -> Ho*Wo) index map once.
_SCATTER_IDX_CACHE: "dict[tuple, torch.Tensor]" = {}


def get_scatter_idx(H: int, W: int, Ho: int, Wo: int, device: torch.device) -> torch.Tensor:
    """Pixel-grid scaling: each source pixel's flattened index in the
    (Ho, Wo) output grid. Source (v, u) -> output (floor(v*Ho/H), floor(u*Wo/W))."""
    key = (H, W, Ho, Wo, str(device))
    cached = _SCATTER_IDX_CACHE.get(key)
    if cached is not None:
        return cached
    yh = torch.arange(H, device=device)
    xh = torch.arange(W, device=device)
    y_lo = ((yh * Ho) // H).clamp(0, Ho - 1)                                       # (H,)
    x_lo = ((xh * Wo) // W).clamp(0, Wo - 1)                                       # (W,)
    yy, xx = torch.meshgrid(y_lo, x_lo, indexing="ij")
    flat_idx = (yy * Wo + xx).reshape(-1).long()                                   # (H*W,)
    _SCATTER_IDX_CACHE[key] = flat_idx
    return flat_idx


def zbuffer_max_pool_disparity(disp: torch.Tensor, out_hw: "tuple[int, int]") -> torch.Tensor:
    """Z-buffer max-pool: (T, H, W) -> (T, Ho, Wo). At each output pixel keep
    the maximum disparity (= minimum depth, = closest surface) among all
    source pixels that fall in that bin under integer-floor scaling.

    For our same-camera setup this is *exactly* the 3D unproject-reproject
    pipeline (with the algebra collapsing to a 2D index map). Disparities are
    non-negative so initialising the output at 0 is safe."""
    T, H, W = disp.shape
    Ho, Wo = out_hw
    flat_idx = get_scatter_idx(H, W, Ho, Wo, disp.device)
    idx_expand = flat_idx.unsqueeze(0).expand(T, -1)                               # (T, H*W)
    disp_flat = disp.reshape(T, -1)                                                # (T, H*W)
    out = torch.zeros((T, Ho * Wo), device=disp.device, dtype=disp.dtype)
    out.scatter_reduce_(1, idx_expand, disp_flat, reduce="amax", include_self=True)
    return out.reshape(T, Ho, Wo)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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


def outputs_complete(out_root: Path, disp_subdir: str, split: str, traj_id: int) -> bool:
    for v in range(3):
        if not (out_root / disp_subdir / split / str(traj_id) / f"{v}.pt").is_file():
            return False
    return True


# ------------------------------------------------------------------
# Per-episode work
# ------------------------------------------------------------------

def process_episode(
    row: dict,
    out_root: Path,
    disp_subdir: str,
    device: torch.device,
) -> "tuple[bool, str]":
    traj_id = row["traj_id"]
    split = row["split"]

    if outputs_complete(out_root, disp_subdir, split, traj_id):
        return True, "skip-complete"

    # Gate on the latent extractor's annotation (atomic-renamed; its presence
    # implies the RGB latents we anchor on are fully committed).
    ann_path = out_root / "annotation" / split / f"{traj_id}.json"
    if not ann_path.is_file():
        return False, "latents_not_ready"

    # Anchor T to the RGB latent length so disparity is pixel-aligned.
    rgb_path = out_root / "latent_videos" / split / str(traj_id) / "0.pt"
    if not rgb_path.is_file():
        return False, f"no_rgb_latent:{rgb_path}"
    try:
        rgb_lat = torch.load(rgb_path, map_location="cpu", weights_only=True)
    except Exception as e:
        return False, f"bad_rgb_latent:{repr(e)}"
    T_target = int(rgb_lat.shape[0])
    del rgb_lat
    if T_target < 2:
        return False, f"rgb_latent_too_short:T_target={T_target}"

    for v_idx, role in enumerate(VIEW_ORDER):
        cam = row["cameras"][role]

        # 1. Load disparity at full resolution (fp16).
        try:
            with np.load(cam["npz_path"]) as z:
                disp_hi_np = z["disparity_px"][...]
            if disp_hi_np.ndim != 3:
                return False, f"bad_disp_ndim:{role}:{disp_hi_np.ndim}"
        except Exception as e:
            return False, f"disp_load_failed:{role}:{repr(e)}"

        # 2. Temporal decimation [::3] to 5 Hz, BEFORE moving to GPU so we
        #    don't waste VRAM holding the full 15 Hz tensor.
        disp_5hz_np = disp_hi_np[::RGB_SKIP]                                       # (T5, H, W) fp16
        del disp_hi_np

        # 3. Move to device as fp32 for the scatter_reduce.
        disp_5hz = torch.from_numpy(np.ascontiguousarray(disp_5hz_np)).to(
            device=device, dtype=torch.float32, non_blocking=False
        )
        del disp_5hz_np

        # 4. Spatial downsample via z-buffer max-on-disparity.
        disp_lo = zbuffer_max_pool_disparity(disp_5hz, TARGET_SIZE)                # (T5, Ho, Wo)
        del disp_5hz

        # 5. Truncate to the RGB latent length (handles ±1 frame off-by-ones).
        if disp_lo.shape[0] < T_target:
            return False, f"too_short_after_decimate:{role}:got={disp_lo.shape[0]}:target={T_target}"
        disp_lo = disp_lo[:T_target]                                               # (T_target, Ho, Wo)

        # 6. Save raw downsampled disparity (fp16, UNCLIPPED, on disk in pt).
        disp_path = out_root / disp_subdir / split / str(traj_id) / f"{v_idx}.pt"
        disp_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(disp_lo.to(torch.float16).cpu(), disp_path)
        del disp_lo

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return True, "ok"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--index_path", type=str,
        default="dataset_meta_info/droid_raw_ctrl/raw_index.jsonl",
    )
    ap.add_argument(
        "--output_root", type=str, default="data/droid_raw_ctrl",
    )
    ap.add_argument(
        "--disp_subdir", type=str, default="disparity_lo",
        help="subdir under output_root for raw downsampled disparity (fp16, no clip)",
    )
    ap.add_argument("--worker_idx", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--max_episodes", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    t0 = time.time()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Loading raw_index from {args.index_path}", flush=True)
    rows = read_jsonl_rows(args.index_path)
    print(f"[{time.strftime('%H:%M:%S')}]   {len(rows)} indexed episodes", flush=True)

    rows = rows[args.worker_idx::args.num_workers]
    if args.max_episodes > 0:
        rows = rows[:args.max_episodes]
    print(
        f"[{time.strftime('%H:%M:%S')}] Worker {args.worker_idx}/{args.num_workers}: "
        f"{len(rows)} episodes",
        flush=True,
    )
    print(f"[{time.strftime('%H:%M:%S')}]   disp_subdir = {args.disp_subdir!r}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _on_term(signum, frame):
        print(f"[{time.strftime('%H:%M:%S')}] caught signal {signum}; exiting", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    signal.signal(signal.SIGUSR1, _on_term)

    n_ok = 0
    n_skip = 0
    n_wait = 0
    n_err = 0
    t1 = time.time()
    for i, row in enumerate(rows):
        try:
            ok, reason = process_episode(row, out_root, args.disp_subdir, device)
        except Exception as e:
            ok = False
            reason = f"exception:{type(e).__name__}:{e}"
            traceback.print_exc()

        if ok and reason == "skip-complete":
            n_skip += 1
        elif ok:
            n_ok += 1
        elif reason == "latents_not_ready":
            n_wait += 1
        else:
            n_err += 1
            if n_err <= 20:
                print(
                    f"  [err] traj_id={row.get('traj_id')} uuid={row.get('episode_uuid')} : {reason}",
                    flush=True,
                )

        if (i + 1) % 10 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - t1
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (len(rows) - (i + 1)) / max(rate, 1e-3)
            print(
                f"[{time.strftime('%H:%M:%S')}]  {i + 1}/{len(rows)} eps "
                f"(ok={n_ok}, skip={n_skip}, wait_latents={n_wait}, err={n_err}, "
                f"{rate:.2f} ep/s, {elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)",
                flush=True,
            )

    print(
        f"\n[{time.strftime('%H:%M:%S')}] Worker {args.worker_idx} done: "
        f"ok={n_ok}, skip={n_skip}, wait_latents={n_wait}, err={n_err}, "
        f"total wall={time.time()-t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
