"""Utilities for recovering metric depth (and 3D point clouds) from the
downsampled disparity stored under `disparity_lo/<split>/<traj_id>/<view>.pt`.

Two ways to obtain camera intrinsics:

    (a) The per-episode annotation `source` field (always available; lightweight):
            ann = json.load(open(".../annotation/train/123.json"))
            baseline = ann["source"]["baselines_m"]["ext1"]
            fx       = ann["source"]["fx_px"]["ext1"]
        — enough for `depth = baseline * fx / disparity`, but missing fy/cx/cy.

    (b) `dataset_meta_info/droid_raw_ctrl/camera_intrinsics.jsonl`
        (one row per episode, produced by scripts/build_camera_intrinsics.py):
        full pinhole intrinsics + distortion + stereo extrinsics. Use when you
        want to unproject depth to 3D points or do anything geometric.

The disparity values are in NATIVE-resolution pixel units (1280 px wide) even
though the array is stored at (192, 320), because we used z-buffer max-pool
(no value rescaling). Therefore the conversion uses the NATIVE fx/baseline as
stored — never scale them just because the spatial grid is smaller. Scaling
intrinsics only matters for 3D unprojection at low-res grid coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import torch


VIEW_ORDER = ["ext1", "ext2", "wrist"]


# ---------------------------------------------------------------------------
# Core conversion (works for any tensor shape and any spatial resolution)
# ---------------------------------------------------------------------------

def disparity_to_depth(
    disp: torch.Tensor,
    baseline_m: float,
    fx_native_px: float,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Convert stereo disparity (in native-image pixels) to metric depth.

    Works for any shape. Pre-cast to float32 to keep precision sane when input
    is fp16; clamp at `eps` to avoid divide-by-zero for sky / no-return pixels.

    Args:
        disp:           any tensor whose values are disparity in native pixels
                        (fp16 is fine; will be promoted to fp32 for the divide).
        baseline_m:     stereo baseline in meters (per-camera calibration).
        fx_native_px:   horizontal focal length in pixels of the native
                        sensor image (1280×720). Same value as stored in
                        annotation `source.fx_px[role]`.
        eps:            floor for disparity before division.

    Returns:
        Depth tensor with the same shape as disp, in meters, float32.
    """
    disp32 = disp.to(torch.float32).clamp(min=eps)
    return baseline_m * fx_native_px / disp32


# ---------------------------------------------------------------------------
# Loaders: full intrinsics from the camera_intrinsics.jsonl
# ---------------------------------------------------------------------------

def load_intrinsics(
    intrinsics_jsonl_path: str | Path,
) -> dict[int, dict]:
    """Load `camera_intrinsics.jsonl` into a dict keyed by traj_id."""
    intrinsics_jsonl_path = Path(intrinsics_jsonl_path)
    out: dict[int, dict] = {}
    with open(intrinsics_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["traj_id"])] = rec
    return out


def get_camera_native_intrinsics(intr_rec: dict, role: str) -> dict:
    """Pull the native-resolution `left`-camera intrinsics (and baseline) for
    one role out of the per-episode intrinsics record."""
    cam = intr_rec["cameras"][role]
    return {
        "fx": float(cam["left"]["fx"]),
        "fy": float(cam["left"]["fy"]),
        "cx": float(cam["left"]["cx"]),
        "cy": float(cam["left"]["cy"]),
        "width": int(cam["left"]["width"]),     # native, typically 1280
        "height": int(cam["left"]["height"]),    # native, typically 720
        "baseline_m": float(cam["stereo_baseline_m"]),
        "distortion": list(cam["left"].get("disto", [])),
        "serial": cam.get("serial", ""),
    }


def load_disparity_lo(
    dataset_root: str | Path,
    split: str,
    traj_id: int,
    view_idx: int,
) -> torch.Tensor:
    """Read the saved (T, 192, 320) fp16 disparity tensor for one (episode, view)."""
    path = Path(dataset_root) / "disparity_lo" / split / str(traj_id) / f"{view_idx}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def load_depth_lo(
    dataset_root: str | Path,
    intrinsics_by_traj: dict[int, dict],
    split: str,
    traj_id: int,
    view_idx: int,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Read disparity_lo and return metric depth at (T, 192, 320) float32 meters."""
    role = VIEW_ORDER[view_idx]
    intr = get_camera_native_intrinsics(intrinsics_by_traj[traj_id], role)
    disp = load_disparity_lo(dataset_root, split, traj_id, view_idx)
    return disparity_to_depth(disp, intr["baseline_m"], intr["fx"], eps=eps)


# ---------------------------------------------------------------------------
# 3D unprojection (needs full intrinsics)
# ---------------------------------------------------------------------------

def unproject_depth_to_xyz(
    depth: torch.Tensor,
    native_intr: dict,
) -> torch.Tensor:
    """Unproject a (T, Ho, Wo) depth map to (T, 3, Ho, Wo) of (X, Y, Z) in the
    camera frame.

    `native_intr` is the dict returned by `get_camera_native_intrinsics`. The
    native intrinsics are scaled to the depth map's actual resolution (Ho, Wo)
    via separate scale factors in x (Wo / native_width) and y (Ho / native_height).
    These can differ (our pipeline uses 192/720 ≠ 320/1280, a slight vertical
    stretch).

    Returns (T, 3, Ho, Wo) float32 in meters.
    """
    assert depth.ndim == 3, "expected (T, Ho, Wo)"
    T, Ho, Wo = depth.shape
    sx = Wo / native_intr["width"]
    sy = Ho / native_intr["height"]
    fx_lo = native_intr["fx"] * sx
    fy_lo = native_intr["fy"] * sy
    cx_lo = native_intr["cx"] * sx
    cy_lo = native_intr["cy"] * sy

    v_grid, u_grid = torch.meshgrid(
        torch.arange(Ho, dtype=torch.float32),
        torch.arange(Wo, dtype=torch.float32),
        indexing="ij",
    )
    depth_f = depth.to(torch.float32)
    X = (u_grid - cx_lo) * depth_f / fx_lo                 # broadcast over T
    Y = (v_grid - cy_lo) * depth_f / fy_lo
    Z = depth_f
    return torch.stack([X, Y, Z], dim=1)                   # (T, 3, Ho, Wo)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _percentile(t: torch.Tensor, pct: float) -> float:
    return float(torch.quantile(t.flatten(), pct / 100.0))


_PKG_META = Path(__file__).resolve().parent / "meta"


def smoke_test(
    dataset_root: str | Path = "dataset_example/droid_raw_ctrl",
    intrinsics_jsonl: str | Path = _PKG_META / "camera_intrinsics.jsonl",
    test_traj_ids: Iterable[int] | None = None,
) -> None:
    """End-to-end check: load a few episodes' disparity_lo, convert to depth,
    cross-check against the annotation source field, unproject to 3D, sanity-check
    every step."""
    dataset_root = Path(dataset_root)
    intrinsics_jsonl = Path(intrinsics_jsonl)

    print(f"Loading intrinsics from {intrinsics_jsonl} ...")
    intr_by_traj = load_intrinsics(intrinsics_jsonl)
    print(f"  {len(intr_by_traj)} episodes")

    # Pick some test traj_ids: a few from train (every ~25k of the dataset).
    if test_traj_ids is None:
        all_train_ids = sorted(int(p.stem) for p in
                               (dataset_root / "annotation" / "train").glob("*.json"))
        n = len(all_train_ids)
        test_traj_ids = [all_train_ids[i] for i in (0, n // 4, n // 2, 3 * n // 4, n - 1)]

    for traj_id in test_traj_ids:
        print(f"\n=== traj_id={traj_id} ===")
        if traj_id not in intr_by_traj:
            print("  no intrinsics record; skipping")
            continue
        ann_path = dataset_root / "annotation" / "train" / f"{traj_id}.json"
        if not ann_path.is_file():
            ann_path = dataset_root / "annotation" / "val" / f"{traj_id}.json"
            split = "val"
        else:
            split = "train"
        with open(ann_path) as f:
            ann = json.load(f)
        print(f"  split={split}  episode_uuid={ann['source']['episode_uuid']}")

        for v_idx, role in enumerate(VIEW_ORDER):
            disp_path = dataset_root / "disparity_lo" / split / str(traj_id) / f"{v_idx}.pt"
            if not disp_path.is_file():
                print(f"  {role}: disparity_lo missing, skipping")
                continue
            disp = load_disparity_lo(dataset_root, split, traj_id, v_idx)
            T, Ho, Wo = disp.shape

            # Cross-check intrinsics: annotation vs intrinsics jsonl.
            ann_fx = float(ann["source"]["fx_px"][role])
            ann_baseline = float(ann["source"]["baselines_m"][role])
            intr_native = get_camera_native_intrinsics(intr_by_traj[traj_id], role)
            agree = (
                abs(intr_native["fx"] - ann_fx) < 1e-3
                and abs(intr_native["baseline_m"] - ann_baseline) < 1e-6
            )
            print(
                f"  {role}: shape=(T={T},{Ho},{Wo})  "
                f"fx={intr_native['fx']:.3f}  baseline={intr_native['baseline_m']:.5f}  "
                f"(annotation==intrinsics_jsonl: {agree})"
            )

            # Convert to depth.
            depth = disparity_to_depth(disp, intr_native["baseline_m"], intr_native["fx"])
            print(
                f"      depth stats:  min={depth.min():.3f}m  "
                f"p1={_percentile(depth, 1):.3f}m  "
                f"p50={_percentile(depth, 50):.3f}m  "
                f"p99={_percentile(depth, 99):.3f}m  "
                f"max={depth.max():.3f}m"
            )

            # 3D unproject: verify Z component equals depth (by construction).
            xyz = unproject_depth_to_xyz(depth, intr_native)
            z_max_err = (xyz[:, 2] - depth).abs().max().item()
            print(
                f"      xyz: X stats [{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]m  "
                f"Y [{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]m  "
                f"Z [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]m  "
                f"Z-vs-depth max-abs-err={z_max_err:.2e}"
            )
            assert z_max_err < 1e-5, "Z component of unprojected XYZ should equal depth"


if __name__ == "__main__":
    smoke_test()
