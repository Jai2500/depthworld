# Horiz + world-frame GT point maps, for CrtlWorldPointmap (+ DPTPointMapDecoder).
#
# Extends `Dataset_mix_horiz` with per-view world-frame GT point maps built
# on the fly from raw disparity + the DROID-3D calibration:
#
#   pointmap_gt    : (V, T, 3, Hpm, Wpm)  fp32   world-frame XYZ in metres
#   pointmap_valid : (V, T, Hpm, Wpm)     fp32   1.0 where calibration + depth
#                                                were available; 0.0 elsewhere.
#                                                Per-pixel (post-finite check).
#
# Pipeline (per view, per episode), via depth_extras/pointmap.construct_pointmap:
#   disparity_lo (loaded from <dataset_dir>/disparity_lo/<split>/<traj>/<view>.pt)
#     -> depth = baseline * fx_native / disparity        (depth_extras/disparity_to_depth)
#     -> clamp to the depth-latent metric range
#        (0.30-3.30 m ext, 0.07-2.00 m wrist)
#     -> camera-frame XYZ via per-pixel ray unprojection (depth_extras/pointmap)
#     -> world-frame XYZ via apply_extrinsics            (ext1/ext2: const-per-episode;
#                                                          wrist: per-frame from FK)
#   Then sliced to the clip's `rgb_id` frame indices and stacked across views.
#
# Calibration sources (see depth_extras/meta/README.md for download):
#   - depth_extras/meta/camera_intrinsics.jsonl : per-episode pinhole intrinsics
#   - depth_extras/meta/extrinsics.jsonl        : per-episode JFG extrinsics
#   - depth_extras/assets/gripper2wrist_transforms.json : per-robot hand-eye
#
# Caveats:
#   * Disparity_lo lives at 192x320 (raw image resolution), not 24x40 (latent).
#     Each (traj, view) file is ~24 MB at fp16 — substantially larger than
#     the (24x40) latent files. We load the full trajectory's disparity once
#     per __getitem__ and slice — simple but I/O-heavy. Revisit if dataloader
#     becomes the bottleneck.
#   * Pixels outside the requested metric range are clamped for XYZ stability
#     but excluded from `pointmap_valid`, so the model is not trained on fake
#     near/far range-boundary surfaces.
#   * Missing JFG extrinsics (~2.9%) or missing gripper2wrist (~5-10% wrist
#     only) produces zero point maps + valid=0 for that view/frame. The loss
#     should skip these via the valid mask.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dataset.dataset_droid import Dataset_mix_horiz
from depth_extras.disparity_to_depth import load_intrinsics
from depth_extras.extrinsics import (
    DEFAULT_GRIPPER2WRIST_ASSET,
    load_extrinsics,
    load_gripper2wrist_transforms,
)
from depth_extras.pointmap import construct_pointmap

# Default search locations relative to repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CAMERA_INTR = _REPO_ROOT / "depth_extras" / "meta" / "camera_intrinsics.jsonl"
_DEFAULT_EXTRINSICS = _REPO_ROOT / "depth_extras" / "meta" / "extrinsics.jsonl"

# Per-view depth range (metres). Matches the depth-latent encoding used by
# the released checkpoints.
_DEFAULT_RANGE_PER_ROLE_M = {0: (0.30, 3.30), 1: (0.30, 3.30), 2: (0.07, 2.00)}


class Dataset_mix_horiz_with_pointmap(Dataset_mix_horiz):
    """Horiz layout + world-frame GT point maps + per-pixel valid mask."""

    def __init__(
        self,
        *args,
        camera_intrinsics_path: Optional[str] = None,
        camera_extrinsics_path: Optional[str] = None,
        gripper2wrist_path: Optional[str] = None,
        lat_height: int = 24,
        lat_width: int = 40,
        pointmap_depth_range_per_view: Optional[dict[int, tuple[float, float]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        intr_path = Path(camera_intrinsics_path or _DEFAULT_CAMERA_INTR)
        extr_path = Path(camera_extrinsics_path or _DEFAULT_EXTRINSICS)
        g2w_path = Path(gripper2wrist_path or DEFAULT_GRIPPER2WRIST_ASSET)

        if not intr_path.is_file():
            raise FileNotFoundError(f"camera_intrinsics.jsonl not found at {intr_path}")
        if not extr_path.is_file():
            raise FileNotFoundError(f"extrinsics.jsonl not found at {extr_path}")
        if not g2w_path.is_file():
            raise FileNotFoundError(f"gripper2wrist asset not found at {g2w_path}")

        # Load all three metadata sources once. These are small enough to
        # live entirely in memory (a few hundred MB).
        self._intrinsics_by_traj = load_intrinsics(intr_path)
        self._extrinsics_by_traj = load_extrinsics(extr_path)
        self._gripper2wrist = load_gripper2wrist_transforms(g2w_path)

        self._lat_h = int(lat_height)
        self._lat_w = int(lat_width)
        self._pm_depth_range = (
            pointmap_depth_range_per_view
            if pointmap_depth_range_per_view is not None
            else dict(_DEFAULT_RANGE_PER_ROLE_M)
        )
        # Filled by the parent's __getitem__ via _stash_sample().
        self._last_sample: Optional[tuple[int, list[int], int, str, str]] = None

    # ------------------------------------------------------------------
    # Parent calls this at the end of __getitem__ (via _maybe_stash_sample).
    # ------------------------------------------------------------------
    def _stash_sample(
        self,
        traj_id: int,
        rgb_id: list[int],
        num_views: int,
        split: str,
        dataset_dir: str,
    ) -> None:
        self._last_sample = (
            int(traj_id),
            [int(f) for f in rgb_id],
            int(num_views),
            str(split),
            str(dataset_dir),
        )

    # ------------------------------------------------------------------
    # Item
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        data = super().__getitem__(index)
        assert self._last_sample is not None, (
            "parent __getitem__ did not call _stash_sample — check that "
            "Dataset_mix_horiz.__getitem__ stashes metadata before returning"
        )
        traj_id, rgb_id, num_views, split, dataset_dir = self._last_sample

        V = int(num_views)
        T = len(rgb_id)
        # Point-map native resolution = lat_h * 8 x lat_w * 8 = 192 x 320.
        H_pm = self._lat_h * 8
        W_pm = self._lat_w * 8

        pointmap_gt = torch.zeros(V, T, 3, H_pm, W_pm, dtype=torch.float32)
        pointmap_valid = torch.zeros(V, T, H_pm, W_pm, dtype=torch.float32)

        dataset_root = Path(dataset_dir)
        for v in range(V):
            d_min, d_max = self._pm_depth_range.get(v, (None, None))
            try:
                xyz_world, observed_mask = construct_pointmap(
                    dataset_root=dataset_root,
                    traj_id=int(traj_id),
                    split=split,
                    view_idx=v,
                    intrinsics_by_traj=self._intrinsics_by_traj,
                    extrinsics_by_traj=self._extrinsics_by_traj,
                    gripper2wrist=self._gripper2wrist,
                    min_depth_m=d_min,
                    max_depth_m=d_max,
                    return_valid_mask=True,
                )
            except (KeyError, FileNotFoundError, ValueError, OSError):
                # Missing data for this (traj, view): leave zeros, valid=0.
                continue
            rgb_idx = np.clip(
                np.array(rgb_id, dtype=np.int64), 0, xyz_world.shape[0] - 1
            )
            xyz_clip = xyz_world[rgb_idx]                                  # (T, 3, Hpm, Wpm)
            obs_clip = observed_mask[rgb_idx]                              # (T, Hpm, Wpm)
            pointmap_gt[v] = xyz_clip
            # Valid = (finite XYZ) ∧ (genuine pre-clamp observation). The
            # observed_mask excludes pixels at the depth-clamp boundary —
            # those are sky / no-return regions that the clamp folds into
            # fake flat surfaces at 2-3 m. Supervising them used to train
            # the model to predict those fake walls; now they're skipped.
            finite_mask = torch.isfinite(xyz_clip).all(dim=1)              # (T, Hpm, Wpm)
            pointmap_valid[v] = (finite_mask & obs_clip).to(torch.float32)

        data["pointmap_gt"] = pointmap_gt        # (V, T, 3, Hpm, Wpm)
        data["pointmap_valid"] = pointmap_valid  # (V, T, Hpm, Wpm)
        return data
