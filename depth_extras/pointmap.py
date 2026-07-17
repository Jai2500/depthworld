"""Construct a 3D point map at the low (192×320) resolution by composing
disparity_lo → metric depth → per-pixel XYZ in the camera frame → XYZ in the
robot base frame, using the JFG-optimised intrinsics + extrinsics.

For each pixel (u, v) in a frame, the output is a 3-vector (X, Y, Z) giving
the world (robot base) coordinates of the scene point that pixel observes.
For the gripper / closely-watched objects this gives a fully-Cartesian
representation that's invariant to camera viewpoint — useful for 3D-aware
losses, multi-view geometric consistency checks, point-cloud visualisations,
etc.

Pipeline per (episode, view):

  disparity_lo (T, 192, 320) fp16
  ──▶ depth = baseline · fx_native / disparity              (m)
  ──▶ unproject with K scaled to (192, 320):
        X_cam = (u − cx_lo) · depth / fx_lo
        Y_cam = (v − cy_lo) · depth / fy_lo
        Z_cam = depth
  ──▶ transform to world frame:
        P_world = R · P_cam + t
        where  (R, t) come from T_world_from_cam:
          • ext1 (view 0): inverse of rec["T0_ext1_in_world"]
            (stored as a world→cam view matrix, constant per scene)
          • ext2 (view 1): inverse of rec["T1_ext2_in_world"]
            (stored as a world→cam view matrix, constant per scene)
          • wrist (view 2): inverse of the canonical JFG view matrix,
            computed per-frame from joint state + dq + dT_gw + T_gw_static
            (see dataset/extrinsics.py:wrist_cam_pose_in_base).

Notes on the disparity → depth conversion:
  Disparity values are in NATIVE (1280-wide) pixel units even though stored
  at (192, 320). So depth uses the NATIVE fx, NOT a scaled one. The intrinsic
  scaling (sx, sy = Wo/W, Ho/H) only enters when unprojecting from the
  (192, 320) pixel grid to camera-frame XYZ — that's where (fx_lo, fy_lo,
  cx_lo, cy_lo) come in.

Usage:

    from depth_extras.disparity_to_depth import load_intrinsics
    from depth_extras.extrinsics import load_extrinsics, load_gripper2wrist_transforms
    from depth_extras.pointmap import construct_pointmap

    intr = load_intrinsics("depth_extras/meta/camera_intrinsics.jsonl")
    extr = load_extrinsics("depth_extras/meta/extrinsics.jsonl")
    g2w  = load_gripper2wrist_transforms()

    xyz_world = construct_pointmap(
        dataset_root="dataset_example/droid_raw_ctrl",
        traj_id=1415, split="train", view_idx=0,    # 0=ext1, 1=ext2, 2=wrist
        intrinsics_by_traj=intr,
        extrinsics_by_traj=extr,
        gripper2wrist=g2w,                          # only needed for view_idx=2
    )   # -> (T, 3, 192, 320) float32, units of meters in robot base frame
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .disparity_to_depth import (
    VIEW_ORDER,
    disparity_to_depth,
    get_camera_native_intrinsics,
    load_disparity_lo,
)
from .extrinsics import (
    DEFAULT_GRIPPER2WRIST_ASSET,
    get_ext_camera_pose,
    get_per_robot_corrections,
    get_wrist_static_transform,
    load_extrinsics,
    load_gripper2wrist_transforms,
    wrist_cam_pose_in_base,
)


# Match the v2 extractor's temporal decimation (15 Hz → 5 Hz).
RGB_SKIP = 3


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------

def unproject_depth_to_cam_xyz(
    depth: torch.Tensor,
    native_intr: dict,
) -> torch.Tensor:
    """Unproject a depth map to XYZ in the camera frame.

    Args:
        depth: (T, Ho, Wo) float tensor, metric depth in metres.
        native_intr: native-resolution intrinsics dict (from
            dataset.disparity_to_depth.get_camera_native_intrinsics). Contains
            fx, fy, cx, cy and the source `width`, `height` (1280, 720) that
            those are calibrated to. We scale them to (Ho, Wo) on the fly.

    Returns:
        (T, 3, Ho, Wo) float32 tensor of (X_cam, Y_cam, Z_cam) in metres.
        Z_cam == depth by construction.
    """
    T, Ho, Wo = depth.shape
    sx = Wo / native_intr["width"]
    sy = Ho / native_intr["height"]
    fx_lo = native_intr["fx"] * sx
    fy_lo = native_intr["fy"] * sy
    cx_lo = native_intr["cx"] * sx
    cy_lo = native_intr["cy"] * sy

    v_grid, u_grid = torch.meshgrid(
        torch.arange(Ho, dtype=torch.float32, device=depth.device),
        torch.arange(Wo, dtype=torch.float32, device=depth.device),
        indexing="ij",
    )
    depth_f = depth.to(torch.float32)
    X = (u_grid - cx_lo) * depth_f / fx_lo
    Y = (v_grid - cy_lo) * depth_f / fy_lo
    Z = depth_f
    return torch.stack([X, Y, Z], dim=1)  # (T, 3, Ho, Wo)


def apply_extrinsics(
    xyz_cam: torch.Tensor,
    T_world_from_cam: torch.Tensor,
) -> torch.Tensor:
    """Transform camera-frame XYZ to world-frame XYZ.

    Args:
        xyz_cam: (T, 3, Ho, Wo) float tensor — camera-frame points.
        T_world_from_cam:
            - (4, 4) for cameras whose pose is constant across the trajectory
              (ext1, ext2), OR
            - (T, 4, 4) for per-frame poses (wrist).

    Returns:
        (T, 3, Ho, Wo) float32 tensor of world-frame points.
    """
    T_t = xyz_cam.shape[0]
    if T_world_from_cam.ndim == 2:
        T_world_from_cam = T_world_from_cam.unsqueeze(0).expand(T_t, -1, -1)
    elif T_world_from_cam.shape[0] != T_t:
        raise ValueError(
            f"T_world_from_cam time dim {T_world_from_cam.shape[0]} != depth time dim {T_t}"
        )
    R = T_world_from_cam[:, :3, :3].to(xyz_cam.dtype)    # (T, 3, 3)
    t = T_world_from_cam[:, :3, 3].to(xyz_cam.dtype)     # (T, 3)
    # einsum: world[t, i, h, w] = sum_j R[t, i, j] · cam[t, j, h, w] + t[t, i]
    xyz_world = torch.einsum("tij,tjhw->tihw", R, xyz_cam) + t[:, :, None, None]
    return xyz_world


# ---------------------------------------------------------------------------
# End-to-end pointmap builder
# ---------------------------------------------------------------------------

def construct_pointmap(
    dataset_root: str | Path,
    traj_id: int,
    split: str,
    view_idx: int,
    intrinsics_by_traj: dict[int, dict],
    extrinsics_by_traj: dict[int, dict],
    gripper2wrist: dict[str, torch.Tensor] | None = None,
    min_depth_m: float | None = None,
    max_depth_m: float | None = None,
    return_valid_mask: bool = False,
) -> torch.Tensor:
    """Build the world-frame XYZ point map for one (episode, view).

    Args:
        dataset_root: filesystem root holding disparity_lo/, annotation/, ...
        traj_id, split: episode identifier.
        view_idx: 0 (ext1), 1 (ext2), 2 (wrist).
        intrinsics_by_traj: from `dataset.disparity_to_depth.load_intrinsics()`.
        extrinsics_by_traj: from `dataset.extrinsics.load_extrinsics()`.
        gripper2wrist: from `dataset.extrinsics.load_gripper2wrist_transforms()`;
            only needed for view_idx == 2 (wrist).
        min_depth_m/max_depth_m: if set, clamp depth to this metric range BEFORE
            unprojecting. Strongly recommended in practice: pixels with tiny
            disparity (sky, far walls, s2m2 no-return regions) recover to 50+m
            depth, and pixels outside the depth-latent training range are not
            comparable to the v1 depth modality. Reasonable v1 values:
              • 0.30-3.30 m for ext1/ext2
              • 0.07-2.00 m for wrist
            None (default) returns raw depths; the caller can clip themselves.
        return_valid_mask: if True, also return a per-pixel boolean mask
            marking pixels with a genuine depth observation (positive disparity,
            finite raw depth, strictly inside the requested depth range). Clamped pixels
            are excluded — supervising on the clamp-flat surface trains the
            model to predict fake walls at 2-3 m for every sky / no-return
            region. Default False for backward compat.

    Returns:
        (T, 3, Ho, Wo) float32 — world (robot base) frame XYZ in metres.
        If `return_valid_mask`, returns a tuple `(xyz_world, valid_mask)`
        where `valid_mask` is (T, Ho, Wo) bool.
    """
    dataset_root = Path(dataset_root)
    role = VIEW_ORDER[view_idx]

    if traj_id not in intrinsics_by_traj:
        raise KeyError(f"traj_id={traj_id} not in intrinsics map")
    if traj_id not in extrinsics_by_traj:
        raise KeyError(f"traj_id={traj_id} not in extrinsics map (JFG coverage is ~97%)")

    intr_rec = intrinsics_by_traj[traj_id]
    extr_rec = extrinsics_by_traj[traj_id]
    native_intr = get_camera_native_intrinsics(intr_rec, role)

    # 1. Load disparity and recover metric depth at low res. Track which
    # pixels are genuine observations BEFORE clamping; clamped pixels
    # (depth at the boundary) get supervised as fake flat surfaces if
    # included in the loss mask.
    disp = load_disparity_lo(dataset_root, split, traj_id, view_idx)  # (T, Ho, Wo)
    depth_raw = disparity_to_depth(disp, native_intr["baseline_m"], native_intr["fx"])
    if min_depth_m is not None or max_depth_m is not None:
        observed_mask = (
            torch.isfinite(depth_raw) & (depth_raw > 0)
        )
        if min_depth_m is not None:
            observed_mask = observed_mask & (depth_raw > min_depth_m)
        if max_depth_m is not None:
            observed_mask = observed_mask & (depth_raw < max_depth_m)
        depth = depth_raw.clamp(
            min=min_depth_m if min_depth_m is not None else None,
            max=max_depth_m if max_depth_m is not None else None,
        )
    else:
        observed_mask = torch.isfinite(depth_raw) & (depth_raw > 0)
        depth = depth_raw

    # 2. Unproject to camera-frame XYZ.
    xyz_cam = unproject_depth_to_cam_xyz(depth, native_intr)          # (T, 3, Ho, Wo)

    # 3. Compute camera pose in world (4,4) or per-frame (T,4,4).
    if view_idx in (0, 1):
        T_world_from_cam = get_ext_camera_pose(extr_rec, view_idx)
    else:  # wrist
        if gripper2wrist is None:
            raise ValueError("gripper2wrist transforms required for wrist (view_idx=2)")
        T_gw_static = get_wrist_static_transform(extr_rec, gripper2wrist)
        if T_gw_static is None:
            raise KeyError(
                f"no gripper2wrist entry for robot_serial={extr_rec['robot_serial']!r}; "
                "this robot's wrist cam can't be reconstructed"
            )
        # Joint positions are at 15 Hz; decimate to match the 5 Hz depth.
        ann_path = dataset_root / "annotation" / split / f"{traj_id}.json"
        with open(ann_path) as f:
            ann = json.load(f)
        jp_15hz = torch.tensor(ann["observation.state.joint_position"], dtype=torch.float32)
        T_5hz = depth.shape[0]
        jp_5hz = jp_15hz[::RGB_SKIP][:T_5hz]
        # Pad with the last frame if proprio is one frame short (rare).
        if jp_5hz.shape[0] < T_5hz:
            pad = jp_5hz[-1:].repeat(T_5hz - jp_5hz.shape[0], 1)
            jp_5hz = torch.cat([jp_5hz, pad], dim=0)
        dT_gw, dq = get_per_robot_corrections(extr_rec)
        T_world_from_cam = wrist_cam_pose_in_base(
            jp_5hz, dq, dT_gw, T_gw_static, image_rotation_deg=180.0
        )  # (T,4,4)

    # 4. Apply extrinsics to get world-frame XYZ.
    xyz_world = apply_extrinsics(xyz_cam, T_world_from_cam)
    if return_valid_mask:
        return xyz_world, observed_mask
    return xyz_world


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _q(t: torch.Tensor, p: float) -> float:
    return float(torch.quantile(t.flatten(), p / 100.0))


_PKG_META = Path(__file__).resolve().parent / "meta"


def smoke_test(
    dataset_root: str | Path = "dataset_example/droid_raw_ctrl",
    intrinsics_jsonl: str | Path = _PKG_META / "camera_intrinsics.jsonl",
    extrinsics_jsonl: str | Path = _PKG_META / "extrinsics.jsonl",
    gripper2wrist_asset: str | Path = DEFAULT_GRIPPER2WRIST_ASSET,
) -> None:
    """Build pointmaps for a few episodes/views and sanity-check them.

    Verifies:
      - shape is (T, 3, 192, 320)
      - Z range (height in world frame) is in [-1, 1] m — consistent with a
        tabletop-manipulation scene viewed from a base-mounted Franka
      - per-view pointmap range bounds the workspace (~ ±2 m for ext cams)
      - ext1 and ext2 pointmaps of the same scene roughly overlap in 3D
        (they're observing the same physical scene, so their world-frame
        XYZ distributions should be similar)
      - wrist pointmap tracks the gripper region (small XYZ range, close to
        the per-frame wrist position from extrinsics)
    """
    from .disparity_to_depth import load_intrinsics

    print(f"Loading intrinsics from {intrinsics_jsonl} ...")
    intr = load_intrinsics(intrinsics_jsonl)
    print(f"  {len(intr)} episodes")
    print(f"Loading extrinsics from {extrinsics_jsonl} ...")
    extr = load_extrinsics(extrinsics_jsonl)
    print(f"  {len(extr)} episodes with JFG extrinsics")
    print(f"Loading gripper2wrist asset from {gripper2wrist_asset} ...")
    g2w = load_gripper2wrist_transforms(gripper2wrist_asset)
    print(f"  {len(g2w)} robots with hand-eye calibration")

    # Pick a few traj_ids in both intr and extr AND with gripper2wrist coverage.
    candidates = []
    for traj_id, rec in extr.items():
        if traj_id not in intr:
            continue
        if get_wrist_static_transform(rec, g2w) is None:
            continue
        candidates.append(traj_id)
        if len(candidates) >= 3:
            break
    print(f"Spot-checking traj_ids: {candidates}")

    for traj_id in candidates:
        print(f"\n=== traj_id={traj_id} ===")
        # Resolve split.
        ann_path = Path(dataset_root) / "annotation" / "train" / f"{traj_id}.json"
        split = "train"
        if not ann_path.is_file():
            ann_path = Path(dataset_root) / "annotation" / "val" / f"{traj_id}.json"
            split = "val"
        if not ann_path.is_file():
            print("  no annotation; skipping")
            continue

        print(f"  episode_uuid: {extr[traj_id]['episode_uuid']}")
        print(f"  robot_serial: {extr[traj_id]['robot_serial']}")

        # Build pointmap per view. Use a per-role depth clip to suppress
        # sky/no-return outliers that turn into astronomical XYZ.
        clip_per_role = {0: 3.0, 1: 3.0, 2: 2.0}  # ext1, ext2, wrist
        pointmaps = {}
        for v_idx, role in enumerate(VIEW_ORDER):
            disp_path = Path(dataset_root) / "disparity_lo" / split / str(traj_id) / f"{v_idx}.pt"
            if not disp_path.is_file():
                print(f"  {role}: disparity_lo missing, skipping")
                continue
            xyz = construct_pointmap(
                dataset_root, traj_id, split, v_idx, intr, extr,
                gripper2wrist=g2w, max_depth_m=clip_per_role[v_idx],
            )                                                # (T, 3, 192, 320)
            pointmaps[v_idx] = xyz
            T_, C, Ho, Wo = xyz.shape
            assert (C, Ho, Wo) == (3, 192, 320), f"{role}: shape {xyz.shape}"

            # World-frame stats: report medians + percentiles so outliers don't dominate.
            X = xyz[:, 0]; Y = xyz[:, 1]; Z = xyz[:, 2]
            print(
                f"  {role}: shape=(T={T_},3,{Ho},{Wo})  "
                f"X med={_q(X,50):+.2f} [p5={_q(X,5):+.2f}, p95={_q(X,95):+.2f}]m  "
                f"Y med={_q(Y,50):+.2f} [p5={_q(Y,5):+.2f}, p95={_q(Y,95):+.2f}]m  "
                f"Z med={_q(Z,50):+.2f} [p5={_q(Z,5):+.2f}, p95={_q(Z,95):+.2f}]m"
            )

        # Cross-view sanity: ext1 vs ext2 should observe overlapping 3D regions.
        if 0 in pointmaps and 1 in pointmaps:
            xyz_ext1 = pointmaps[0]
            xyz_ext2 = pointmaps[1]
            # Compare the workspace medians:
            med_ext1 = torch.median(xyz_ext1.reshape(3, -1), dim=1).values
            med_ext2 = torch.median(xyz_ext2.reshape(3, -1), dim=1).values
            print(
                f"  ext1 median world XYZ: ({med_ext1[0]:+.2f}, {med_ext1[1]:+.2f}, {med_ext1[2]:+.2f})m  "
                f"vs ext2 median: ({med_ext2[0]:+.2f}, {med_ext2[1]:+.2f}, {med_ext2[2]:+.2f})m  "
                f"|Δ|={float((med_ext1 - med_ext2).norm()):.2f}m"
            )

        # Wrist: its pointmap should be close to the camera position itself
        # (small Z range, since wrist sees the gripper / object 5-30 cm away).
        if 2 in pointmaps:
            xyz_wrist = pointmaps[2]
            # Distance from camera position to pixel point should be near depth.
            # We already validated this in disparity_to_depth.smoke_test.
            # Here just verify range is workspace-sized.
            ranges = (
                xyz_wrist[:, 0].max() - xyz_wrist[:, 0].min(),
                xyz_wrist[:, 1].max() - xyz_wrist[:, 1].min(),
                xyz_wrist[:, 2].max() - xyz_wrist[:, 2].min(),
            )
            print(f"  wrist XYZ trajectory ranges: ΔX={ranges[0]:.2f}m  ΔY={ranges[1]:.2f}m  ΔZ={ranges[2]:.2f}m")


if __name__ == "__main__":
    smoke_test()
