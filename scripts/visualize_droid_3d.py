#!/usr/bin/env python
"""Visualize a DROID-3D episode as a fused world-frame point cloud.

Reference for consuming the DROID-3D calibration
(https://huggingface.co/datasets/jaibrdhn/droid_3d_extrinsics): loads an
episode's raw downsampled disparity (`disparity_lo/`), converts it to metric
depth, unprojects each view through the per-episode intrinsics, applies the
optimized extrinsics (ext1/ext2 from the JFG record; wrist reconstructed
per-frame from joint state via FK + per-robot corrections), and fuses the
views into a single robot-base-frame point cloud.

Conventions handled here (see depth_extras/extrinsics.py for details):
  * The stored `T0_ext1_in_world` / `T1_ext2_in_world` matrices are
    world->camera VIEW matrices despite their names — `get_ext_camera_pose`
    inverts them to camera->world poses.
  * The wrist camera pose is not stored; it is rebuilt per frame as
    inv(FLIP · T_gw_static · dT_gw · inv(franka_fk(q + dq))) and then
    rotated 180° about the optical axis because the stored wrist
    images/disparity are upside-down relative to the optimized frustum.

Outputs a PLY (one file per requested frame, points colored per view) or an
interactive viser server (`--viser`, requires `pip install viser`).

Examples:
    # Fused cloud of frame 0 -> droid3d_traj100_f000.ply
    python scripts/visualize_droid_3d.py --traj_id 100 --split val

    # Interactive: frame slider at http://localhost:8090
    python scripts/visualize_droid_3d.py --traj_id 100 --split val --viser
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_example.extract_latent_droid_raw import RANGES_M, VIEW_ORDER
from depth_extras.disparity_to_depth import load_intrinsics
from depth_extras.extrinsics import (
    DEFAULT_GRIPPER2WRIST_ASSET,
    load_extrinsics,
    load_gripper2wrist_transforms,
)
from depth_extras.pointmap import construct_pointmap

VIEW_COLORS = {
    "ext1": np.array([60, 170, 255], dtype=np.uint8),
    "ext2": np.array([255, 170, 60], dtype=np.uint8),
    "wrist": np.array([190, 90, 255], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traj_id", type=int, required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--dataset_root", default="data/droid_raw_ctrl")
    p.add_argument("--intrinsics_jsonl",
                   default="depth_extras/meta/camera_intrinsics.jsonl")
    p.add_argument("--extrinsics_jsonl",
                   default="depth_extras/meta/extrinsics.jsonl")
    p.add_argument("--views", default="ext1,ext2,wrist",
                   help="comma-separated subset of ext1,ext2,wrist")
    p.add_argument("--frames", default="0",
                   help="comma-separated frame indices (5 Hz), or 'all'")
    p.add_argument("--max_depth_grad", type=float, default=0.05,
                   help="drop pixels whose 3D distance to a pixel neighbour "
                        "exceeds this many metres (kills depth-edge floaters); "
                        "0 disables")
    p.add_argument("--out_prefix", default="",
                   help="PLY output prefix (default droid3d_traj<ID>)")
    p.add_argument("--viser", action="store_true",
                   help="serve interactively with a frame slider instead of "
                        "writing PLY files")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--point_stride", type=int, default=1,
                   help="subsample pixels by this stride (speed/size)")
    return p.parse_args()


def edge_mask(xyz: torch.Tensor, max_jump: float) -> torch.Tensor:
    """(T, 3, H, W) world XYZ -> (T, H, W) bool mask dropping pixels whose 3D
    distance to a pixel neighbour exceeds `max_jump` metres (depth-edge
    floaters stretched across discontinuities)."""
    T, _, H, W = xyz.shape
    if max_jump <= 0:
        return torch.ones((T, H, W), dtype=torch.bool)
    du = torch.zeros((T, H, W), dtype=xyz.dtype)
    dv = torch.zeros((T, H, W), dtype=xyz.dtype)
    du[:, :, :-1] = (xyz[:, :, :, 1:] - xyz[:, :, :, :-1]).norm(dim=1)
    dv[:, :-1, :] = (xyz[:, :, 1:, :] - xyz[:, :, :-1, :]).norm(dim=1)
    return (du < max_jump) & (dv < max_jump)


def build_episode_clouds(args) -> tuple[list[dict], int]:
    """Returns (per-view dicts with xyz (T,3,H,W) + mask (T,H,W), n_frames)."""
    intr = load_intrinsics(args.intrinsics_jsonl)
    extr = load_extrinsics(args.extrinsics_jsonl)
    g2w = load_gripper2wrist_transforms(DEFAULT_GRIPPER2WRIST_ASSET)

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    clouds: list[dict] = []
    n_frames = None
    for role in views:
        v_idx = VIEW_ORDER.index(role)
        d_min, d_max = RANGES_M[role]
        try:
            xyz, valid = construct_pointmap(
                args.dataset_root, args.traj_id, args.split, v_idx,
                intrinsics_by_traj=intr, extrinsics_by_traj=extr,
                gripper2wrist=g2w, min_depth_m=d_min, max_depth_m=d_max,
                return_valid_mask=True,
            )
        except KeyError as e:
            print(f"[visualize_droid_3d] skipping {role}: {e}", flush=True)
            continue
        keep = valid & edge_mask(xyz, args.max_depth_grad)
        clouds.append({"role": role, "xyz": xyz, "mask": keep})
        n_frames = xyz.shape[0] if n_frames is None else min(n_frames, xyz.shape[0])
    if not clouds:
        raise SystemExit("no views could be reconstructed")
    return clouds, int(n_frames)


def frame_points(clouds: list[dict], f: int, stride: int):
    pts, cols = [], []
    for c in clouds:
        m = c["mask"][f]
        xyz = c["xyz"][f].permute(1, 2, 0)              # (H, W, 3)
        if stride > 1:
            xyz = xyz[::stride, ::stride]
            m = m[::stride, ::stride]
        p = xyz[m].numpy()
        pts.append(p)
        cols.append(np.tile(VIEW_COLORS[c["role"]], (p.shape[0], 1)))
    return np.concatenate(pts, axis=0), np.concatenate(cols, axis=0)


def write_ply(path: str, pts: np.ndarray, cols: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rec = np.empty(pts.shape[0],
                   dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    rec["xyz"] = pts.astype(np.float32)
    rec["rgb"] = cols
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        rec.tofile(f)


def main() -> None:
    args = parse_args()
    clouds, n_frames = build_episode_clouds(args)

    if args.frames == "all":
        frames = list(range(n_frames))
    else:
        frames = [int(x) for x in args.frames.split(",") if x.strip()]
        frames = [f for f in frames if 0 <= f < n_frames]

    if args.viser:
        import time

        import viser

        server = viser.ViserServer(port=args.port)
        slider = server.gui.add_slider(
            "frame", min=0, max=n_frames - 1, step=1, initial_value=frames[0]
        )

        def show(f: int) -> None:
            pts, cols = frame_points(clouds, f, args.point_stride)
            server.scene.add_point_cloud(
                "/droid3d", points=pts, colors=cols, point_size=0.004
            )

        slider.on_update(lambda _: show(int(slider.value)))
        show(frames[0])
        print(f"[visualize_droid_3d] serving on http://localhost:{args.port} "
              "(ctrl-c to stop)", flush=True)
        while True:
            time.sleep(1.0)

    prefix = args.out_prefix or f"droid3d_traj{args.traj_id}"
    for f in frames:
        pts, cols = frame_points(clouds, f, args.point_stride)
        out = f"{prefix}_f{f:03d}.ply"
        write_ply(out, pts, cols)
        print(f"[visualize_droid_3d] wrote {out}  ({pts.shape[0]:,} points)",
              flush=True)


if __name__ == "__main__":
    main()
