"""Per-episode camera extrinsics from the joint-factor-graph (JFG)
calibration, loaded from `dataset_meta_info/droid_raw_ctrl/extrinsics.jsonl`.

Each record carries:

  Per-scene 4×4 view matrices (robot world → camera frame), despite their
  historical field names:
    T0_ext1_in_world          : ext1 left-camera view matrix
    T1_ext2_in_world          : ext2 left-camera view matrix
    T_w2w_scene_correction    : per-scene world correction (typically near I)

  Per-robot (applies to every episode for that robot_serial):
    dT_gw_gripper_in_world_correction  : 4×4 small correction to analytic FK
    dq_joint_correction                 : 7-dim joint angle offset to apply
                                          before computing wrist-camera pose via FK

Wrist camera pose is NOT stored explicitly — it's reconstructed per-frame from
proprio. The canonical chain (from the JFG codebase) is:

    T_wristcam_from_base(fi) = FLIP · T_gw_static · dT_gw · inv(franka_fk(jp + dq))

Where:
    jp                  : (7,)   joint positions at frame fi (15 Hz, from
                                 annotation observation.state.joint_position)
    dq                  : (7,)   per-robot JFG joint correction (this module)
    franka_fk(jp + dq)  : (4,4)  base→flange (panda_link8) FK
    dT_gw               : (4,4)  per-robot gripper-world correction (this module)
    T_gw_static         : (4,4)  per-robot wrist←gripper hand-eye transform
                                 (dataset/assets/gripper2wrist_transforms.json,
                                  field `mean_mat`)
    FLIP                : (4,4)  diag(-1, -1, 1, 1) — camera convention adjust

The output is a *view matrix* (world→cam). To get the wrist camera *pose in
the base frame* (the more common representation), invert it. Both senses are
exposed below.

Coverage: ~97.1% of indexed episodes have JFG extrinsics; the rest are listed
in extrinsics_missing.json. The gripper2wrist asset covers ~29 robot serials,
which is a strict subset of the labs' robots — episodes whose robot_serial
isn't in the asset can still use ext1/ext2 but won't have wrist-pose
reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_extrinsics(extrinsics_jsonl: str | Path) -> dict[int, dict]:
    """Load extrinsics.jsonl into {traj_id: record}."""
    out: dict[int, dict] = {}
    with open(extrinsics_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["traj_id"])] = rec
    return out


def get_ext_camera_pose(rec: dict, view_idx: int) -> torch.Tensor:
    """Return the 4×4 (world ← cam) pose of an exterior camera for one episode.

    The JFG records store exterior camera transforms as view matrices
    (world → cam), despite the `*_in_world` field names. This helper returns
    the cam → world pose expected by pointmap unprojection and camera
    conditioning code.

    view_idx must be 0 (=ext1) or 1 (=ext2). Wrist camera (view_idx=2) is not
    available in this record; compute it from FK + dq + dT_gw at downstream
    consumer time.
    """
    if view_idx == 0:
        T = rec["T0_ext1_in_world"]
    elif view_idx == 1:
        T = rec["T1_ext2_in_world"]
    else:
        raise ValueError(
            f"view_idx must be 0 (ext1) or 1 (ext2); got {view_idx}. "
            "Wrist camera pose isn't stored per-scene — compute via FK from "
            "joint_position + dq + dT_gw."
        )
    T_world_to_cam = torch.tensor(T, dtype=torch.float32)
    return torch.linalg.inv(T_world_to_cam)


def get_per_robot_corrections(rec: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (dT_gw, dq) — the per-robot calibration corrections to apply
    to analytic FK when computing the wrist-camera pose."""
    dT_gw = torch.tensor(rec["dT_gw_gripper_in_world_correction"], dtype=torch.float32)  # (4,4)
    dq = torch.tensor(rec["dq_joint_correction"], dtype=torch.float32)                    # (7,)
    return dT_gw, dq


def get_scene_world_correction(rec: dict) -> torch.Tensor:
    """Return T_w2w — the 4×4 per-scene world correction (typically near I)."""
    return torch.tensor(rec["T_w2w_scene_correction"], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Wrist camera: reconstructed per-frame from joint state + per-robot calibration
# ---------------------------------------------------------------------------

# Per-robot hand-eye asset path. `mean_mat` is T_wrist_from_gripper (= 4×4
# matrix that maps points in the gripper/flange frame into the wrist-camera
# frame). The same matrix appears as `T_gw` in JFG codebase comments.
DEFAULT_GRIPPER2WRIST_ASSET = (
    Path(__file__).parent / "assets" / "gripper2wrist_transforms.json"
)

# Camera convention flip used by the JFG/V29 codebase. Negates X and Y of the
# wrist-camera frame to align with the renderer's "y-down, x-right" image
# convention. This is built into the asset's intended use.
_FLIP = torch.tensor(
    [[-1.0, 0.0, 0.0, 0.0],
     [ 0.0, -1.0, 0.0, 0.0],
     [ 0.0,  0.0, 1.0, 0.0],
     [ 0.0,  0.0, 0.0, 1.0]],
    dtype=torch.float32,
)


def load_gripper2wrist_transforms(
    path: str | Path = DEFAULT_GRIPPER2WRIST_ASSET,
) -> dict[str, torch.Tensor]:
    """Load the per-robot hand-eye asset into {robot_serial: (4,4) fp32}.

    Returned matrix is `T_wrist_from_gripper` (V29's `T_gw_static`)."""
    with open(path) as f:
        data = json.load(f)
    out: dict[str, torch.Tensor] = {}
    for serial, entry in data.items():
        out[serial] = torch.tensor(entry["mean_mat"], dtype=torch.float32)
    return out


def wrist_cam_view_matrix(
    joint_position: torch.Tensor,
    dq: torch.Tensor,
    dT_gw: torch.Tensor,
    T_gw_static: torch.Tensor,
) -> torch.Tensor:
    """Compute the wrist-camera view matrix (world→cam) for one or many frames.

    This matches the canonical chain used in the JFG codebase (e.g.
    `real/jfg/_dlt_distributions.py::compute_wrist_poses`):

        T_wristcam_from_base(fi) = FLIP · T_gw_static · dT_gw · inv(T_bf(jp+dq))

    Args:
        joint_position: (..., 7)  joint angles in radians at the frame(s) of
                                  interest. Use the 15 Hz values from the
                                  annotation.
        dq:             (7,)      per-robot joint correction (from extrinsics.jsonl).
        dT_gw:          (4, 4)    per-robot gripper-world correction.
        T_gw_static:    (4, 4)    per-robot hand-eye (wrist←gripper).

    Returns:
        (..., 4, 4) view matrix (= world/base → wrist-cam frame).
    """
    from .franka_fk import franka_fk

    q_eff = joint_position + dq                       # (..., 7)
    T_bf = franka_fk(q_eff)                            # (..., 4, 4)
    inv_T_bf = torch.linalg.inv(T_bf)
    flip = _FLIP.to(T_bf.dtype).to(T_bf.device)
    return flip @ T_gw_static @ dT_gw @ inv_T_bf


def wrist_cam_pose_in_base(
    joint_position: torch.Tensor,
    dq: torch.Tensor,
    dT_gw: torch.Tensor,
    T_gw_static: torch.Tensor,
    image_rotation_deg: float = 0.0,
) -> torch.Tensor:
    """Wrist-camera *pose* in the robot base frame (the inverse of the view
    matrix returned by `wrist_cam_view_matrix`).

    `image_rotation_deg` applies an additional local-Z rotation to the returned
    cam→world pose. Use 180 degrees for DROID wrist RGB/depth tensors: the JFG
    FK chain's FLIP matches the optimized frustum convention, while the stored
    wrist images/depth maps are rotated 180 degrees in plane relative to that
    convention. Right-multiplying the pose by Rz(180) is equivalent to rotating
    the camera local X/Y axes while leaving the optical axis fixed.

    Returns (..., 4, 4) such that `T[:3, 3]` is the wrist camera's position
    in robot-base coordinates, and `T[:3, :3]` is its orientation as a basis
    expressed in robot-base coordinates."""
    pose = torch.linalg.inv(wrist_cam_view_matrix(joint_position, dq, dT_gw, T_gw_static))
    if abs(float(image_rotation_deg)) > 1e-6:
        theta = torch.tensor(
            float(image_rotation_deg) * torch.pi / 180.0,
            dtype=pose.dtype,
            device=pose.device,
        )
        c = torch.cos(theta)
        s = torch.sin(theta)
        Rz = torch.eye(4, dtype=pose.dtype, device=pose.device)
        Rz[0, 0] = c
        Rz[0, 1] = -s
        Rz[1, 0] = s
        Rz[1, 1] = c
        pose = pose @ Rz
    return pose


def get_wrist_static_transform(
    rec: dict,
    gripper2wrist_transforms: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """Look up `T_gw_static` for the robot of this episode, or None if the
    robot isn't in the asset (~3-5% of robots are missing)."""
    serial = rec.get("robot_serial", "")
    # Try the literal serial; some asset keys are case-variants of the same
    # robot (e.g. "fr3-..." vs "FR3-..." in metadata). Fall back to a
    # case-insensitive lookup.
    if serial in gripper2wrist_transforms:
        return gripper2wrist_transforms[serial]
    lower = serial.lower()
    for k, v in gripper2wrist_transforms.items():
        if k.lower() == lower:
            return v
    return None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _T_translation(T: torch.Tensor) -> torch.Tensor:
    return T[:3, 3]


_PKG_META = Path(__file__).resolve().parent / "meta"


def smoke_test(
    extrinsics_jsonl: str | Path = _PKG_META / "extrinsics.jsonl",
    raw_index_jsonl: str | Path = "dataset_meta_info/droid_raw_ctrl/raw_index.jsonl",
    test_traj_ids: Iterable[int] | None = None,
) -> None:
    """Load a few episodes' extrinsics, verify shapes and sanity-check the
    translation magnitudes against the Franka workspace geometry (~50 cm
    base-to-ext-camera distance)."""
    extrinsics_jsonl = Path(extrinsics_jsonl)
    raw_index_jsonl = Path(raw_index_jsonl)

    print(f"Loading extrinsics from {extrinsics_jsonl} ...")
    extr = load_extrinsics(extrinsics_jsonl)
    print(f"  {len(extr)} episodes with JFG extrinsics")

    # Also pull raw_index so we can cross-check scene_path == raw_dir.
    print(f"Loading raw_index from {raw_index_jsonl} ...")
    raw: dict[int, dict] = {}
    with open(raw_index_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_meta" in obj:
                continue
            raw[int(obj["traj_id"])] = obj
    print(f"  {len(raw)} indexed episodes")

    # Pick 5 test traj_ids if not provided.
    if test_traj_ids is None:
        sorted_extr = sorted(extr.keys())
        n = len(sorted_extr)
        test_traj_ids = [sorted_extr[i] for i in (0, n // 4, n // 2, 3 * n // 4, n - 1)]

    for traj_id in test_traj_ids:
        print(f"\n=== traj_id={traj_id} ===")
        rec = extr.get(traj_id)
        if rec is None:
            print("  no extrinsics record; skipping")
            continue

        print(f"  episode_uuid:    {rec['episode_uuid']}")
        print(f"  lab:             {rec['lab']}")
        print(f"  robot_serial:    {rec['robot_serial']}")
        print(f"  optimization:    iters={rec['_meta']['n_iters']}  "
              f"stop_reason={rec['_meta']['stop_reason']}")

        # Cross-check scene_path == raw_dir from the index.
        idx = raw.get(traj_id)
        if idx is not None:
            scene_path_match = rec["scene_path"] == idx["raw_dir"]
            print(f"  scene_path == raw_dir: {scene_path_match}")
            if not scene_path_match:
                print(f"      JFG: {rec['scene_path']}")
                print(f"      idx: {idx['raw_dir']}")

        # Ext1 and ext2 poses + translation magnitudes.
        for v_idx, label in [(0, "ext1"), (1, "ext2")]:
            T = get_ext_camera_pose(rec, v_idx)
            assert T.shape == (4, 4), f"{label} pose shape {T.shape} != (4,4)"
            t = _T_translation(T)
            # Last row of homogeneous transform should be [0, 0, 0, 1].
            last_row_ok = torch.allclose(T[3], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)
            # Rotation block should be near-orthogonal (det ≈ ±1).
            R = T[:3, :3]
            det = float(torch.det(R))
            print(
                f"  {label}: t=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})m  "
                f"|t|={torch.norm(t):.3f}m  det(R)={det:+.4f}  "
                f"hom_row_ok={last_row_ok}"
            )

        # Per-robot corrections.
        dT_gw, dq = get_per_robot_corrections(rec)
        # dT_gw should be near identity (small correction).
        I = torch.eye(4)
        delta = (dT_gw - I).abs().max().item()
        # dq should be small (joint offsets are degrees-fraction-of-a-radian level).
        dq_max = dq.abs().max().item()
        print(
            f"  dT_gw: |delta-I|_inf={delta:.4f}  "
            f"dq: max|dq|={dq_max:.4f}rad  ({dq.tolist()})"
        )

        # Per-scene world correction.
        T_w2w = get_scene_world_correction(rec)
        delta_w2w = (T_w2w - I).abs().max().item()
        print(f"  T_w2w: |delta-I|_inf={delta_w2w:.4f}")


def smoke_test_wrist_pose(
    dataset_root: str | Path = "dataset_example/droid_raw_ctrl",
    extrinsics_jsonl: str | Path = _PKG_META / "extrinsics.jsonl",
    gripper2wrist_asset: str | Path = DEFAULT_GRIPPER2WRIST_ASSET,
) -> None:
    """Verify wrist-pose reconstruction on real episodes.

    For each test episode:
      * load its JFG extrinsics record
      * look up gripper2wrist asset for the robot
      * load the annotation's `observation.state.joint_position` (15 Hz)
      * compute the wrist-cam pose-in-base for every frame
      * report translation magnitudes (expect ~0.3-1 m near gripper workspace)
      * check temporal smoothness (consecutive frame distance ≤ ~few cm)
    """
    import json as _json

    dataset_root = Path(dataset_root)
    print(f"Loading JFG extrinsics from {extrinsics_jsonl} ...")
    extr = load_extrinsics(extrinsics_jsonl)
    print(f"  {len(extr)} episodes with JFG extrinsics")

    print(f"Loading gripper2wrist asset from {gripper2wrist_asset} ...")
    g2w = load_gripper2wrist_transforms(gripper2wrist_asset)
    print(f"  {len(g2w)} robot serials in asset")

    # Pick a few traj_ids whose robot is in BOTH the JFG and the asset.
    test_traj_ids = []
    for traj_id, rec in extr.items():
        if get_wrist_static_transform(rec, g2w) is not None:
            test_traj_ids.append(traj_id)
            if len(test_traj_ids) >= 4:
                break
    print(f"Spot-checking traj_ids: {test_traj_ids}")

    for traj_id in test_traj_ids:
        print(f"\n=== traj_id={traj_id} ===")
        rec = extr[traj_id]
        print(f"  episode_uuid:   {rec['episode_uuid']}")
        print(f"  robot_serial:   {rec['robot_serial']}")

        # Locate the annotation; it could be in train or val.
        ann_path = dataset_root / "annotation" / "train" / f"{traj_id}.json"
        split = "train"
        if not ann_path.is_file():
            ann_path = dataset_root / "annotation" / "val" / f"{traj_id}.json"
            split = "val"
        if not ann_path.is_file():
            print("  no annotation on disk, skipping")
            continue
        with open(ann_path) as f:
            ann = _json.load(f)

        # 15 Hz joint state.
        jp_list = ann["observation.state.joint_position"]
        jp = torch.tensor(jp_list, dtype=torch.float32)         # (T_raw, 7)
        T_raw = jp.shape[0]
        print(f"  joint_position: shape=({T_raw}, 7)")

        # Per-robot calibration.
        dT_gw, dq = get_per_robot_corrections(rec)               # (4,4), (7,)
        T_gw_static = get_wrist_static_transform(rec, g2w)       # (4,4)
        print(f"  T_gw_static fro={float(T_gw_static.norm()):.3f}")

        # Compute wrist pose-in-base for every 15-Hz frame.
        T_world_wrist = wrist_cam_pose_in_base(jp, dq, dT_gw, T_gw_static)  # (T,4,4)
        translations = T_world_wrist[:, :3, 3]                    # (T, 3)

        # Range of wrist-cam position over the trajectory.
        t_min = translations.min(dim=0).values
        t_max = translations.max(dim=0).values
        t_norm = translations.norm(dim=-1)
        print(
            f"  wrist pose in base — position bounds:"
            f"  X[{t_min[0]:+.3f}, {t_max[0]:+.3f}]m"
            f"  Y[{t_min[1]:+.3f}, {t_max[1]:+.3f}]m"
            f"  Z[{t_min[2]:+.3f}, {t_max[2]:+.3f}]m"
        )
        print(
            f"  |position|: min={t_norm.min():.3f}m  med={t_norm.median():.3f}m"
            f"  max={t_norm.max():.3f}m"
        )

        # Temporal smoothness: distance between consecutive frames.
        d_step = (translations[1:] - translations[:-1]).norm(dim=-1)
        print(
            f"  consec-frame step (m): min={d_step.min():.4f}"
            f"  med={d_step.median():.4f}  max={d_step.max():.4f}"
        )

        # Rotation sanity: det(R) = +1 every frame.
        R = T_world_wrist[:, :3, :3]
        dets = torch.linalg.det(R)
        print(f"  det(R) range: [{dets.min():.5f}, {dets.max():.5f}]")
        assert (dets - 1.0).abs().max() < 1e-3, f"non-rotation R detected (det range: {dets.min()}, {dets.max()})"

        # Quick view-matrix sanity (it should be the inverse of the pose).
        view = wrist_cam_view_matrix(jp[:5], dq, dT_gw, T_gw_static)   # (5, 4, 4)
        pose = wrist_cam_pose_in_base(jp[:5], dq, dT_gw, T_gw_static)
        ident_err = (view @ pose - torch.eye(4)).abs().max().item()
        print(f"  view @ pose = I :  max-abs-err = {ident_err:.2e}")


if __name__ == "__main__":
    smoke_test()
    print("\n" + "=" * 60)
    print(" WRIST POSE SMOKE TEST")
    print("=" * 60)
    smoke_test_wrist_pose()
