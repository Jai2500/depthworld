"""depth_extras — standalone helpers for DROID raw stereo → depth → 3D pointmaps.

Pipeline:
    disparity_lo (T, Ho, Wo) fp16
      ──▶ disparity_to_depth.disparity_to_depth(...)    metric depth (m)
      ──▶ pointmap.unproject_depth_to_cam_xyz(...)      camera-frame XYZ
      ──▶ pointmap.apply_extrinsics(...)                world-frame XYZ
                                                        (robot base = world)
For wrist (view 2): per-frame T_world_from_cam from
`extrinsics.wrist_cam_pose_in_base`, which calls `franka_fk.franka_fk`.

This package depends only on `torch`, `numpy`, and (for the video script)
`mediapy`. It does NOT depend on the rest of this repo and can be
copied into another project as-is.

Layout:
    disparity_to_depth.py  disparity → depth, intrinsics loader
    extrinsics.py          JFG extrinsics + wrist-pose chain
    franka_fk.py           analytical Franka FK (modified DH)
    pointmap.py            end-to-end camera + world XYZ builder
    save_pointmap_videos.py CLI script that writes RGB MP4s
    assets/gripper2wrist_transforms.json  per-robot hand-eye

Common one-liner:

    from depth_extras.pointmap import construct_pointmap
    from depth_extras.disparity_to_depth import load_intrinsics
    from depth_extras.extrinsics import load_extrinsics, load_gripper2wrist_transforms

    intr = load_intrinsics("camera_intrinsics.jsonl")
    extr = load_extrinsics("extrinsics.jsonl")
    g2w  = load_gripper2wrist_transforms()

    xyz_world = construct_pointmap(
        dataset_root="droid_raw_ctrl", traj_id=1415, split="train",
        view_idx=2,  # wrist
        intrinsics_by_traj=intr, extrinsics_by_traj=extr,
        gripper2wrist=g2w, max_depth_m=2.0,
    )  # -> (T, 3, 192, 320) float32, world-frame XYZ in metres
"""
