# depth_extras/meta/

Per-episode camera calibration for the raw DROID dump. Required for pointmap
training (world-frame XYZ ground truth) and for the depth/pointmap eval
metrics. Large jsonl files — not tracked in git; download from the
[DROID-3D calibration dataset](https://huggingface.co/datasets/jaibrdhn/droid_3d_extrinsics).
Access is granted after approval: complete the
[DROID-3D Access Request Form](https://forms.gle/E84Kg4BedVpgMe436) first,
using the same email as your Hugging Face account.

```bash
hf download jaibrdhn/droid_3d_extrinsics --repo-type dataset \
    extrinsics.jsonl camera_intrinsics.jsonl --local-dir depth_extras/meta
```

Expected files:

```
camera_intrinsics.jsonl    # per-episode pinhole intrinsics per camera role (~203 MB)
extrinsics.jsonl           # per-episode factor-graph-optimized extrinsics (~100 MB)
```

⚠️ Both files are looked up by `traj_id`, which only matches your
`raw_index.jsonl` if it contains exactly the same episodes as ours. See the
`traj_id` check under Data preparation in the top-level README.

⚠️ The `T0_ext1_in_world` / `T1_ext2_in_world` fields are world→camera view
matrices despite their names (`depth_extras/extrinsics.py::get_ext_camera_pose`
inverts them), and the wrist camera pose is reconstructed per frame from
joint state via FK — see the dataset card and `depth_extras/extrinsics.py`
for the full conventions.

Paths are overridable via the `camera_intrinsics_path` /
`camera_extrinsics_path` kwargs of `Dataset_mix_horiz_with_pointmap`
(`dataset/dataset_droid_pointmap.py`) and `PointmapHelper`
(`scripts/eval_horiz_utils.py`).
