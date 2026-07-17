# depth_extras/meta/

Per-episode camera calibration for the raw DROID dump. Required for pointmap
training (world-frame XYZ ground truth) and for the depth/pointmap eval
metrics. Large jsonl files — not tracked in git; download from the
[DROID-3D calibration dataset](https://huggingface.co/datasets/jaibrdhn/droid_3d_extrinsics):

```bash
hf download jaibrdhn/droid_3d_extrinsics --repo-type dataset \
    extrinsics.jsonl camera_intrinsics.jsonl --local-dir depth_extras/meta
```

Expected files:

```
camera_intrinsics.jsonl    # per-episode pinhole intrinsics per camera role (~203 MB)
extrinsics.jsonl           # per-episode factor-graph-optimized extrinsics (~100 MB)
```

⚠️ The `T0_ext1_in_world` / `T1_ext2_in_world` fields are world→camera view
matrices despite their names (`depth_extras/extrinsics.py::get_ext_camera_pose`
inverts them), and the wrist camera pose is reconstructed per frame from
joint state via FK — see the dataset card and `depth_extras/extrinsics.py`
for the full conventions.

Paths are overridable via the `camera_intrinsics_path` /
`camera_extrinsics_path` kwargs of `Dataset_mix_camera`
(dataset/dataset_droid_exp33_camera.py).
