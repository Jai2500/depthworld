# data/

Dataset root (pass via `--dataset_root_path`). Large — not tracked in git;
built from the raw DROID release via the data-preparation pipeline in the
top-level README (`extract_latent_droid_raw.py` +
`export_disparity_lo.py`).

Expected per-dataset layout (paths are resolved as
`<dataset_root_path>/<entry from dataset_meta_info sample json>`):

```
<name>/
  annotation/{train,val}/<traj_id>.json        # frame-level metadata (state, actions, text)
  latent_videos/{train,val}/<traj_id>/<view>.pt        # SVD-encoded RGB latents (4, T, 24, 40)
  latent_videos_depth/{train,val}/<traj_id>/<view>.pt  # SVD-encoded depth latents
  disparity_lo/{train,val}/<traj_id>/<view>.pt         # raw fp16 disparity (T, 192, 320);
                                                       #   only needed for pointmap training
                                                       #   + depth/pointmap eval metrics
```

`annotation/`, `latent_videos/`, and `latent_videos_depth/` are produced by
`dataset_example/extract_latent_droid_raw.py`; `disparity_lo/` by
`dataset_example/export_disparity_lo.py`.
The depth-latent folder name is overridable via the `depth_folder` /
`depth_root` kwargs of `Dataset_mix_dual`.
