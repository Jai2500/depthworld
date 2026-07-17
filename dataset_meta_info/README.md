# dataset_meta_info/

Per-dataset index + normalization metadata, referenced via `--dataset_cfgs`.
Not tracked in git — produced by the data-preparation pipeline:
`scripts/build_droid_raw_index.py` writes `raw_index.jsonl`, and
`create_meta_info.py` (in this folder) builds `{train,val}_sample.json` +
`stat.json` from the extracted annotations.

Expected layout for the production dataset config (`--dataset_cfgs
droid_raw_ctrl`):

```
droid_raw_ctrl/
  raw_index.jsonl       # per-episode index: traj_id, split, camera serials,
                        #   intrinsics/baseline, file paths, language annotation
  depth_ranges.json     # per-camera-role depth percentiles (analysis
                        #   artifact; the encoding ranges themselves are the
                        #   RANGES_M constants baked into the extractors)
  stat.json             # state_01 / state_99 action-normalization stats
  train_sample.json     # sampled clip index (train split)
  val_sample.json       # sampled clip index (val split)
```

A smaller subset config (same file names under a different subdir) can be
used for smoke tests.
