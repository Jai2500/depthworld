"""Convert an FSDP distributed-checkpoint (dcp) directory into a single .pt
state_dict that `load_single_modality_models` can `torch.load`.

Usage:
    python scripts/convert_dcp_to_single_pt.py \
        --input model_ckpt/<run>/checkpoint-<step>/pytorch_model_fsdp_0 \
        --output model_ckpt/<run>/checkpoint-<step>_merged.pt

Implementation: reads the DCP .metadata directly (which holds parameter names,
shapes, and a (file, offset, length) plan for each tensor), opens each .distcp
shard once, slices out the bytes for each parameter chunk, and reassembles
each parameter. Avoids the expensive `dcp_to_torch_save` path which iterates
the full per-rank state_dict_metadata.
"""

from __future__ import annotations

import argparse
import io
import pickle
import struct
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner


def _strip_known_prefixes(sd: dict, drop_first: bool = True) -> dict:
    out = {}
    for k, v in sd.items():
        kk = k
        for pref in ("model.", "_orig_mod.", "module."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        out[kk] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to pytorch_model_fsdp_0/ dir")
    p.add_argument("--output", required=True, help="output .pt path")
    p.add_argument(
        "--keys_prefixes",
        nargs="*",
        default=["model.unet.", "model.action_encoder.", "model.pointmap_decoder."],
        help=(
            "Only load tensors whose key starts with one of these prefixes. "
            "Default keeps just the UNet and action encoder — enough for "
            "load_single_modality_models warm-start. Pass an empty value "
            "(e.g. '') as the only entry to keep everything."
        ),
    )
    args = p.parse_args()

    in_dir = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[convert] input  = {in_dir}")
    print(f"[convert] output = {out_path}")
    t0 = time.time()

    # Step 1: read the .metadata to enumerate all keys.
    with open(in_dir / ".metadata", "rb") as f:
        md = pickle.load(f)
    all_keys = list(md.state_dict_metadata.keys())
    print(f"[convert] .metadata has {len(all_keys)} keys "
          f"({time.time()-t0:.1f}s)")

    # Step 2: filter to requested prefixes.
    if args.keys_prefixes and any(p for p in args.keys_prefixes):
        prefs = tuple(p for p in args.keys_prefixes if p)
        keep = [k for k in all_keys if k.startswith(prefs)]
    else:
        keep = all_keys
    print(f"[convert] keeping {len(keep)} keys after prefix filter "
          f"({args.keys_prefixes})")

    # Step 3: use dcp's _load_state_dict_from_keys with the filtered set.
    storage_reader = FileSystemReader(in_dir)
    print(f"[convert] reading shards (this can take a few minutes) ...")
    t1 = time.time()
    sd = _load_state_dict_from_keys(
        keys=set(keep),
        storage_reader=storage_reader,
    )
    print(f"[convert] read complete in {time.time()-t1:.1f}s "
          f"(total {time.time()-t0:.1f}s)")

    # Step 4: strip known prefixes (model. / _orig_mod. / module.).
    sd = _strip_known_prefixes(sd)
    n_params = sum(t.numel() for t in sd.values() if isinstance(t, torch.Tensor))
    n_unet = sum(1 for k in sd if k.startswith("unet."))
    n_ae = sum(1 for k in sd if k.startswith("action_encoder."))
    print(f"[convert] merged: {len(sd):,} keys, {n_params:,} params "
          f"(unet.*: {n_unet}, action_encoder.*: {n_ae})")
    print(f"[convert] first 5 keys: {list(sd.keys())[:5]}")

    torch.save(sd, out_path)
    print(f"[convert] wrote {out_path} "
          f"({out_path.stat().st_size / 1e9:.2f} GB, "
          f"total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
