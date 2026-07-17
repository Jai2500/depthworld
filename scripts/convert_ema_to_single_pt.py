"""Fast converter: assemble FSDP-sharded EMA checkpoint into a single state_dict.

DCP's `dcp_to_torch_save` is impractically slow for the 50k world-model run
(measured: >60 min and never completed on a login node — the 157k storage
entries × per-chunk file seeks chokes on Lustre). This script bypasses DCP
entirely by reading the simpler `ema_rank*.pt` files, which are flat torch
saves of `{decay, ema_parameters: List[Tensor]}` per rank.

Layout assumption — verified for the final_rgb_8n_ema/checkpoint-50000 dump:
- `ema_parameters[i]` on rank R is the FSDP shard of the i-th trainable
  parameter, contiguous 1D, possibly zero-length if this rank doesn't hold
  any slice of that parameter.
- Concatenating `ema_parameters[i]` across ranks 0..W-1 yields exactly the
  original 1D flat representation; reshape to the parameter's original
  shape gives back the weight tensor.
- The order of `ema_parameters[i]` matches `model.parameters()` iteration
  order in the source CrtlWorld, which is also the order of the
  `model.unet.*` + `model.action_encoder.*` keys in the DCP `.metadata`
  (after prefix filtering). For checkpoint-50000 this gives 1434 entries,
  which equals `len(ema_parameters)`. Sanity-checked on the first few:
    conv_in.weight (320,8,3,3) → 23040 = ema_parameters[0].numel()
    conv_in.bias   (320,)     →   320 = ema_parameters[1].numel()

If `.metadata` is missing or the count mismatches, the script aborts loudly.

Output keys have the `model.` prefix stripped so the result loads cleanly
into `CrtlWorld` / `CrtlWorldPointmap` via `load_state_dict`.

Usage:
    python scripts/convert_ema_to_single_pt.py \
        --checkpoint_dir model_ckpt/<run>/checkpoint-<step> \
        --output         model_ckpt/<run>/checkpoint-<step>_merged.pt
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import torch


def _strip_prefix(k: str, prefix: str) -> str:
    return k[len(prefix):] if k.startswith(prefix) else k


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint_dir", required=True,
        help="The checkpoint-<step> directory (contains ema_rank*.pt and pytorch_model_fsdp_0/).",
    )
    p.add_argument("--output", required=True, help="Output .pt path.")
    p.add_argument(
        "--key_prefixes",
        nargs="*",
        default=["model.unet.", "model.action_encoder.", "model.pointmap_decoder."],
        help="Filter metadata to keys with these prefixes (preserve order).",
    )
    args = p.parse_args()

    ckpt = Path(args.checkpoint_dir).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Step 1: enumerate keys/shapes. Prefer DCP metadata when present; some
    # Accelerate saves instead write a full `pytorch_model_fsdp.bin` state dict.
    md_path = ckpt / "pytorch_model_fsdp_0" / ".metadata"
    prefs = tuple(args.key_prefixes)
    if md_path.exists():
        with open(md_path, "rb") as f:
            md = pickle.load(f)
        all_keys = list(md.state_dict_metadata.keys())
        sel_keys = [k for k in all_keys if k.startswith(prefs)]
        sel_meta = [md.state_dict_metadata[k] for k in sel_keys]
        sel_shapes = [tuple(m.size) for m in sel_meta]
        print(f"[merge] metadata: {len(all_keys)} keys, "
              f"{len(sel_keys)} after prefix filter {args.key_prefixes} "
              f"({time.time()-t0:.1f}s)")
    else:
        model_bin = ckpt / "pytorch_model_fsdp.bin"
        if not model_bin.exists():
            raise SystemExit(f"missing {md_path} and {model_bin}")
        sd = torch.load(model_bin, map_location="cpu", weights_only=False)
        all_keys = list(sd.keys())
        sel_keys = [k for k in all_keys if k.startswith(prefs)]
        sel_shapes = [tuple(sd[k].shape) for k in sel_keys]
        del sd
        print(f"[merge] model bin: {len(all_keys)} keys, "
              f"{len(sel_keys)} after prefix filter {args.key_prefixes} "
              f"({time.time()-t0:.1f}s)")

    # Step 2: discover EMA rank files; assume rank IDs are contiguous.
    rank_files = sorted(
        ckpt.glob("ema_rank*.pt"),
        key=lambda p: int(p.stem.replace("ema_rank", "")),
    )
    if not rank_files:
        raise SystemExit(f"no ema_rank*.pt under {ckpt}")
    n_ranks = len(rank_files)
    print(f"[merge] found {n_ranks} EMA shards in {ckpt}")

    # Step 3: load each shard once, keep `ema_parameters` lists in memory.
    # 64 × 145 MB = ~9.5 GB peak — fits comfortably on a login node.
    per_rank_params: list[list[torch.Tensor]] = []
    for i, rp in enumerate(rank_files):
        t_load = time.time()
        sd = torch.load(rp, map_location="cpu", weights_only=False)
        eps = sd["ema_parameters"]
        per_rank_params.append(eps)
        if i == 0:
            n_params = len(eps)
            if n_params != len(sel_keys):
                raise SystemExit(
                    f"ema_parameters has {n_params} entries but the "
                    f"metadata filter selected {len(sel_keys)} keys. "
                    f"Adjust --key_prefixes."
                )
            print(f"[merge]  rank {i:>2}: {n_params} entries, "
                  f"{sum(t.numel() for t in eps):,} elems ({time.time()-t_load:.2f}s)")
        elif i % 8 == 0 or i == n_ranks - 1:
            print(f"[merge]  rank {i:>2}: {sum(t.numel() for t in eps):,} elems "
                  f"({time.time()-t_load:.2f}s)")
    print(f"[merge] all shards loaded ({time.time()-t0:.1f}s)")

    # Step 4: for each selected parameter, concat shards across ranks and reshape.
    out_sd: dict[str, torch.Tensor] = {}
    t_asm = time.time()
    total_elem = 0
    for j, (full_key, shape) in enumerate(zip(sel_keys, sel_shapes)):
        chunks = [per_rank_params[r][j] for r in range(n_ranks)]
        # Empty shards (this rank doesn't own a slice) have numel=0; concat is fine.
        flat = torch.cat([c for c in chunks if c.numel() > 0], dim=0)
        target_numel = 1
        for s in shape:
            target_numel *= s
        if flat.numel() != target_numel:
            raise SystemExit(
                f"shape mismatch for {full_key}: assembled flat={flat.numel()} "
                f"vs expected={target_numel} (shape={shape})"
            )
        param = flat.view(shape).contiguous()
        out_sd[_strip_prefix(full_key, "model.")] = param
        total_elem += param.numel()
    print(f"[merge] assembled {len(out_sd):,} params, "
          f"{total_elem:,} elements ({time.time()-t_asm:.1f}s)")

    # Step 5: save.
    n_unet = sum(1 for k in out_sd if k.startswith("unet."))
    n_ae = sum(1 for k in out_sd if k.startswith("action_encoder."))
    print(f"[merge] unet.*: {n_unet}, action_encoder.*: {n_ae}")
    print(f"[merge] first 5 keys: {list(out_sd.keys())[:5]}")
    torch.save(out_sd, out)
    print(f"[merge] wrote {out} "
          f"({out.stat().st_size / 1e9:.2f} GB, total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
