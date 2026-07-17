"""Shared helpers for horiz (single-UNet RGB|depth-on-width) evaluation.

Used by `scripts/eval_horiz_chunk.py` and `scripts/eval_horiz_rollout.py`. Provides:

  * Horiz model construction (auto-detects the latent-fed DPT pointmap head).
  * Dataset wrapper that surfaces (episode_id, split) per sample.
  * Depth-latent -> metric-depth inversion using `RANGES_M` (per-view).
  * Pointmap helpers: camera calibration lookup + extrinsics for depth
    unprojection; pointmap-decoder direct readout.
  * Standard mono-depth metrics (abs_rel, sq_rel, rmse, rmse_log, d1/d2/d3).
  * Pointmap metrics (L1, log-L1).
  * RGB metrics (LPIPS, PSNR, SSIM).
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Repo-relative imports (eval scripts insert repo root into sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_example.extract_latent_droid_raw import RANGES_M, VIEW_ORDER
from depth_extras.disparity_to_depth import (
    get_camera_native_intrinsics,
    load_intrinsics,
)
from depth_extras.extrinsics import (
    DEFAULT_GRIPPER2WRIST_ASSET,
    get_ext_camera_pose,
    get_per_robot_corrections,
    get_wrist_static_transform,
    load_extrinsics,
    load_gripper2wrist_transforms,
    wrist_cam_pose_in_base,
)

def canonical_action_space(action_space: str | None) -> str:
    action_space = str(action_space or "cartesian").lower()
    aliases = {
        "ee": "cartesian",
        "eef": "cartesian",
        "end_effector": "cartesian",
        "cartesian_position": "cartesian",
        "joint": "joint_position",
        "joints": "joint_position",
        "joint_pos": "joint_position",
        "qpos": "joint_position",
    }
    return aliases.get(action_space, action_space)


def apply_action_space_config(cfg) -> None:
    cfg.action_space = canonical_action_space(getattr(cfg, "action_space", "cartesian"))
    if cfg.action_space == "cartesian":
        cfg.action_dim = 7
    elif cfg.action_space == "joint_position":
        cfg.action_dim = 8
    else:
        raise ValueError(
            f"Unknown action_space={cfg.action_space!r}. "
            "Expected 'cartesian' or 'joint_position'."
        )


def filter_state_dict_for_model(state_dict: dict, model, *, prefix: str) -> dict:
    """Drop checkpoint tensors whose shape cannot load into this model.

    This is mainly for action-space changes: cartesian checkpoints have a 7D
    first action-projection weight, while joint+gripper models expect 8D.
    """
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is not None and tuple(target.shape) != tuple(value.shape):
            skipped.append((key, tuple(value.shape), tuple(target.shape)))
            continue
        filtered[key] = value
    if skipped:
        print(
            f"[{prefix}] skipped {len(skipped)} shape-mismatched checkpoint keys; "
            f"first={skipped[:3]}",
            flush=True,
        )
    return filtered


# ── Sparse-history rollout buffer ──────────────────────────────────────────
# Maintains a chunk-strided ring buffer where *one* item is appended per
# autoregressive chunk (the chunk's final frame).
# Each `stack()` call samples 6 slots via `history_idx`, producing the same
# sparse-history pattern the model was trained against.
#
# The two `0` slots in the default history_idx always look up `buffer[0]` =
# the *initial* observed frame, which never gets overwritten — so the model
# sees the GT anchor as a stable conditioning signal throughout the rollout,
# with the other slots filling in as chunks accumulate.

DEFAULT_HISTORY_IDX = [0, 0, -12, -9, -6, -3]


def resolve_history_idx(cfg, args_cli=None) -> list[int]:
    """Pick history_idx from CLI override > cfg.history_idx > DEFAULT.

    `args_cli` may carry a `--history_idx` flag as a comma-separated string
    (e.g. "0,0,-12,-9,-6,-3"); it overrides whatever `cfg.history_idx` holds.
    """
    if args_cli is not None and getattr(args_cli, "history_idx", None):
        raw = args_cli.history_idx
        if isinstance(raw, str):
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
        return [int(x) for x in raw]
    hist = getattr(cfg, "history_idx", None)
    if hist is None:
        return list(DEFAULT_HISTORY_IDX)
    return [int(x) for x in hist]


class SparseHistoryBuffer:
    """Chunk-strided ring buffer with sparse history_idx sampling.

    Usage::

        buf = SparseHistoryBuffer(observed_latent, history_idx)
        # Each chunk:
        hist = buf.stack(dim=1)           # (B, num_his, C, H, W) for latents
                                          # (B, num_his, A)       for actions
        # ... denoise ...
        buf.append(pred_latents[:, -1])   # one entry per chunk (the chunk's last frame)

    For per-view tensors of shape (B, V, 4) or (B, V, 3) or (B, V), pass
    ``stack_dim=2`` so the stacked tensor has time on dim 2: (B, V, T, ·).
    """

    def __init__(self, initial_value: torch.Tensor, history_idx: list[int]):
        self.history_idx = list(history_idx)
        # Initial buffer length: longest negative offset, padded to
        # `len(history_idx) * 4` slots.
        self.buffer: list[torch.Tensor] = [initial_value] * (len(history_idx) * 4)

    def stack(self, dim: int = 1) -> torch.Tensor:
        return torch.stack([self.buffer[idx] for idx in self.history_idx], dim=dim)

    def append(self, value: torch.Tensor) -> None:
        self.buffer.append(value)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def _patch_transformers_tp_plan_none_bug() -> None:
    import transformers.modeling_utils as modeling_utils

    if getattr(modeling_utils, "_ctrl_world_tp_plan_none_patch_applied", False):
        return
    orig_fn = getattr(modeling_utils, "get_total_byte_count", None)
    if orig_fn is None:
        return

    def _patched_get_total_byte_count(model, *args, **kwargs):
        if getattr(model, "_tp_plan", None) is None:
            model._tp_plan = []
        return orig_fn(model, *args, **kwargs)

    modeling_utils.get_total_byte_count = _patched_get_total_byte_count
    modeling_utils._ctrl_world_tp_plan_none_patch_applied = True


# ---------------------------------------------------------------------------
# Horizontal-stack ("horiz") architecture support
#
# The horiz checkpoint trains a SINGLE CrtlWorld UNet on a dataset that packs
# RGB and depth latents side-by-side per view (width is doubled to 2*W_lat).
# At decode time the VAE produces a wide image (V*H_img tall, 2*W_img wide)
# where the left half is RGB and the right half is depth-as-grayscale; the
# depth half is then decoded back to metres using the same per-view RANGES_M.
# ---------------------------------------------------------------------------


class Dataset_mix_horiz_with_meta:
    """Lazy-imported wrapper around `Dataset_mix_horiz`. Adds episode_id /
    split / dataset_dir / frame_ids to each batch via an RNG-replay pattern:
    seed RNG, call the parent, then rederive the sample metadata by
    reproducing the parent's selection prefix under the same RNG state.
    """

    def __new__(cls, *args, **kwargs):
        from dataset.dataset_droid import Dataset_mix_horiz

        class _Inner(Dataset_mix_horiz):
            def __getitem__(self, index):
                import random
                import numpy as _np

                rng_state_random = random.getstate()
                rng_state_np = _np.random.get_state()
                torch_rng_state = torch.get_rng_state()

                sample_data = super().__getitem__(index)

                random.setstate(rng_state_random)
                _np.random.set_state(rng_state_np)
                torch.set_rng_state(torch_rng_state)

                if self._use_gripper_filter:
                    skip = random.randint(1, 2) if self.mode != "val" else 1
                    use_filtered = random.random() < self.gripper_filter_sample_prob
                    if use_filtered:
                        dataset_ids, weights = self._dataset_choices_by_skip[skip]
                        dataset_id = int(_np.random.choice(dataset_ids, p=weights))
                    else:
                        dataset_id = _np.random.choice(
                            len(self.samples_all), p=self.prob
                        )
                else:
                    dataset_id = _np.random.choice(len(self.samples_all), p=self.prob)
                    skip = random.randint(1, 2) if self.mode != "val" else 1

                if self._use_gripper_filter:
                    if use_filtered:
                        valid_indices = self._valid_sample_indices_by_skip[
                            dataset_id
                        ][skip]
                        sample_idx = valid_indices[index % len(valid_indices)]
                    else:
                        sample_idx = index % len(self.samples_all[dataset_id])
                else:
                    sample_idx = index % len(self.samples_all[dataset_id])

                sample_meta = self.samples_all[dataset_id][sample_idx]
                dataset_dir = self.dataset_path_all[dataset_id][sample_idx]
                # Replay frame-index construction (matches parent verbatim).
                import json as _json
                ann_file = (
                    f"{dataset_dir}/{self.args.annotation_name}/{self.mode}/"
                    f"{sample_meta['episode_id']}.json"
                )
                with open(ann_file, "r") as _f:
                    label = _json.load(_f)
                joint_len = len(label["observation.state.joint_position"]) - 1
                frame_len = _np.floor(joint_len / int(self.args.down_sample))
                skip_his = int(skip * 4) if self.sparse_history else skip
                _ = random.random()  # consume the his_dropout draw to stay aligned
                if self.his_dropout:
                    # Parent overwrites skip_his when p<0.15; we don't know the
                    # exact draw here without consuming RNG further, so leave
                    # skip_his at its post-replay value. frame_ids may differ
                    # by one shift relative to the actual sample under his_dropout
                    # — this is fine for raw-disparity GT alignment which looks
                    # up by traj_id + index, not exact frame mapping.
                    pass
                frame_ids = list(sample_meta["frame_ids"])
                frame_now = frame_ids[0]
                rgb_id = []
                for i in range(self.args.num_history, 0, -1):
                    rgb_id.append(int(frame_now - i * skip_his))
                rgb_id.append(frame_now)
                for i in range(1, self.args.num_frames):
                    rgb_id.append(int(frame_now + i * skip))
                rgb_id = _np.array(rgb_id)
                rgb_id = _np.clip(rgb_id, 0, frame_len).tolist()
                rgb_id = [int(f) for f in rgb_id]

                sample_data["episode_id"] = int(sample_meta["episode_id"])
                sample_data["split"] = str(self.mode)
                sample_data["dataset_dir"] = str(dataset_dir)
                sample_data["frame_ids"] = rgb_id
                return sample_data

        return _Inner(*args, **kwargs)


def build_horiz_model(cfg, state_dict: dict) -> "object":
    """Construct a `CrtlWorld` (single UNet) and load the horiz checkpoint.

    `cfg.width` MUST be set to 640 (2× the per-view RGB width) before calling;
    the UNet expects the doubled-width latent shape.
    """
    _patch_transformers_tp_plan_none_bug()
    has_pointmap = any(k.startswith("pointmap_decoder.") for k in state_dict.keys())
    if has_pointmap:
        from models.ctrl_world_pointmap import CrtlWorldPointmap
        from models.pointmap_decoder import DPTPointMapDecoder

        if any(k.startswith("pointmap_decoder.feature_projs.") for k in state_dict.keys()):
            raise ValueError(
                "This checkpoint's pointmap head reads intermediate U-Net "
                "features (pointmap_decoder.feature_projs.* keys), which is not "
                "supported by this codebase (latent-fed DPT only)."
            )
        if not any(k.startswith("pointmap_decoder.stem.") for k in state_dict.keys()):
            raise ValueError(
                "This checkpoint's pointmap head does not match the DPT decoder "
                "architecture in this codebase (missing pointmap_decoder.stem.* "
                "keys)."
            )
        model = CrtlWorldPointmap(cfg)
        stem_w = state_dict.get("pointmap_decoder.stem.0.block.0.weight")
        refine_w = state_dict.get("pointmap_decoder.scratch.refinenet1.out_conv.weight")
        base_ch = int(stem_w.shape[0]) if stem_w is not None else 64
        features = int(refine_w.shape[0]) if refine_w is not None else 256
        if "pointmap_decoder.scratch.output_conv2.2.weight" not in state_dict:
            raise ValueError(
                "This checkpoint's pointmap head does not match the DPT decoder "
                "architecture in this codebase (missing "
                "pointmap_decoder.scratch.output_conv2.* keys)."
            )
        decoder = DPTPointMapDecoder(
            base_ch=base_ch,
            features=features,
        )
        print(
            f"[eval_horiz] Attaching DPT pointmap decoder base_ch={base_ch} "
            f"features={features}",
            flush=True,
        )
        model.attach_pointmap_decoder(decoder, loss_weight=0.0, train_decoder=False)
    else:
        from models.ctrl_world import CrtlWorld
        model = CrtlWorld(cfg)

    state_dict = filter_state_dict_for_model(state_dict, model, prefix="eval_horiz")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(
            f"[eval_horiz] load_state_dict missing keys ({len(missing)}):",
            flush=True,
        )
        for k in missing[:20]:
            print(f"  - {k}", flush=True)
        if len(missing) > 20:
            print(f"  ... ({len(missing) - 20} more)", flush=True)
    if unexpected:
        print(
            f"[eval_horiz] load_state_dict unexpected keys ({len(unexpected)}):",
            flush=True,
        )
        for k in unexpected[:20]:
            print(f"  - {k}", flush=True)
        if len(unexpected) > 20:
            print(f"  ... ({len(unexpected) - 20} more)", flush=True)
    return model


@torch.no_grad()
def decode_horiz_latents(
    model,
    latents: torch.Tensor,
    decode_chunk_size: int = 8,
    ranges: dict = RANGES_M,
) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """Decode horiz latents and split width into RGB + depth halves.

    Args:
        latents: (B, F, 4, V*H_lat, 2*W_lat). The width axis carries RGB on
            the left half and depth-as-grayscale on the right half.
    Returns:
        rgb_uint8     : (V, B*F, H_img, W_img, 3) uint8 — visible image.
        depth_metric  : (V, B*F, H_img, W_img) float32 — metres after per-view
                        inverse log+range with `RANGES_M`.
        depth_u8      : (V, B*F, H_img, W_img) uint8 — encoder-side gray image
                        (close=bright). Useful for video saving and
                        PSNR/SSIM-on-grayscale comparisons.
    """
    vae = model.vae
    B, Fr, C, VH, W2 = latents.shape
    flat = latents.reshape(B * Fr, C, VH, W2) / vae.config.scaling_factor
    decoded_chunks = []
    for i in range(0, flat.shape[0], decode_chunk_size):
        chunk = flat[i : i + decode_chunk_size].to(dtype=vae.dtype)
        decoded_chunks.append(vae.decode(chunk, num_frames=chunk.shape[0]).sample)
    decoded = torch.cat(decoded_chunks, dim=0)              # (B*F, 3, V*H_img, 2*W_img)
    decoded = decoded.clamp(-1, 1)
    V = 3
    H_img = decoded.shape[-2] // V
    W_img = decoded.shape[-1] // 2

    rgb_half = decoded[..., :W_img]                          # (B*F, 3, V*H, W_img)
    depth_half = decoded[..., W_img:]                        # (B*F, 3, V*H, W_img)

    rgb_u8_t = ((rgb_half + 1) / 2 * 255).to(torch.uint8)
    rgb_u8_t = rgb_u8_t.reshape(B * Fr, 3, V, H_img, W_img).permute(2, 0, 3, 4, 1)
    rgb_u8 = rgb_u8_t.cpu().numpy()                          # (V, B*F, H, W, 3)

    # Depth half: average 3 channels (encoder replicated), invert per-view.
    dep_norm = (depth_half.float() + 1.0) / 2.0              # [0,1] close=1 (g_inv)
    dep_norm = dep_norm.mean(dim=1)                          # (B*F, V*H, W_img)
    dep_norm = dep_norm.reshape(B * Fr, V, H_img, W_img).clamp(0.0, 1.0)
    g = 1.0 - dep_norm
    depth_m_t = torch.zeros_like(g)
    for v_idx, role in enumerate(VIEW_ORDER):
        d_min, d_max = ranges[role]
        log_min = math.log(d_min)
        log_max = math.log(d_max)
        depth_m_t[:, v_idx] = (log_min + g[:, v_idx] * (log_max - log_min)).exp()
    # Reshape to (V, B*F, H, W)
    depth_m = depth_m_t.permute(1, 0, 2, 3).cpu().float().numpy()

    # Encoder-side uint8 grayscale for video saving (and depth-as-image PSNR).
    # Match `depth_to_uint8_for_video`: close objects are bright, far objects
    # are dark.
    g_u8 = ((1.0 - g) * 255.0).clamp(0, 255).to(torch.uint8) # (B*F, V, H, W)
    g_u8 = g_u8.permute(1, 0, 2, 3).cpu().numpy()            # (V, B*F, H, W)

    return rgb_u8, torch.from_numpy(depth_m), g_u8


# ---------------------------------------------------------------------------
# Decoding latents
# ---------------------------------------------------------------------------


@torch.no_grad()
def decode_rgb_latent(model, latents: torch.Tensor, decode_chunk_size: int = 8) -> np.ndarray:
    """Decode RGB latents (B, F, 4, V*H_lat, W_lat) -> (V, B*F, H_img, W_img, 3) uint8.

    Views are height-stacked in groups of H_img — splitting along height
    rebuilds per-view images. B is typically 1 for eval.
    """
    vae = model.vae
    B, Fr, C, VH, W_lat = latents.shape
    flat = latents.reshape(B * Fr, C, VH, W_lat) / vae.config.scaling_factor
    decoded_chunks = []
    for i in range(0, flat.shape[0], decode_chunk_size):
        chunk = flat[i : i + decode_chunk_size].to(dtype=vae.dtype)
        decoded_chunks.append(vae.decode(chunk, num_frames=chunk.shape[0]).sample)
    decoded = torch.cat(decoded_chunks, dim=0)              # (B*F, 3, V*H_img, W_img)
    img = ((decoded.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
    # Split views along height. V is the dataset's num_views (typically 3).
    V = 3
    H_img = img.shape[-2] // V
    W_img = img.shape[-1]
    img = img.reshape(B * Fr, 3, V, H_img, W_img).permute(2, 0, 3, 4, 1)
    return img.cpu().numpy()                                 # (V, B*F, H, W, 3)


@torch.no_grad()
def decode_depth_latent_to_metric(
    model,
    latents_depth: torch.Tensor,
    ranges: dict = RANGES_M,
    decode_chunk_size: int = 8,
) -> torch.Tensor:
    """Decode depth latents and invert the log+range encoding to metric depth.

    Args:
        latents_depth: (B, F, 4, V*H_lat, W_lat).
    Returns:
        (B, V, F, H_img, W_img) float32 — metric depth in metres.
    """
    decoded = model._decode_depth_latent(latents_depth)      # (B, F, 3, V*H_img, W_img)
    B, Fr, C, VH, W_img = decoded.shape
    V = 3
    H_img = VH // V
    # Split height into views.
    per_view = decoded.reshape(B, Fr, C, V, H_img, W_img).permute(0, 3, 1, 2, 4, 5)
    # per_view: (B, V, F, 3, H, W) in [-1, 1].
    g_inv = ((per_view.float().mean(dim=3) + 1.0) / 2.0).clamp(0.0, 1.0)  # (B, V, F, H, W)
    g = 1.0 - g_inv
    # Per-view inverse exp.
    depth_m = torch.zeros_like(g)
    for v_idx, role in enumerate(VIEW_ORDER):
        d_min, d_max = ranges[role]
        log_min = math.log(d_min)
        log_max = math.log(d_max)
        depth_m[:, v_idx] = (log_min + g[:, v_idx] * (log_max - log_min)).exp()
    return depth_m                                            # (B, V, F, H, W) metres


# ---------------------------------------------------------------------------
# Pointmap helpers (camera calibration lookup for depth unprojection)
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parent.parent


class PointmapHelper:
    """Caches intrinsics / extrinsics / hand-eye maps and offers per-(traj,
    view) native intrinsics + world-from-camera transforms on demand."""

    def __init__(
        self,
        camera_intrinsics_path: Optional[str] = None,
        camera_extrinsics_path: Optional[str] = None,
        gripper2wrist_path: Optional[str] = None,
    ):
        intr_path = Path(
            camera_intrinsics_path
            or _REPO_ROOT / "depth_extras" / "meta" / "camera_intrinsics.jsonl"
        )
        extr_path = Path(
            camera_extrinsics_path
            or _REPO_ROOT / "depth_extras" / "meta" / "extrinsics.jsonl"
        )
        g2w_path = Path(gripper2wrist_path or DEFAULT_GRIPPER2WRIST_ASSET)
        self._intr_by_traj = load_intrinsics(intr_path)
        self._extr_by_traj = load_extrinsics(extr_path)
        self._g2w = load_gripper2wrist_transforms(g2w_path)

    def has_traj(self, traj_id: int) -> bool:
        return int(traj_id) in self._intr_by_traj and int(traj_id) in self._extr_by_traj

    def get_native_intr(self, traj_id: int, view_idx: int) -> Optional[dict]:
        rec = self._intr_by_traj.get(int(traj_id))
        if rec is None:
            return None
        try:
            return get_camera_native_intrinsics(rec, VIEW_ORDER[view_idx])
        except (KeyError, ValueError):
            return None

    def get_world_from_cam(
        self,
        traj_id: int,
        view_idx: int,
        joint_position_5hz: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return (4,4) for ext1/ext2 or (T,4,4) for wrist.

        `joint_position_5hz` is required for the wrist (view_idx == 2), shape
        (T, 7) — joint state decimated to match the depth/latent temporal grid.
        """
        rec = self._extr_by_traj.get(int(traj_id))
        if rec is None:
            return None
        if view_idx in (0, 1):
            try:
                return get_ext_camera_pose(rec, view_idx)
            except (KeyError, ValueError):
                return None
        if view_idx == 2:
            if joint_position_5hz is None:
                return None
            T_gw = get_wrist_static_transform(rec, self._g2w)
            if T_gw is None:
                return None
            try:
                dT_gw, dq = get_per_robot_corrections(rec)
            except (KeyError, ValueError):
                return None
            return wrist_cam_pose_in_base(
                joint_position_5hz, dq, dT_gw, T_gw, image_rotation_deg=180.0
            )
        return None

# ---------------------------------------------------------------------------
# Metrics: depth, pointmap, RGB
# ---------------------------------------------------------------------------


def compute_depth_metrics(
    pred_m: torch.Tensor,
    gt_m: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> dict[str, np.ndarray]:
    """Standard mono-depth metrics. Both tensors shape (V, T, H, W) metres.

    Returns per-(V, T) arrays for: abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3.
    """
    pred = pred_m.float().clamp(min=1e-3)
    gt = gt_m.float().clamp(min=1e-3)
    if valid_mask is None:
        valid_mask = torch.isfinite(gt) & (gt > 0)
    valid = valid_mask.to(torch.bool)
    V, Ts = pred.shape[:2]
    metrics = {
        "abs_rel": np.full((V, Ts), np.nan, dtype=np.float32),
        "sq_rel": np.full((V, Ts), np.nan, dtype=np.float32),
        "rmse": np.full((V, Ts), np.nan, dtype=np.float32),
        "rmse_log": np.full((V, Ts), np.nan, dtype=np.float32),
        "d1": np.full((V, Ts), np.nan, dtype=np.float32),
        "d2": np.full((V, Ts), np.nan, dtype=np.float32),
        "d3": np.full((V, Ts), np.nan, dtype=np.float32),
    }
    for v in range(V):
        for t in range(Ts):
            m = valid[v, t]
            if not m.any():
                continue
            p = pred[v, t][m]
            g = gt[v, t][m]
            diff = p - g
            ratio = torch.maximum(p / g, g / p)
            metrics["abs_rel"][v, t] = float((diff.abs() / g).mean().item())
            metrics["sq_rel"][v, t] = float(((diff * diff) / g).mean().item())
            metrics["rmse"][v, t] = float(torch.sqrt((diff * diff).mean()).item())
            metrics["rmse_log"][v, t] = float(
                torch.sqrt(((p.log() - g.log()) ** 2).mean()).item()
            )
            metrics["d1"][v, t] = float((ratio < 1.25).float().mean().item())
            metrics["d2"][v, t] = float((ratio < 1.25 ** 2).float().mean().item())
            metrics["d3"][v, t] = float((ratio < 1.25 ** 3).float().mean().item())
    return metrics


def compute_pointmap_metrics(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> dict[str, np.ndarray]:
    """Pointmap metrics. Tensors shape (V, T, 3, H, W).

    Returns per-(V, T) arrays: pm_l1 (metres), pm_log_l1 (MA log space).
    """
    V, Ts = pred_xyz.shape[:2]
    if valid_mask is None:
        valid_mask = torch.isfinite(gt_xyz).all(dim=2)
    valid = valid_mask.to(torch.bool)

    def _f_log(x: torch.Tensor) -> torch.Tensor:
        d = x.norm(dim=2, keepdim=True)
        return x / d.clamp(min=1e-8) * torch.log1p(d)

    pred_l = _f_log(pred_xyz.float())
    gt_l = _f_log(gt_xyz.float())

    out = {
        "pm_l1": np.full((V, Ts), np.nan, dtype=np.float32),
        "pm_log_l1": np.full((V, Ts), np.nan, dtype=np.float32),
    }
    diff = (pred_xyz - gt_xyz).abs().sum(dim=2).float()         # (V, T, H, W)
    diff_l = (pred_l - gt_l).abs().sum(dim=2).float()
    for v in range(V):
        for t in range(Ts):
            m = valid[v, t]
            if not m.any():
                continue
            out["pm_l1"][v, t] = float(diff[v, t][m].mean().item())
            out["pm_log_l1"][v, t] = float(diff_l[v, t][m].mean().item())
    return out


def compute_rgb_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    lpips_model,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """LPIPS / PSNR / SSIM. pred, gt shape (V, F, H, W, 3) uint8.
    Returns per-(V, F) arrays.
    """
    import lpips as _lpips_lib  # only to enforce model type; unused at call site
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    V, Fr = pred.shape[:2]
    out = {
        "lpips": np.zeros((V, Fr), dtype=np.float32),
        "psnr": np.zeros((V, Fr), dtype=np.float32),
        "ssim": np.zeros((V, Fr), dtype=np.float32),
    }
    pred_t = torch.from_numpy(pred).permute(0, 1, 4, 2, 3).float() / 127.5 - 1.0
    gt_t = torch.from_numpy(gt).permute(0, 1, 4, 2, 3).float() / 127.5 - 1.0
    for v in range(V):
        with torch.no_grad():
            lp = lpips_model(pred_t[v].to(device), gt_t[v].to(device)).squeeze()
        out["lpips"][v] = lp.detach().cpu().float().numpy()
        for f in range(Fr):
            out["psnr"][v, f] = psnr_fn(gt[v, f], pred[v, f], data_range=255)
            out["ssim"][v, f] = ssim_fn(
                gt[v, f], pred[v, f], data_range=255, channel_axis=2
            )
    return out


def compute_depth_psnr_ssim(
    pred_dec_u8: np.ndarray,
    gt_dec_u8: np.ndarray,
) -> dict[str, np.ndarray]:
    """PSNR/SSIM on decoded depth-as-grayscale images. (LPIPS skipped — see
    the explicit non-goal in the plan; LPIPS on depth isn't meaningful.)

    pred_dec_u8, gt_dec_u8: (V, F, H, W) uint8 (single channel).
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    V, Fr = pred_dec_u8.shape[:2]
    out = {
        "depth_psnr": np.zeros((V, Fr), dtype=np.float32),
        "depth_ssim": np.zeros((V, Fr), dtype=np.float32),
    }
    for v in range(V):
        for f in range(Fr):
            out["depth_psnr"][v, f] = psnr_fn(
                gt_dec_u8[v, f], pred_dec_u8[v, f], data_range=255
            )
            out["depth_ssim"][v, f] = ssim_fn(
                gt_dec_u8[v, f], pred_dec_u8[v, f], data_range=255
            )
    return out


# ---------------------------------------------------------------------------
# Pointmap decoder readout (cam+pm checkpoints only)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_pointmap_decoder(
    model,
    pred_lat_rgb: torch.Tensor,
    pred_lat_dep: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Run the attached PointMapDecoder on (pred_rgb_latent, pred_depth_latent)
    pair. Returns (V, T, 3, H_pm, W_pm) world-frame XYZ in metres, or None
    if no decoder is attached.

    Mirrors the layout in `CrtlWorldPointmap._maybe_add_pointmap_loss`.
    """
    decoder = getattr(model, "pointmap_decoder", None)
    if decoder is None:
        return None
    if getattr(decoder, "requires_unet_features", False):
        # This decoder is train-time only in the current eval path: it needs
        # intermediate U-Net activations from the denoising forward, while the
        # rollout/chunk metric code only has decoded x0 latents here. Keep
        # unprojection depth metrics working and skip decoder-direct metrics.
        return None
    B, Fr, C, VH, W_lat = pred_lat_rgb.shape
    V = 3
    H_lat = VH // V
    x0_rgb_v = pred_lat_rgb.view(B, Fr, C, V, H_lat, W_lat).permute(0, 3, 1, 2, 4, 5).contiguous()
    x0_dep_v = pred_lat_dep.view(B, Fr, C, V, H_lat, W_lat).permute(0, 3, 1, 2, 4, 5).contiguous()
    dec_in = torch.cat([x0_rgb_v, x0_dep_v], dim=3).view(B * V, Fr, 8, H_lat, W_lat)
    dec_out = decoder(dec_in)                              # (B*V, T, 4, H_pm, W_pm)
    H_pm, W_pm = dec_out.shape[-2:]
    dec_out = dec_out.view(B, V, Fr, 4, H_pm, W_pm)
    xyz = dec_out[:, :, :, :3]                             # (B, V, T, 3, H_pm, W_pm)
    # Eval uses B=1; drop batch dim for caller convenience.
    return xyz[0]                                           # (V, T, 3, H_pm, W_pm)


# ---------------------------------------------------------------------------
# Video saving (shared between chunk and rollout)
# ---------------------------------------------------------------------------


def save_per_sample_videos(
    out_dir: str,
    sample_id: int,
    pred_rgb_vfhwc: np.ndarray,
    gt_rgb_vfhwc: np.ndarray,
    pred_dep_vfh: Optional[np.ndarray],
    gt_dep_vfh: Optional[np.ndarray],
    fps: int = 5,
) -> None:
    """Save side-by-side comparison videos.

    Args shapes:
        pred_rgb_vfhwc, gt_rgb_vfhwc : (V, F, H, W, 3) uint8.
        pred_dep_vfh, gt_dep_vfh     : (V, F, H, W) uint8 grayscale (or None).
    """
    import mediapy
    os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
    V, Fr, H, W = pred_rgb_vfhwc.shape[:4]
    tiles = []
    for v in range(V):
        row = np.concatenate([gt_rgb_vfhwc[v], pred_rgb_vfhwc[v]], axis=2)
        if pred_dep_vfh is not None and gt_dep_vfh is not None:
            # Stack depth as 3-channel rows below the RGB.
            d_pred_rgb = np.stack([pred_dep_vfh[v]] * 3, axis=-1)
            d_gt_rgb = np.stack([gt_dep_vfh[v]] * 3, axis=-1)
            d_row = np.concatenate([d_gt_rgb, d_pred_rgb], axis=2)
            row = np.concatenate([row, d_row], axis=1)
        tiles.append(row)
    combined = np.concatenate(tiles, axis=1)                   # stack views vertically
    mediapy.write_video(
        os.path.join(out_dir, "videos", f"sample_{sample_id:04d}.mp4"),
        combined, fps=fps,
    )


def depth_to_uint8_for_video(depth_m: torch.Tensor) -> np.ndarray:
    """(V, F, H, W) metric depth -> (V, F, H, W) uint8 grayscale for video.

    Maps per-view log-depth to [0, 255] with RANGES_M — matches the encoder
    side so visualizations are directly comparable.
    """
    V, Fr, H, W = depth_m.shape
    out = np.zeros((V, Fr, H, W), dtype=np.uint8)
    for v_idx, role in enumerate(VIEW_ORDER):
        d_min, d_max = RANGES_M[role]
        d = depth_m[v_idx].float().clamp(d_min, d_max)
        log_min = math.log(d_min)
        log_max = math.log(d_max)
        g = (d.log() - log_min) / (log_max - log_min)         # [0, 1] close=0
        g = (1.0 - g) * 255.0
        out[v_idx] = g.clamp(0, 255).to(torch.uint8).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


def write_metrics_csv(
    out_csv: str,
    rows: list[dict],
) -> None:
    """Write a list of metric-row dicts to CSV. Keys become columns; missing
    cells are blank. Existence of out_csv's parent is assumed."""
    import csv
    if not rows:
        print(f"[eval_horiz] no rows to write to {out_csv}", flush=True)
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[eval_horiz] wrote {len(rows)} rows -> {out_csv}", flush=True)
