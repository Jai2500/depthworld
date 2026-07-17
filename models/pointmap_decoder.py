# DPT-style decoder mapping per-view (RGB + depth) SVD latents to per-pixel
# point maps (XYZ + raw_conf) in the robot-base world frame.
#
# `DPTPointMapDecoder` consumes the model's predicted x0 latents
# (`--pointmap_feature_source=latent` path): the horiz latent is split into
# per-view RGB|depth halves, channel-concatenated to 8 channels, and decoded
# to full per-view image resolution.
#
# Forward expects (B, T, in_ch, H_lat, W_lat) where H_lat=24, W_lat=40 for
# the standard CrtlWorld latent. Output is (B, T, out_ch, H_lat*8, W_lat*8)
# = (B, T, 4, 192, 320).
#
# Warm-start: `from_vggt(...)` copies VGGT's direct (XYZ, raw_conf) point
# head (`point_head.scratch.*`) into the DPT trunk. The latent
# stem/downsampling stages always stay at random init — VGGT consumes ViT
# tokens, not 8-channel SVD latents.

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Warm-start checkpoint location. Override via env var or the
# --pointmap_vggt_ckpt CLI flag; the fallback is repo-relative (see
# checkpoints/README.md).
DEFAULT_VGGT_CKPT = Path(
    os.environ.get(
        "CTRLWORLD_VGGT_CKPT",
        "checkpoints/vggt/model.pt",
    )
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _Clamp(nn.Module):
    """tanh(x/3)*3 — saturating input clamp."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3.0) * 3.0


class _ConvGNAct(nn.Module):
    """Small conv block used by the DPT-style pointmap head."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 8,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(num_groups=min(groups, n_out), num_channels=n_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ResidualConvUnit(nn.Module):
    """DPT/RefineNet-style residual conv unit."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = _ConvGNAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class _FeatureFusionBlock(nn.Module):
    """Fuse a coarse DPT feature with a same-resolution skip feature."""

    def __init__(self, channels: int):
        super().__init__()
        self.skip_unit = _ResidualConvUnit(channels)
        self.out_unit = _ResidualConvUnit(channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.skip_unit(skip)
        return self.out_unit(x)


class _VGGTResidualConvUnit(nn.Module):
    """Residual conv unit with VGGT DPT-compatible key names."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class _VGGTFeatureFusionBlock(nn.Module):
    """Feature fusion block matching VGGT DPT scratch keys."""

    def __init__(self, features: int, *, has_residual: bool = True):
        super().__init__()
        self.has_residual = bool(has_residual)
        if has_residual:
            self.resConfUnit1 = _VGGTResidualConvUnit(features)
        self.resConfUnit2 = _VGGTResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1, bias=True)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None, *, size=None) -> torch.Tensor:
        if self.has_residual and skip is not None:
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        if size is None:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        else:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class DPTPointMapDecoder(nn.Module):
    """DPT-style dense pointmap head for CrtlWorld latents.

      Input:  (B, T, 8, 24, 40)
      Output: (B, T, 4, 192, 320)

    It builds a four-scale latent pyramid, projects all scales to a common
    feature width, fuses coarse-to-fine with RefineNet-style residual blocks,
    then upsamples 8x to full per-view image resolution. Trained from scratch
    unless warm-started via `from_vggt`.
    """

    def __init__(
        self,
        in_ch: int = 8,
        out_ch: int = 4,
        base_ch: int = 64,
        features: int = 256,
        conf_bias_init: float = -3.0,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.base_ch = int(base_ch)
        self.features = int(features)

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8

        self.input_clamp = _Clamp()
        self.stem = nn.Sequential(
            _ConvGNAct(in_ch, c1),
            _ResidualConvUnit(c1),
        )
        self.stage2 = nn.Sequential(
            _ConvGNAct(c1, c2, stride=2),
            _ResidualConvUnit(c2),
        )
        self.stage3 = nn.Sequential(
            _ConvGNAct(c2, c3, stride=2),
            _ResidualConvUnit(c3),
        )
        self.stage4 = nn.Sequential(
            _ConvGNAct(c3, c4, stride=2),
            _ResidualConvUnit(c4),
        )

        self.scratch = nn.Module()
        self.scratch.layer1_rn = nn.Conv2d(c1, features, 3, padding=1, bias=False)
        self.scratch.layer2_rn = nn.Conv2d(c2, features, 3, padding=1, bias=False)
        self.scratch.layer3_rn = nn.Conv2d(c3, features, 3, padding=1, bias=False)
        self.scratch.layer4_rn = nn.Conv2d(c4, features, 3, padding=1, bias=False)
        self.scratch.refinenet1 = _VGGTFeatureFusionBlock(features)
        self.scratch.refinenet2 = _VGGTFeatureFusionBlock(features)
        self.scratch.refinenet3 = _VGGTFeatureFusionBlock(features)
        self.scratch.refinenet4 = _VGGTFeatureFusionBlock(features, has_residual=False)
        # Match VGGT's point_head.scratch tail exactly so the direct
        # XYZ+confidence point-head weights can be copied without shape
        # surgery. VGGT applies inv_log/expp1 after this conv; our loss
        # consumes the raw logits and applies the matching confidence
        # activation itself.
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 1),
        )
        self._init_output(conf_bias_init=conf_bias_init)

    def _final_conv(self) -> nn.Conv2d:
        final = self.scratch.output_conv2[-1]
        assert isinstance(final, nn.Conv2d)
        return final

    def _init_output(self, *, conf_bias_init: float) -> None:
        # Keep initial XYZ predictions near zero and initial confidence near
        # vmin: C = 1 + exp(-3) ~= 1.05. Small nonzero weights let gradients
        # reach the full decoder immediately.
        final = self._final_conv()
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)
        if final.bias.numel() >= 4:
            final.bias.data[3] = float(conf_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5, f"DPTPointMapDecoder expects 5-D (B,T,C,H,W), got {x.shape}"
        B, T, C, H_lat, W_lat = x.shape
        assert C == self.in_ch, (
            f"channel mismatch: input has C={C}, decoder expects in_ch={self.in_ch}"
        )
        x = x.reshape(B * T, C, H_lat, W_lat)
        x = self.input_clamp(x)

        s1 = self.stem(x)      # 24 x 40
        s2 = self.stage2(s1)   # 12 x 20
        s3 = self.stage3(s2)   #  6 x 10
        s4 = self.stage4(s3)   #  3 x  5

        layer1 = self.scratch.layer1_rn(s1)
        layer2 = self.scratch.layer2_rn(s2)
        layer3 = self.scratch.layer3_rn(s3)
        layer4 = self.scratch.layer4_rn(s4)

        out = self.scratch.refinenet4(layer4, size=layer3.shape[-2:])
        out = self.scratch.refinenet3(out, layer3, size=layer2.shape[-2:])
        out = self.scratch.refinenet2(out, layer2, size=layer1.shape[-2:])
        out = self.scratch.refinenet1(out, layer1)
        out = self.scratch.output_conv1(out)
        out = F.interpolate(out, size=(H_lat * 8, W_lat * 8), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        _, out_C, H_out, W_out = out.shape
        return out.reshape(B, T, out_C, H_out, W_out)

    @classmethod
    def from_vggt(
        cls,
        in_ch: int = 8,
        out_ch: int = 4,
        base_ch: int = 64,
        features: int = 256,
        vggt_ckpt_path: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> "DPTPointMapDecoder":
        """Warm-start the DPT fusion trunk and direct point head from VGGT.

        VGGT's public point head is a direct `(XYZ, raw_conf)` DPT head:
        `point_head = DPTHead(output_dim=4, activation="inv_log",
        conf_activation="expp1")`, a direct semantic match for our aux loss.

        We still leave the latent stem and scale adapters at random init
        because VGGT consumes aggregator token features, while CrtlWorld gives
        us 8-channel SVD latents.
        """
        model = cls(
            in_ch=in_ch,
            out_ch=out_ch,
            base_ch=base_ch,
            features=features,
        )
        ckpt_path = Path(vggt_ckpt_path or DEFAULT_VGGT_CKPT)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"VGGT checkpoint not found: {ckpt_path}")
        if features != 256:
            raise ValueError(
                "VGGT DPT warm-start requires --pointmap_dpt_features 256 "
                f"because facebook/VGGT-1B's DPT trunk is 256-wide; got {features}."
            )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src_items = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt

        dst_sd = model.state_dict()
        copied: list[str] = []
        skipped: list[str] = []

        src_prefix = "point_head.scratch."
        dst_prefix = "scratch."
        for src_key, src in src_items.items():
            if not src_key.startswith(src_prefix):
                continue
            dst_key = dst_prefix + src_key[len(src_prefix):]
            if dst_key not in dst_sd:
                skipped.append(f"{src_key} -> {dst_key} (missing dst)")
                continue
            dst = dst_sd[dst_key]
            if dst.shape == src.shape:
                dst.copy_(src)
                copied.append(f"{src_key} -> {dst_key}")
            else:
                skipped.append(
                    f"{src_key} -> {dst_key} "
                    f"(shape dst={tuple(dst.shape)} src={tuple(src.shape)})"
                )

        model.load_state_dict(dst_sd, strict=True)

        if verbose:
            print(
                f"[DPTPointMapDecoder.from_vggt] loaded from {ckpt_path} "
                f"copied={len(copied)} skipped={len(skipped)}",
                flush=True,
            )
            if skipped:
                print(f"  - skipped examples: {skipped[:5]}", flush=True)
        return model
