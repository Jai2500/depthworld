# CrtlWorld (single-UNet horiz architecture) + auxiliary world-frame
# pointmap loss through a latent-fed DPT head.
#
# Pure subclass of `CrtlWorld` — bit-exact equivalent to the parent when
# `pointmap_decoder is None`. The horiz latent layout is
# `(T, 4, V*H_lat, 2*W_lat)`: RGB | depth packed side-by-side along width,
# views stacked along height.
#
# Required batch keys for the aux loss (emitted by
# `Dataset_mix_horiz_with_pointmap`):
#   batch["pointmap_gt"]    (B, V, T, 3, H_pm, W_pm)  world-frame XYZ (m)
#   batch["pointmap_valid"] (B, V, T, H_pm, W_pm)
# When missing, the forward falls through to the plain denoising loss.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.ctrl_world import CrtlWorld


class CrtlWorldPointmap(CrtlWorld):
    """Drop-in replacement for `CrtlWorld` with an optional pointmap aux head."""

    def __init__(self, args, num_history=None):
        super().__init__(args, num_history=num_history)
        # Optional auxiliary world-frame point-map decoder (latent-fed DPT).
        self.pointmap_decoder: Optional[nn.Module] = None
        self.pointmap_loss_weight: float = 0.0
        self.pointmap_drop_top_pct: float = 0.0
        self.pointmap_loss_in_log: bool = True
        self.pointmap_conf_alpha: float = 0.2
        self.pointmap_robust_c: float = 0.05
        self.pointmap_robust_alpha: float = 0.5
        self.pointmap_min_snr_gamma: float = 5.0

    # -----------------------------------------------------------------------
    # Pointmap decoder attachment
    # -----------------------------------------------------------------------

    def attach_pointmap_decoder(
        self,
        decoder: nn.Module,
        loss_weight: float = 0.05,
        train_decoder: bool = True,
        drop_top_pct: float = 0.05,
        loss_in_log: bool = True,
        min_snr_gamma: float = 5.0,
        conf_alpha: float = 0.2,
        robust_c: float = 0.05,
        robust_alpha: float = 0.5,
    ) -> None:
        self.pointmap_decoder = decoder
        self.pointmap_loss_weight = float(loss_weight)
        self.pointmap_drop_top_pct = float(drop_top_pct)
        self.pointmap_loss_in_log = bool(loss_in_log)
        self.pointmap_min_snr_gamma = float(min_snr_gamma)
        self.pointmap_conf_alpha = float(conf_alpha)
        self.pointmap_robust_c = float(robust_c)
        self.pointmap_robust_alpha = float(robust_alpha)
        for p in decoder.parameters():
            p.requires_grad_(train_decoder)

    # -----------------------------------------------------------------------
    # Pointmap aux loss — split horiz latent into per-view RGB|depth halves
    # then feed the decoder. MA-style criterion: f_log compression + Barron
    # robust vector loss + exp confidence + per-(B,V) top-K drop +
    # Min-SNR-gamma. No avg-distance normalization: DROID pointmaps are
    # metric and clipped to a fixed depth range.
    # -----------------------------------------------------------------------

    def _maybe_add_aux_losses(
        self, *, predict_x0, latents, num_history, loss_weight, batch,
    ):
        return self._maybe_add_pointmap_loss(
            predict_x0=predict_x0,
            latents=latents,
            num_history=num_history,
            loss_weight=loss_weight,
            batch=batch,
        )

    def _maybe_add_pointmap_loss(
        self, *, predict_x0, latents, num_history, loss_weight, batch,
    ):
        if self.pointmap_decoder is None or self.pointmap_loss_weight <= 0.0:
            return None
        if "pointmap_gt" not in batch or "pointmap_valid" not in batch:
            return None

        device = predict_x0.device
        # Slice future frames. predict_x0 shape: (B, T, 4, V*H_lat, 2*W_lat).
        x0_f = predict_x0[:, num_history:]                # (B, T_f, 4, V*H, 2*W)
        B, T_f, C, VH, W2 = x0_f.shape
        # Derive V from the latent shape. Per-view latent height = args.height // 8.
        H_lat_per_view = int(getattr(self.args, "height", 192)) // 8
        V = VH // H_lat_per_view
        assert C == 4 and VH % V == 0 and W2 % 2 == 0, (
            f"unexpected horiz layout: predict_x0 shape={tuple(x0_f.shape)}"
        )
        H_lat = VH // V
        W_lat = W2 // 2

        # Width-split: left = RGB latent, right = depth latent. Each is
        # (B, T_f, 4, V*H_lat, W_lat).
        rgb_half = x0_f[:, :, :, :, :W_lat]
        dep_half = x0_f[:, :, :, :, W_lat:]

        # Unpack views from the height-stacked layout, then per-view cat
        # into (B*V, T_f, 8, H_lat, W_lat).
        x0_rgb_v = rgb_half.view(B, T_f, C, V, H_lat, W_lat).permute(0, 3, 1, 2, 4, 5).contiguous()
        x0_dep_v = dep_half.view(B, T_f, C, V, H_lat, W_lat).permute(0, 3, 1, 2, 4, 5).contiguous()
        dec_in = torch.cat([x0_rgb_v, x0_dep_v], dim=3)     # (B, V, T_f, 8, H, W)
        dec_in = dec_in.view(B * V, T_f, 8, H_lat, W_lat)

        dec_out = self.pointmap_decoder(dec_in)             # (B*V, T_f, 4, H_pm, W_pm)
        H_pm, W_pm = dec_out.shape[-2:]
        dec_out = dec_out.view(B, V, T_f, 4, H_pm, W_pm)
        xyz_pred = dec_out[:, :, :, :3]
        conf_logit = dec_out[:, :, :, 3]

        pm_gt = batch["pointmap_gt"].to(device=device, dtype=dec_out.dtype)
        pm_valid = batch["pointmap_valid"].to(device=device, dtype=dec_out.dtype)
        pm_gt_f = pm_gt[:, :, num_history:]                 # (B, V, T_f, 3, H_pm, W_pm)
        pm_valid_f = pm_valid[:, :, num_history:]           # (B, V, T_f, H_pm, W_pm)
        assert pm_gt_f.shape == xyz_pred.shape, (
            f"pointmap_gt shape {tuple(pm_gt_f.shape)} != pred {tuple(xyz_pred.shape)}"
        )

        # MapAnything-style pointmap loss. It matches the important MA pieces
        # for our decoder target:
        #   - optionally f_log-compress XYZ magnitude
        #   - Barron robust loss on the XYZ vector norm
        #   - C = 1 + exp(raw_conf), vmin=1 confidence
        #   - per-(B,V) top-K outlier drop
        def _f_log(x, dim):
            d = x.norm(dim=dim, keepdim=True)
            unit = x / d.clamp(min=1e-8)
            return unit * torch.log1p(d)

        def _avg_dis_stats(gt, valid):
            valid_f = valid.to(dtype=torch.float32)
            gt_dist = gt.float().norm(dim=3)  # (B, V, T_f, H, W)
            valid_count = valid_f.reshape(B, -1).sum(dim=1)
            dist_sum = (gt_dist * valid_f).reshape(B, -1).sum(dim=1)
            return torch.where(
                valid_count > 0,
                dist_sum / valid_count.clamp(min=1.0),
                torch.ones_like(dist_sum),
            ).clamp(min=1e-6)

        # Diagnostic only — mean GT distance per sample (not used in the loss).
        avg_dis = _avg_dis_stats(pm_gt_f, pm_valid_f.to(torch.bool))

        if self.pointmap_loss_in_log:
            xyz_pred_l = _f_log(xyz_pred, dim=3)
            pm_gt_l = _f_log(pm_gt_f, dim=3)
        else:
            xyz_pred_l = xyz_pred
            pm_gt_l = pm_gt_f

        c_robust = self.pointmap_robust_c
        a_robust = self.pointmap_robust_alpha
        diff = xyz_pred_l.float() - pm_gt_l.float()
        raw_l1_xyz = diff.abs().sum(dim=3)
        error_scaled = torch.sum((diff / c_robust) ** 2, dim=3)
        if abs(a_robust) < 1e-8:
            l1_xyz = torch.log1p(0.5 * error_scaled)
        elif abs(a_robust - 2.0) < 1e-8:
            l1_xyz = 0.5 * error_scaled
        else:
            beta = abs(a_robust - 2.0)
            l1_xyz = (beta / a_robust) * (
                torch.pow(error_scaled / beta + 1.0, a_robust / 2.0) - 1.0
            )
        conf_alpha = self.pointmap_conf_alpha
        C_conf = 1.0 + torch.exp(conf_logit.float())
        log_C = torch.log(C_conf)
        kg = C_conf * l1_xyz - conf_alpha * log_C

        drop_top_pct = self.pointmap_drop_top_pct
        if drop_top_pct > 0.0:
            with torch.no_grad():
                l1_flat = l1_xyz.reshape(B, V, -1).float()
                valid_flat = pm_valid_f.reshape(B, V, -1).to(torch.bool)
                l1_for_quant = l1_flat.masked_fill(~valid_flat, float("inf"))
                q = 1.0 - drop_top_pct
                thr_bv = torch.quantile(l1_for_quant, q, dim=-1, interpolation="lower")
                thr = thr_bv.view(B, V, 1, 1, 1)
                pm_valid_f = pm_valid_f * (l1_xyz <= thr).to(pm_valid_f.dtype)

        gamma = self.pointmap_min_snr_gamma
        if gamma > 0.0:
            sigma_w = loss_weight.clamp(max=gamma + 1.0)
        else:
            sigma_w = loss_weight
        sigma_w = sigma_w.view(B, 1, 1, 1, 1).to(dtype=kg.dtype)
        kg_w = kg * sigma_w * pm_valid_f
        denom = pm_valid_f.sum().clamp(min=1.0)
        loss_pointmap = kg_w.sum() / denom

        with torch.no_grad():
            l1_mean = (l1_xyz * pm_valid_f).sum() / denom
            raw_l1_mean = (raw_l1_xyz * pm_valid_f).sum() / denom
            logc_mean = (log_C * pm_valid_f).sum() / denom
            kg_mean = (kg * pm_valid_f).sum() / denom
            valid_frac = pm_valid_f.mean()
            extra_log = {
                "loss_pointmap": loss_pointmap.detach(),
                "loss_pointmap_l1_mean": l1_mean,
                "loss_pointmap_robust_mean": l1_mean,
                "loss_pointmap_raw_l1_mean": raw_l1_mean,
                "loss_pointmap_logconf_mean": logc_mean,
                "loss_pointmap_kg_mean": kg_mean,
                "loss_pointmap_valid_frac": valid_frac,
                "loss_pointmap_sigma_w_mean": sigma_w.detach().mean(),
                "loss_pointmap_sigma_w_max": sigma_w.detach().max(),
                "loss_pointmap_avg_dis_mean": avg_dis.mean(),
            }

        extra_loss = self.pointmap_loss_weight * loss_pointmap
        return extra_loss, extra_log
