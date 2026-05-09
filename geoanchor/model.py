from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    porosity: torch.Tensor
    aux: dict[str, torch.Tensor]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 6):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.bottleneck = ConvBlock(base * 2, base * 4)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec1 = ConvBlock(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec2 = ConvBlock(base * 2, base)
        self.outc = nn.Conv2d(base, 1, 1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.enc1(x)
        x2 = self.enc2(F.avg_pool2d(x1, 2))
        xb = self.bottleneck(F.avg_pool2d(x2, 2))
        return x1, x2, xb

    def decode(self, x1: torch.Tensor, x2: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        x = self.up1(xb)
        if x.shape[-2:] != x2.shape[-2:]:
            x = F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x2, x], dim=1))
        x = self.up2(x)
        if x.shape[-2:] != x1.shape[-2:]:
            x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x1, x], dim=1))
        return self.outc(x)


class TrendBackbone(nn.Module):
    def __init__(self, structure_ch: int = 4):
        super().__init__()
        self.net = TinyUNet(1 + structure_ch, base=4)

    def forward(self, seismic: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
        x = torch.cat([seismic, structure], dim=1)
        x1, x2, xb = self.net.encode(x)
        return torch.sigmoid(self.net.decode(x1, x2, xb)) * 0.34


class GeoAnchorNet(nn.Module):
    def __init__(
        self,
        structure_ch: int = 4,
        n_components: int = 4,
        prior_blend_alpha: float = 0.70,
        use_prior_condition: bool = True,
        use_external_prior: bool = True,
        use_anchor_curve: bool = True,
        use_gain: bool = True,
        use_exact_anchor: bool = True,
        anchor_band_strength: float = 1.0,
        curve_gate_strength: float = 1.0,
        curve_gate_mode: str = "column",
    ):
        super().__init__()
        self.n_components = n_components
        self.prior_blend_alpha = prior_blend_alpha
        self.use_prior_condition = use_prior_condition
        self.use_external_prior = use_external_prior
        self.use_anchor_curve = use_anchor_curve
        self.use_gain = use_gain
        self.use_exact_anchor = use_exact_anchor
        self.anchor_band_strength = anchor_band_strength
        self.curve_gate_strength = curve_gate_strength
        self.curve_gate_mode = curve_gate_mode

        in_ch = 1 + structure_ch + 1 + 2 + (1 if use_prior_condition else 0)
        self.trend_backbone = TrendBackbone(structure_ch=structure_ch)
        self.encoder = TinyUNet(in_ch=in_ch, base=6)
        self.router_feat = nn.Sequential(
            nn.Conv2d(in_ch, 12, 1),
            nn.SiLU(),
            nn.Conv2d(12, 12, 3, padding=1),
            nn.SiLU(),
        )
        self.router_head = nn.Conv2d(12, n_components, 1)
        self.gain_head = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(12, 1, 1),
            nn.Sigmoid(),
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(structure_ch, 8, 1),
            nn.SiLU(),
            nn.Conv2d(8, 2, 1),
            nn.Sigmoid(),
        )
        vertical = torch.tensor([[1.0], [4.0], [7.0], [10.0], [12.0], [10.0], [7.0], [4.0], [1.0]], dtype=torch.float32)
        horizontal = torch.tensor([[1.0, 2.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
        self.register_buffer("anchor_curve_kernel", (vertical / vertical.sum()).view(1, 1, 9, 1))
        self.register_buffer("anchor_band_kernel", (horizontal / horizontal.sum()).view(1, 1, 1, 5))

    def graph_filter(self, x: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
        edges = self.edge_head(structure)
        wx = 1.0 - 0.75 * edges[:, 0:1]
        wz = 1.0 - 0.90 * edges[:, 1:2]
        left = F.pad(x[:, :, :, :-1], (1, 0, 0, 0), mode="replicate")
        right = F.pad(x[:, :, :, 1:], (0, 1, 0, 0), mode="replicate")
        up = F.pad(x[:, :, :-1, :], (0, 0, 1, 0), mode="replicate")
        down = F.pad(x[:, :, 1:, :], (0, 0, 0, 1), mode="replicate")
        return (x + wx * left + wx * right + wz * up + wz * down) / (1.0 + 2.0 * wx + 2.0 * wz)

    def build_components(self, raw: torch.Tensor, structure: torch.Tensor, anchor_mask: torch.Tensor) -> torch.Tensor:
        smooth = self.graph_filter(0.030 * torch.tanh(raw), structure)
        boundary = torch.clamp(structure[:, 2:3] + structure[:, 3:4], 0.0, 1.0)
        detail = 0.020 * torch.tanh(raw) * (0.35 + 1.65 * boundary)
        well_local = 0.015 * torch.tanh(raw) * (0.20 + 1.80 * anchor_mask)
        return torch.cat([smooth, detail, well_local], dim=1)

    def propagate_anchor_curve(self, anchor: torch.Tensor, anchor_mask: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        if anchor.device.type != "cpu":
            num = F.conv2d(anchor * anchor_mask, self.anchor_curve_kernel, padding=(4, 0))
            den = F.conv2d(anchor_mask, self.anchor_curve_kernel, padding=(4, 0))
            curve = torch.where(den > 1e-4, num / (den + 1e-6), trend)
        else:
            anchor_np = anchor.detach().cpu().numpy()
            mask_np = anchor_mask.detach().cpu().numpy() > 0.5
            trend_np = trend.detach().cpu().numpy()
            curve_np = trend_np.copy()
            depth = np.arange(anchor_np.shape[2], dtype=np.float32)
            for b in range(anchor_np.shape[0]):
                cols = np.unique(np.where(mask_np[b, 0])[1])
                for col in cols:
                    rows = np.where(mask_np[b, 0, :, col])[0]
                    if rows.size == 1:
                        curve_np[b, 0, :, col] = anchor_np[b, 0, rows[0], col]
                    elif rows.size > 1:
                        curve_np[b, 0, :, col] = np.interp(depth, rows.astype(np.float32), anchor_np[b, 0, rows, col].astype(np.float32))
            curve = torch.from_numpy(np.clip(curve_np, 0.0, 0.5)).to(anchor.device)

        col_support = (anchor_mask.sum(dim=2, keepdim=True) > 0.0).float().expand_as(anchor_mask)
        band_num = F.conv2d(curve * col_support, self.anchor_band_kernel, padding=(0, 2))
        band_den = F.conv2d(col_support, self.anchor_band_kernel, padding=(0, 2))
        spread_curve = torch.where(band_den > 1e-4, band_num / (band_den + 1e-6), trend)
        mixed_curve = torch.clamp((1.0 - self.anchor_band_strength) * trend + self.anchor_band_strength * spread_curve, 0.0, 0.5)
        return torch.where(col_support > 0.5, curve, mixed_curve).clamp(0.0, 0.5)

    def build_curve_gate(self, anchor_mask: torch.Tensor) -> torch.Tensor:
        if self.curve_gate_mode == "column":
            col_support = (anchor_mask.sum(dim=2, keepdim=True) > 0.0).float().expand_as(anchor_mask)
            band_support = F.conv2d(col_support, self.anchor_band_kernel, padding=(0, 2))
            return torch.maximum(col_support, torch.clamp(self.curve_gate_strength * band_support, 0.0, 1.0))
        if self.curve_gate_mode == "soft_column":
            col_support = (anchor_mask.sum(dim=2, keepdim=True) > 0.0).float().expand_as(anchor_mask)
            band_support = F.conv2d(col_support, self.anchor_band_kernel, padding=(0, 2))
            return torch.maximum(anchor_mask, torch.clamp(self.curve_gate_strength * band_support, 0.0, 1.0))
        if self.curve_gate_mode == "local":
            vertical = F.conv2d(anchor_mask, self.anchor_curve_kernel, padding=(4, 0))
            local = F.conv2d(vertical, self.anchor_band_kernel, padding=(0, 2))
            return torch.maximum(anchor_mask, torch.clamp(self.curve_gate_strength * local, 0.0, 1.0))
        if self.curve_gate_mode == "none":
            return anchor_mask
        raise ValueError(f"unsupported curve_gate_mode: {self.curve_gate_mode}")

    def forward(
        self,
        seismic: torch.Tensor,
        structure: torch.Tensor,
        prior: torch.Tensor | None = None,
        anchor: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        learned_trend = self.trend_backbone(seismic, structure)
        if prior is None or not self.use_external_prior:
            prior = learned_trend
        if anchor is None:
            anchor = torch.zeros_like(prior)
        if anchor_mask is None:
            anchor_mask = torch.zeros_like(prior)

        alpha = float(self.prior_blend_alpha)
        base_trend = torch.clamp(alpha * prior + (1.0 - alpha) * learned_trend, 0.0, 0.5)
        trend = torch.where(anchor_mask > 0.5, anchor, base_trend)
        model_inputs = [seismic, structure, trend, anchor, anchor_mask]
        if self.use_prior_condition:
            model_inputs.append(prior)
        z = torch.cat(model_inputs, dim=1)

        x1, x2, xb = self.encoder.encode(z)
        raw = self.encoder.decode(x1, x2, xb)
        components = self.build_components(raw, structure, anchor_mask)
        anchor_curve = self.propagate_anchor_curve(anchor, anchor_mask, trend)
        curve_component = anchor_curve - trend
        components = torch.cat([components, curve_component], dim=1)

        feat = self.router_feat(z)
        logits = self.router_head(feat)
        weights = torch.softmax(logits, dim=1)
        if weights.shape[1] != components.shape[1]:
            raise RuntimeError("component and weight counts do not match")
        residual = (weights * components).sum(dim=1, keepdim=True)
        gain = self.gain_head(feat) if self.use_gain else torch.ones_like(residual)
        base_pred = torch.clamp(trend + gain * residual, 0.0, 0.5)

        if self.use_anchor_curve:
            curve_gate = self.build_curve_gate(anchor_mask)
            pred = torch.clamp((1.0 - curve_gate) * base_pred + curve_gate * anchor_curve, 0.0, 0.5)
        else:
            pred = base_pred
        if self.use_exact_anchor:
            pred = torch.where(anchor_mask > 0.5, anchor, pred)

        return ModelOutput(
            porosity=pred,
            aux={
                "learned_trend": learned_trend,
                "base_trend": base_trend,
                "trend": trend,
                "residual": residual,
                "gain": gain,
                "weights": weights,
                "anchor_curve": anchor_curve,
                "anchor_mask": anchor_mask,
            },
        )


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
