import torch
from torch import nn


def _kaiming_conv(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)
    return conv


class _ReassembleStage(nn.Module):
    """Reads a backbone feature stage out against a class embedding, then reshapes
    it back to a feature map and resizes it to the stage's target resolution."""

    def __init__(self, in_channels: int, out_channels: int, resize: nn.Module, use_batchnorm: bool) -> None:
        super().__init__()
        self.readout_project = nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
        self.norm = nn.SyncBatchNorm(in_channels) if use_batchnorm else nn.Identity()
        self.project = nn.Sequential(_kaiming_conv(nn.Conv2d(in_channels, out_channels, kernel_size=1)), nn.ReLU())
        self.resize = resize

    def forward(self, x: torch.Tensor, target_class_emb: torch.Tensor) -> torch.Tensor:
        feature_shape = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        readout = target_class_emb.unsqueeze(1).expand_as(x)
        x = self.readout_project(torch.cat((x, readout), dim=-1))
        x = x.permute(0, 2, 1).reshape(feature_shape)
        x = self.norm(x)
        x = self.project(x)
        return self.resize(x)


class _PreActResidualConvUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = _kaiming_conv(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
        self.conv2 = _kaiming_conv(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.act(x))
        x = self.conv2(self.act(x))
        return x + residual


class _FeatureFusionBlock(nn.Module):
    def __init__(self, channels: int, with_res_conv_unit1: bool) -> None:
        super().__init__()
        self.res_conv_unit1 = _PreActResidualConvUnit(channels) if with_res_conv_unit1 else None
        self.res_conv_unit2 = _PreActResidualConvUnit(channels)
        self.project = nn.Sequential(_kaiming_conv(nn.Conv2d(channels, channels, kernel_size=1)), nn.ReLU())

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            if skip.shape != x.shape:
                skip = nn.functional.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = x + self.res_conv_unit1(skip)
        x = self.res_conv_unit2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.project(x)


class _UpConvHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DPTHead(nn.Module):
    """DPT decoder (Ranftl et al., https://arxiv.org/abs/2103.13413) reassembling 4
    backbone feature stages, fusing them top-down, and predicting a single-channel
    map. Each stage is read out against a class embedding."""

    def __init__(
        self,
        in_channels: int,
        stage_channels: int,
        fusion_channels: int,
        out_channels: int,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        resize_layers = [
            nn.ConvTranspose2d(stage_channels, stage_channels, kernel_size=4, stride=4),
            nn.ConvTranspose2d(stage_channels, stage_channels, kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(stage_channels, stage_channels, kernel_size=3, stride=2, padding=1),
        ]
        self.reassemble_stages = nn.ModuleList([
            _ReassembleStage(in_channels, stage_channels, resize, use_batchnorm) for resize in resize_layers
        ])
        self.stage_convs = nn.ModuleList([
            nn.Sequential(
                _kaiming_conv(nn.Conv2d(stage_channels, fusion_channels, kernel_size=3, padding=1, bias=False)),
                nn.ReLU(),
            )
            for _ in range(4)
        ])
        self.fusion_blocks = nn.ModuleList([
            _FeatureFusionBlock(fusion_channels, with_res_conv_unit1=(i > 0)) for i in range(4)
        ])
        self.project = nn.Sequential(
            _kaiming_conv(nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1)), nn.ReLU()
        )
        self.head = _UpConvHead(fusion_channels, out_channels)

    def forward(self, features: list[tuple[torch.Tensor, torch.Tensor]], target_class_emb: torch.Tensor) -> torch.Tensor:
        stages = [
            self.stage_convs[i](self.reassemble_stages[i](patch_feats, target_class_emb))
            for i, (patch_feats, _vit_cls_token) in enumerate(features)
        ]

        out = self.fusion_blocks[0](stages[-1])
        for i in range(1, 4):
            out = self.fusion_blocks[i](out, stages[-(i + 1)])

        out = self.project(out)
        return self.head(out)
