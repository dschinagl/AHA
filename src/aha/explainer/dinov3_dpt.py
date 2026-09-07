import torch
from torch import nn
from torchvision import transforms as T

from aha.explainer.dpt_head import DPTHead
from aha.explainer.encoder import Dinov3FeatureEncoder

_BACKBONE_OUT_LAYERS = {
    "dinov3_vitl16": (4, 11, 17, 23),
}


class Dinov3DptExplainer(nn.Module):
    """Class-conditional attribution map predictor: frozen DINOv3 backbone,
    DPT decoder and refinement block"""

    def __init__(
        self,
        dinov3_path: str,
        backbone_name: str,
        backbone_weights: str,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()

        backbone = torch.hub.load(dinov3_path, backbone_name, source="local", weights=backbone_weights)
        self.encoder = Dinov3FeatureEncoder(backbone, out_layers=_BACKBONE_OUT_LAYERS[backbone_name])

        embed_dim = self.encoder.embed_dim
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        fusion_channels = 128
        self.decoder = DPTHead(
            in_channels=embed_dim,
            stage_channels=256,
            fusion_channels=fusion_channels,
            out_channels=fusion_channels,
            use_batchnorm=True,
        )
        self.target_class_embedding = nn.Embedding(num_classes, embed_dim)

        mid_channels = max(32, fusion_channels // 2)
        self.dropout = nn.Dropout2d(0.1)
        self.pre_refine_norm = nn.SyncBatchNorm(fusion_channels)
        self.refine_block = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1, groups=fusion_channels, bias=False),
            nn.SyncBatchNorm(fusion_channels),
            nn.GELU(),
            nn.Conv2d(fusion_channels, mid_channels, kernel_size=1, bias=False),
            nn.SyncBatchNorm(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.SyncBatchNorm(mid_channels),
            nn.GELU(),
        )
        for module in self.refine_block.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

        self.final_conv = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1)
        nn.init.normal_(self.final_conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.final_conv.bias, 0)

    def forward(self, x: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        target_class_emb = self.target_class_embedding(target_idx.long())
        x = self.normalize(x)

        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            features = self.encoder(x)
        features = [
            (patch.float().contiguous(), vit_cls_token.float().contiguous()) for patch, vit_cls_token in features
        ]

        out = self.decoder(features, target_class_emb).contiguous()
        out = self.dropout(out)
        out = self.pre_refine_norm(out)
        out = self.refine_block(out)
        return self.final_conv(out).float()
