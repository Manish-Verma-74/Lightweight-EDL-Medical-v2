"""Backbone + framework-specific head. forward() returns a dict of raw head outputs."""
import torch.nn as nn

from edl.constants import FRAMEWORKS, NUM_CLASSES
from edl.losses.fedl import FEDLHead
from edl.models.backbones import build_backbone


class LightweightModel(nn.Module):
    def __init__(self, backbone: str, framework: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        if framework not in FRAMEWORKS:
            raise ValueError(f"unknown framework {framework}")
        self.backbone_name = backbone
        self.framework_name = framework
        self.num_classes = num_classes
        self.features, self.feature_dim = build_backbone(backbone, pretrained)
        self.head = FEDLHead(self.feature_dim, num_classes) if framework == "f_edl" \
            else nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        z = self.features(x)
        if self.framework_name == "f_edl":
            alpha, p, tau = self.head(z)
            return {"alpha": alpha, "p": p, "tau": tau}
        return {"logits": self.head(z)}
