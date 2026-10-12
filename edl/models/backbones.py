"""Lightweight ImageNet-pretrained backbones with explicit torchvision weight enums.

build_backbone() returns (feature_extractor, feature_dim). The final classification Linear layer
is replaced by Identity so the framework-specific head is attached separately.
"""
import torch.nn as nn
import torchvision.models as tvm

from edl.constants import BACKBONES

# name -> (builder name, weights enum name). IMAGENET1K_V1 is pinned explicitly; DEFAULT is never used.
SPEC = {
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "shufflenet_v2": ("shufflenet_v2_x1_0", "ShuffleNet_V2_X1_0_Weights"),
}
WEIGHTS_MEMBER = "IMAGENET1K_V1"


def build_backbone(name: str, pretrained: bool = True):
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name}; expected one of {BACKBONES}")
    builder_name, enum_name = SPEC[name]
    weights = getattr(getattr(tvm, enum_name), WEIGHTS_MEMBER) if pretrained else None
    net = getattr(tvm, builder_name)(weights=weights)

    if name == "shufflenet_v2":
        if not isinstance(net.fc, nn.Linear):
            raise ValueError("unexpected ShuffleNetV2 head")
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
    else:
        classifier = net.classifier
        if not isinstance(classifier, nn.Sequential) or not isinstance(classifier[-1], nn.Linear):
            raise ValueError(f"unexpected classifier structure for {name}")
        feat_dim = classifier[-1].in_features
        classifier[-1] = nn.Identity()
    return net, feat_dim
