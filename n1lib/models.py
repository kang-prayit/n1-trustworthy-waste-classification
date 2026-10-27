"""timm model registry with a uniform wrapper: backbone(s) -> pooled features -> dropout -> linear.

Inputs to the wrapper are float images in [0,1] at img_size (224). Each branch resizes to its
native resolution if needed (InceptionV3 branch -> 299) and applies its own ImageNet
normalisation, so the hybrid ResNet101+InceptionV3 feature-concat model is handled correctly.
The explicit dropout before the linear head is used for MC-dropout.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# name -> list of (timm_name, branch_input_size or None = use img_size)
REGISTRY = {
    "resnet101": [("resnet101.tv_in1k", None)],
    "hybrid_r101_incv3": [("resnet101.tv_in1k", None), ("inception_v3.tv_in1k", 299)],
    "efficientnetv2_s": [("tf_efficientnetv2_s.in1k", None)],
    "efficientnet_b0": [("efficientnet_b0.ra_in1k", None)],
    "mobilenetv3_large": [("mobilenetv3_large_100.ra_in1k", None)],
    "convnext_tiny": [("convnext_tiny.fb_in1k", None)],
    "deit_small": [("deit_small_patch16_224.fb_in1k", None)],
    # tiny models for the CPU smoke test (random init)
    "tiny_cnn": [("resnet10t", None)],
    "tiny_hybrid": [("resnet10t", None), ("resnet10t", 80)],
}
GRADCAM_OK = {"resnet101", "hybrid_r101_incv3", "efficientnetv2_s", "efficientnet_b0",
              "mobilenetv3_large", "convnext_tiny", "tiny_cnn", "tiny_hybrid"}


class Branch(nn.Module):
    def __init__(self, timm_name, size, pretrained):
        super().__init__()
        self.net = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        cfg = self.net.pretrained_cfg
        self.size = size
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))
        self.num_features = getattr(self.net, "head_hidden_size", None) or self.net.num_features
        self.fmap = None
        self.keep_fmap = False

    def forward(self, x):
        if self.size is not None and x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        f = self.net.forward_features(x)
        if self.keep_fmap:
            if f.requires_grad:
                f.retain_grad()
            self.fmap = f
        return self.net.forward_head(f, pre_logits=True)


class WasteNet(nn.Module):
    def __init__(self, name, num_classes, drop_rate=0.2, pretrained=True):
        super().__init__()
        self.name = name
        self.branches = nn.ModuleList([Branch(t, s, pretrained) for t, s in REGISTRY[name]])
        nf = sum(b.num_features for b in self.branches)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(nf, num_classes)

    def features(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)

    def head(self, f, mc: bool = False):
        if mc:
            f = F.dropout(f, self.dropout.p, training=True)
        else:
            f = self.dropout(f)
        return self.fc(f)

    def forward(self, x):
        return self.head(self.features(x))

    def param_groups(self, lr_backbone, lr_head, wd):
        bb = [p for p in self.branches.parameters() if p.requires_grad]
        hd = list(self.fc.parameters())
        return [dict(params=bb, lr=lr_backbone, weight_decay=wd),
                dict(params=hd, lr=lr_head, weight_decay=wd)]


def build_model(name, num_classes, drop_rate=0.2, pretrained=True):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name}; options: {list(REGISTRY)}")
    return WasteNet(name, num_classes, drop_rate, pretrained)
