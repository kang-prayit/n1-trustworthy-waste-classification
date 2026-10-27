import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017; Mukhoti et al. 2020 for calibration)."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits.float(), dim=1).gather(1, y[:, None]).squeeze(1)
        p = logp.exp()
        return (-(1 - p) ** self.gamma * logp).mean()


def make_loss(name, tcfg):
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "ls":
        return nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    if name == "focal":
        return FocalLoss(tcfg["focal_gamma"])
    raise KeyError(name)
