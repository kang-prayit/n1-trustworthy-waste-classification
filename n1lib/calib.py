"""Post-hoc calibration maps fitted on the calfit split: temperature and vector scaling."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y, max_iter=200) -> float:
    z = torch.as_tensor(logits, dtype=torch.float64)
    t = torch.as_tensor(y, dtype=torch.long)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / logT.exp(), t)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(logT.detach().exp())
    return T if np.isfinite(T) and 0.01 < T < 100 else 1.0


def fit_vector_scaling(logits, y, l2=1e-3, max_iter=300):
    z = torch.as_tensor(logits, dtype=torch.float64)
    t = torch.as_tensor(y, dtype=torch.long)
    C = z.shape[1]
    w = torch.ones(C, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(C, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z * w + b, t) + l2 * ((w - 1) ** 2).sum() + l2 * (b ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy(), b.detach().numpy()


def apply_temperature(logits, T):
    return softmax_np(logits / T)


def apply_vector(logits, wb):
    w, b = wb
    return softmax_np(logits * w + b)
