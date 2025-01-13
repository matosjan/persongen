import torch
from torch import nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, model_pred, target, **batch):
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return {'loss': loss}
