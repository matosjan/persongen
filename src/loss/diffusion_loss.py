import torch
from torch import nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, model_pred, target, **batch):
        if isinstance(model_pred, list):
            loss = 0
            for i in range(len(model_pred)):
                loss = loss + F.mse_loss(model_pred[i].float(), target[i].float())
            loss = loss / len(model_pred)
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return {'loss': loss}
