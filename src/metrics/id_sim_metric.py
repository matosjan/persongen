import torch
import numpy as np
from src.metrics.base_metric import BaseMetric
from src.id_utils.id_metric import IDMetric
from src.id_utils.aligner import Aligner

ID_EMBEDS_PTH = "id_embeds.pth"
VAL_ID_EMBEDS_PTH = "val_id_embeds.pth"


def cos_sim(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    return arr1 @ arr2 / np.linalg.norm(arr1) / np.linalg.norm(arr2)


class IDSim(BaseMetric):
    def __init__(self, device='cuda', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.aligner = Aligner()
        self.metric = IDMetric(device=device)
        self.id_embeds = torch.load(ID_EMBEDS_PTH)
        self.id_embeds.update(torch.load(VAL_ID_EMBEDS_PTH))

    def __call__(self, **batch):
        generated_img_cropped, _, embeds = self.aligner(batch['generated'])
        if len(generated_img_cropped) == 0:
            print("FACES NOT FOUND")
            return 0

        assert type(batch["id"]) is str

        result = 0
        for embed in embeds:
            result += cos_sim(embed, self.id_embeds[batch["id"]])
        result = result / len(embeds)
        return result