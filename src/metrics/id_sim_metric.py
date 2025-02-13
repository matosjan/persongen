import torch
from src.metrics.base_metric import BaseMetric
from src.id_utils.id_metric import IDMetric
from src.id_utils.aligner import Aligner

class IDSim(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.aligner = Aligner()
        self.metric = IDMetric()

    def __call__(self, **batch):
        ref_img_cropped, _ = self.aligner([batch['ref_images'][0]])
        generated_img_cropped, _ = self.aligner([batch['generated'][0]])
        from_data = {}
        from_data["inp_data"] = ref_img_cropped
        from_data["fake_data"] = generated_img_cropped

        return self.metric("", "", from_data)