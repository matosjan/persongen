import torch
from src.metrics.base_metric import BaseMetric
import numpy as np
import clip

class TextSimMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self.model.eval()

    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        prompt = batch['prompt']
        tokenized_prompt = clip.tokenize([prompt]).to(self.device)
        generated = batch['generated']
        assert type(generated) is list, type(generated) # list of PIL.Image
        assert type(prompt) is str, type(prompt)

            
        score = 0
        preprecessed = [self.preprocess(img) for img in generated]
        images = torch.stack(preprecessed).to(self.device) 
        logits_per_image, logits_per_text = self.model(images, tokenized_prompt)
        return logits_per_text.mean()
