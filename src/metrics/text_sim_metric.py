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
        prompts = batch['prompt']
        generated = batch['generated']
        all_scores = []
        for prompt, gen_img_list in zip(prompts, generated):
            tokenized_prompt = clip.tokenize([prompt]).to(self.device)
            
            images = []
            for gen_img in gen_img_list:
                prepr_img = self.preprocess(gen_img)
                images.append(prepr_img)
            images = torch.stack(images).to(self.device)
            logits_per_image, logits_per_text = self.model(images, tokenized_prompt)
            assert len(logits_per_text) == 1 and len(logits_per_text[0]) == images.shape[0], (logits_per_text.shape, images.shape)
            scores_for_prompt = list(logits_per_text[0].cpu().numpy())
            all_scores.extend(scores_for_prompt)

        return np.mean(all_scores)