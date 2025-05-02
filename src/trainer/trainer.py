import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import itertools
from src.logger.utils import BaseTimer
import os


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()
            
        #######################
        do_cfg =  (torch.rand(1) < self.config.hyperparams.do_cfg_p).item()
        masked_loss = (torch.rand(1) < self.config.hyperparams.masked_loss_p).item()
        output = self.model(**batch, do_cfg=do_cfg, masked_loss=masked_loss)
        batch.update(output)

        #######################
        all_losses = self.criterion(**batch)
        batch.update(all_losses)
        try:
            assert torch.isfinite(batch["loss"])
        except AssertionError as e:
            print(masked_loss)
            for pred, target, box in zip(batch['model_pred'], batch['target'], batch['bbox']):
                print(pred.shape, target.shape, box)
            raise e
        if self.is_train:
            self.accelerator.backward(batch["loss"]) # sum of all losses is always called loss
            # print(os.getpid(), [len(a) for a in batch['ref_images']], do_cfg, masked_loss)
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)

        for loss_name in self.config.writer.loss_names:
            batch["loss"] = self.accelerator.gather(batch["loss"]).mean()
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            mean_metric = self.accelerator.gather(met(**batch)).mean()
            metrics.update(met.name, mean_metric)

        return batch
    
    def process_evaluation_batch(self, batch, metrics=None, part='val/'):
        generator = torch.Generator(device='cpu').manual_seed(42)
        generated_images = self.pipe(
            prompt=batch['prompt'],
            input_id_images=list(batch['ref_images']),
            generator=generator,
            target_size=(512, 512),
            original_size=(512, 512),
            crops_coords_top_left=(0, 0),
            **self.config.validation_args
        ).images

        batch[f'generated'] = generated_images

        for met in self.metrics['inference']:
            metrics.update(met.name, met(**batch))
        return batch
        
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            prompt = batch['prompt']
            ref_num = len(batch['ref_images'])
            assert len(batch['ref_images']) == 1, len(batch['ref_images'])

            ref_img = batch['ref_images'][0]
            generated_img = batch['generated'][0]

            image_arrays = [np.array(ref_img.resize((256, 256))), np.array(generated_img.resize((256, 256)))]
            concated_image = Image.fromarray(np.concatenate(image_arrays, axis=1))

            image_name = f"{mode}_images/{batch['id']}/{batch['image_name']}/{prompt[:30]}..."
            self.writer.add_image(image_name, concated_image)

