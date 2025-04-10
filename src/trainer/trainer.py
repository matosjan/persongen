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
        
        # self.accelerator.wait_for_everyone()
         
        #######################
        do_cfg = (torch.rand(1) < 0.1).item()
        masked_loss = (torch.rand(1) < 0.5).item()
        print(os.getpid(), do_cfg, masked_loss)
        output = self.model(**batch, do_cfg=do_cfg, masked_loss=masked_loss)
        batch.update(output)

        #######################
        all_losses = self.criterion(**batch)
        batch.update(all_losses)
        print(os.getpid(), all_losses)
        assert torch.isfinite(batch["loss"])

        if self.is_train:
            print(f'before backward {os.getpid()}')
            self.accelerator.backward(batch['loss']) # sum of all losses is always called loss
            print(f'after backward {os.getpid()}')
            self._clip_grad_norm()
            self.optimizer.step()
            print(f'after optstep {os.getpid()}')
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            # batch[loss_name] = self.accelerator.reduce(batch[loss_name], reduction="mean")
            metrics.update("train/" + loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metric_reduced = self.accelerator.reduce(met(**batch), reduction="mean")
            metrics.update("train/" + met.name, metric_reduced)
        
        print(f'after reduce {os.getpid()}')

        return batch
    
    def process_evaluation_batch(self, batch, pipe=None, metrics=None):
        generator = torch.Generator(device=self.device).manual_seed(42)
        generated_images = pipe(
            prompt=batch['prompt'],
            input_id_images=list(batch['ref_images']),
            generator=generator,
            **self.config.validation_args
        ).images

        batch[f'generated'] = generated_images

        for met in self.metrics['inference']:
            metrics.update("val/" + met.name, met(**batch))
            
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
            # assert len(batch['ref_images']) == 1, len(batch['ref_images'])

            ref_img = batch['ref_images'][0]
            generated_img = batch['generated'][0]

            image_arrays = [np.array(ref_img.resize((256, 256))), np.array(generated_img.resize((256, 256)))]
            concated_image = Image.fromarray(np.concatenate(image_arrays, axis=1))

            image_name = f"val_images/{batch['id']}/{batch['image_name']}/{prompt[:30]}..."
            self.writer.add_image(image_name, concated_image)

