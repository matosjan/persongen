from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from peft.utils import get_peft_model_state_dict
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from src.model.photomaker.our_pipeline import OurPhotoMakerStableDiffusionXLPipeline
from src.model.photomaker.pipeline_orig import PhotoMakerStableDiffusionXLPipeline
from src.model.ipmakerbg.pipeline_orig import IPMakerBGStableDiffusionXLPipeline
from src.model.ipmakerbgbody.pipeline_orig import IPMakerBGBodyStableDiffusionXLPipeline
from src.model.ip_lora.pipeline_orig import IPMakerLoraStableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler
from src.logger.utils import BaseTimer
from itertools import chain
import os

import subprocess
from src.utils.init_utils import set_random_seed

def get_nvidia_smi_output():
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        pipe,
        accelerator,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.pipe = pipe
        self.accelerator = accelerator
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        train_metrics_names = [m for m in self.config.writer.loss_names] + ["total_grad_norm", "unet_grad_norm", "id_enc_grad_norm", "vis_backbone_grad_norm", "vis_proj_grad_norm", "vis_proj_2_grad_norm", "vis_proj_3_grad_norm", "fuse_module_grad_norm"] + [m.name for m in self.metrics["train"]]
        self.train_metrics = MetricTracker(
            *train_metrics_names,
            writer=self.writer,
        )
        val_metrics_names = [m for m in self.config.writer.loss_names] + [m.name for m in self.metrics["inference"]]
        self.evaluation_metrics = MetricTracker(
            *val_metrics_names,
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))


    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            if self.accelerator.is_main_process:
                self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            if self.accelerator.is_main_process:
                # save logged information into logs dict
                logs = {"epoch": epoch}
                logs.update(result)

                # print logged information to the screen
                for key, value in logs.items():
                    self.logger.info(f"    {key:15s}: {value}")

                # evaluate model performance according to configured metric,
                # save best checkpoint as model_best
                best, stop_process, not_improved_count = self._monitor_performance(
                    logs, not_improved_count
                )

                if epoch % self.save_period == 0 or best:
                    self._save_checkpoint(epoch, save_best=best, only_best=True)

                if stop_process:  # early_stop
                    break

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        logs = {}
        self.is_train = True
        pid = os.getpid()
        self.train_metrics.reset()

        if self.accelerator.is_main_process:
            # print(epoch, os.getpid(), get_nvidia_smi_output())
            self.writer.set_step((epoch - 1) * self.epoch_len)
            self.writer.add_scalar("general/epoch", epoch)

        if epoch == 1 and self.accelerator.is_main_process:
            for part, dataloader in self.evaluation_dataloaders.items():
                if isinstance(self.pipe, PhotoMakerStableDiffusionXLPipeline) or isinstance(self.pipe, IPMakerBGStableDiffusionXLPipeline) or isinstance(self.pipe, IPMakerLoraStableDiffusionXLPipeline) or isinstance(self.pipe, IPMakerBGBodyStableDiffusionXLPipeline):
                    self.pipe.unfuse_lora()
                    self.pipe.delete_adapters("photomaker")
                val_logs = self._evaluation_epoch(epoch - 1, part, dataloader)
                logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})
            self.is_train = True
        
        set_random_seed(self.config.trainer.seed + epoch)

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc=f"train_{pid}", total=self.epoch_len)
        ):  
            
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as oom_e:
                if self.skip_oom:
                    print(pid, "OOM on batch. Skipping batch.")
                    if self.accelerator.is_main_process:
                        self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise oom_e
            except RuntimeError as e:
                print(pid, "Runtimer Error on batch: ", batch)
                raise RuntimeError("Runtime Error")

            total_norm, unet_norm, id_encoder_norm, vis_backbone_norm, vis_proj_norm, vis_proj_2_norm, vis_proj_3_norm, fuse_module_norm = self._get_grad_norm()
            self.train_metrics.update("total_grad_norm", total_norm)
            self.train_metrics.update("unet_grad_norm", unet_norm)
            self.train_metrics.update("id_enc_grad_norm", id_encoder_norm)
            self.train_metrics.update("vis_backbone_grad_norm", vis_backbone_norm)
            self.train_metrics.update("vis_proj_grad_norm", vis_proj_norm)
            self.train_metrics.update("vis_proj_2_grad_norm", vis_proj_2_norm)
            self.train_metrics.update("vis_proj_3_grad_norm", vis_proj_3_norm)
            self.train_metrics.update("fuse_module_grad_norm", fuse_module_norm)
            
            # log current results
            if batch_idx % self.log_step == 0:
                if self.accelerator.is_main_process:
                    self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Reduced Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), batch["loss"].item()
                        )
                    )

                    for group in self.optimizer.param_groups:
                        self.writer.add_scalar(
                            f"general/lr for {group['name']}", group['lr']
                        )

                    # self.writer.add_scalar(
                    #     "general/lr for lora layers", self.lr_scheduler.get_last_lr()[0]
                    # )
                    # self.writer.add_scalar(
                    #     "general/lr for other modules", self.lr_scheduler.get_last_lr()[1]
                    # )

                    # self.writer.add_scalar(
                    #     "general/lr for adapter_modules", self.lr_scheduler.get_last_lr()[1]
                    # )

                    # self.writer.add_scalar(
                    #     "general/lr for lora and enc heads", self.lr_scheduler.get_last_lr()[0]
                    # )

                    # self.writer.add_scalar(
                    #     "general/lr for vis_backbone", self.lr_scheduler.get_last_lr()[0]
                    # )

                    # self.writer.add_scalar(
                    #     "general/lr for vis_proj", self.lr_scheduler.get_last_lr()[1]
                    # )

                    # self.writer.add_scalar(
                    #     "general/lr for vis_proj2 and fuse", self.lr_scheduler.get_last_lr()[2]
                    # )

                    self._log_scalars(self.train_metrics, part="train/")
                    self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs.update(last_train_metrics)

        # Run val/test
        if self.accelerator.is_main_process:
            for part, dataloader in self.evaluation_dataloaders.items():
                if isinstance(self.pipe, PhotoMakerStableDiffusionXLPipeline) or isinstance(self.pipe, IPMakerBGStableDiffusionXLPipeline) or isinstance(self.pipe, IPMakerLoraStableDiffusionXLPipeline) or isinstance(self.pipe, IPMakerBGBodyStableDiffusionXLPipeline):
                    self.pipe.unfuse_lora()
                    self.pipe.delete_adapters("photomaker")
                val_logs = self._evaluation_epoch(epoch, part, dataloader)
                logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.evaluation_metrics.reset()

        torch.cuda.empty_cache()

        self.pipe.to(self.device)
        self.pipe.load_photomaker_adapter(
            self.accelerator.unwrap_model(self.model).get_state_dict(),
            trigger_word="img"
        )

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        # if isinstance(self.pipe, PhotoMakerStableDiffusionXLPipeline):
        self.pipe.fuse_lora()

        self.writer.set_step(epoch * self.epoch_len, part)
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_evaluation_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
                self._log_batch(
                    batch_idx, batch, part
                )  # log only the last batch during inference
            self._log_scalars(self.evaluation_metrics, part=f'{part}/')
        self.pipe.to('cpu')
        torch.cuda.empty_cache()
        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            modules = [self.accelerator.unwrap_model(self.model).unet, self.accelerator.unwrap_model(self.model).id_encoder]  # extend as needed
            params_to_optimize = chain(*(filter(lambda p: p.requires_grad, m.parameters()) for m in modules))
            self.accelerator.clip_grad_norm_(
                params_to_optimize, self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        model = self.accelerator.unwrap_model(self.model)
        unet = model.unet
        id_encoder = model.id_encoder
        adapter_modules = model.adapter_modules if hasattr(model, 'adapter_modules') else None

        # Helper function to compute norm for a module
        def compute_module_grad_norm(module):
            parameters = [p for p in module.parameters() if p.requires_grad and p.grad is not None]
            if not parameters:
                return 0.0
            return torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type).item()

        # Compute individual norms
        unet_norm = compute_module_grad_norm(unet)
        id_encoder_norm = compute_module_grad_norm(id_encoder)
        vis_backbone_norm = compute_module_grad_norm(id_encoder.vision_model)
        vis_proj_norm = compute_module_grad_norm(id_encoder.visual_projection)
        vis_proj_2_norm = compute_module_grad_norm(id_encoder.visual_projection_2)

        if hasattr(id_encoder, 'visual_projection_3'):
            vis_proj_3_norm = compute_module_grad_norm(id_encoder.visual_projection_3)
        else:
            vis_proj_3_norm = 0

        if hasattr(id_encoder, 'fuse_module'):
            fuse_module_norm = compute_module_grad_norm(id_encoder.fuse_module)
        else:
            fuse_module_norm = 0

        if adapter_modules is not None:
            adapter_modules_norm = compute_module_grad_norm(adapter_modules)
        else:
            adapter_modules_norm = 0


        # Compute total norm
        total_norm = torch.norm(torch.tensor([unet_norm, id_encoder_norm, adapter_modules_norm]), norm_type).item()
        # vis_proj_2_and_fuse_norm = torch.norm(torch.tensor([vis_proj_2_norm, fuse_module_norm]), norm_type).item()

        return total_norm, unet_norm, id_encoder_norm, vis_backbone_norm, vis_proj_norm, vis_proj_2_norm, vis_proj_3_norm, fuse_module_norm

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker, part=""):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(part + f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.accelerator.unwrap_model(self.model)).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.accelerator.unwrap_model(self.model).get_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            if self.accelerator.is_main_process:
                self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            if self.accelerator.is_main_process:
                self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        if self.accelerator.is_main_process:
            self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            if self.accelerator.is_main_process:
                self.logger.warning(
                    "Warning: Architecture configuration given in the config file is different from that "
                    "of the checkpoint. This may yield an exception when state_dict is loaded."
                )
        self.accelerator.unwrap_model(self.model).load_state_dict_(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            # or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            if self.accelerator.is_main_process:
                self.logger.warning(
                    "Warning: Optimizer or lr_scheduler given in the config file is different "
                    "from that of the checkpoint. Optimizer and scheduler parameters "
                    "are not resumed."
                )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
            )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            if self.accelerator.is_main_process:
                self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            if hasattr(self, 'accelerator'):
                self.accelerator.unwrap_model(self.model).load_state_dict_(checkpoint["state_dict"])
            else:
                self.model.load_state_dict_(checkpoint["state_dict"])
        else:
            if hasattr(self, 'accelerator'):
                self.accelerator.unwrap_model(self.model).load_state_dict_(checkpoint)
            else:
                self.model.load_state_dict_(checkpoint)
