import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

from src.model.photomaker.our_pipeline import OurPhotoMakerStableDiffusionXLPipeline
from src.model.photomaker.pipeline_orig import PhotoMakerStableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None


        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))


    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}

        dataloader = self.evaluation_dataloaders["val"]
        logs = self._inference_part("val", dataloader)
        part_logs["val"] = logs

        return part_logs

    def process_batch(self, batch_idx, batch, metrics, pipe, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            pipe (SDXLPipeline): Model pipeline to inference it.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """

        generator = torch.Generator(device='cuda').manual_seed(42)
        generated_images = pipe(
            prompt=batch['prompt'],
            input_id_images=list(batch['ref_images']),
            generator=generator,
             **self.config.validation_args
        ).images

        batch["generated"] = generated_images
        prompt_to_path = '_'.join(batch['prompt'][:30].split())
        id = batch["id"]
        img_name = batch["image_name"]

        save_dir = self.save_path / part / f"{id}/{prompt_to_path}/{img_name}"
        save_dir.mkdir(exist_ok=True, parents=True)
        for i, img in enumerate(batch["generated"]):
            save_pth = save_dir / f"{i}.jpg"
            img.save(save_pth)

        for metric in self.metrics['inference']:
            metrics.update(metric.name, metric(**batch))
        return batch



        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)


        pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0', #'SG161222/RealVisXL_V3.0',  
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16"
        )
        pipe.load_photomaker_adapter(
            self.model.state_dict(),
            trigger_word="img"
        )
        pipe.id_encoder.to(self.device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.fuse_lora()
        pipe.to(self.device)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    pipe=pipe,
                    metrics=self.evaluation_metrics,
                    part=part
                )

        return self.evaluation_metrics.result()

