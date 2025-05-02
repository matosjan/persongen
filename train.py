import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from accelerate import Accelerator

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
import itertools
import os

import subprocess
import datetime
import torch.distributed

def get_nvidia_smi_output():
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    # torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    set_random_seed(config.trainer.seed)
    print(config.model.rank)
    print(os.getpid())

    accelerator = Accelerator()

    project_config = OmegaConf.to_container(config)
    if accelerator.is_main_process:
        logger = setup_saving_and_logging(config)
        writer = instantiate(config.writer, logger, project_config)
    else:
        logger = None
        writer = None

    device = accelerator.device

    # setup data_loader instances
    # batch_transforms should be put on device
    # if accelerator.num_processes > 1:
    #     process_rank = accelerator.process_index
    # else:
    #     process_rank = None
    dataloaders, batch_transforms = get_dataloaders(config, device) #, ddp_rank=process_rank)

    # build model architecture, then print to console
    model = instantiate(config.model, device=device)
    print(device)
    # if accelerator.is_main_process:
        # logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(
                instantiate(metric_config, device=device)
            )

    # build optimizer, learning rate scheduler
    lora_params = filter(lambda p: p.requires_grad, model.unet.parameters())
    other_params =  filter(lambda p: p.requires_grad, model.id_encoder.parameters())
    # visual_projection_params = filter(lambda p: p.requires_grad, model.id_encoder.visual_projection.parameters())
    # vision_backbone_params = filter(lambda p: p.requires_grad, itertools.chain(model.id_encoder.vision_model.parameters(), model.id_encoder.visual_projection.parameters()))
    # visual_projection_2_and_fuse_params = filter(lambda p: p.requires_grad, itertools.chain(model.id_encoder.visual_projection_2.parameters(), model.id_encoder.fuse_module.parameters()))
    trainable_params = [
        {'params': lora_params, 'lr': config.lr_for_lora},
        {'params': other_params, 'lr': config.lr_for_other}
        # {'params': vision_backbone_params, 'lr': config.lr_for_vision_backbone},
        # {'params': visual_projection_params, 'lr': config.lr_for_vis_proj},
        # {'params': visual_projection_2_and_fuse_params, 'lr': config.lr_for_vis_proj2_and_fuse},
    ]

    optimizer = instantiate(config.optimizer, params=trainable_params)
    if accelerator.is_main_process:
        for i, group in enumerate(optimizer.param_groups):
            print(f"Param group {i}:")
            print(f"  learning rate: {group['lr']}")
            print(f"  weight decay:  {group['weight_decay']}")
            print(f"  betas:  {group['betas']}")
            print(f"  eps:  {group['eps']}")

            # list the names or number of params
            print(f"  num params:    {len(group['params'])}")

    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer) 

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    train_dataloader = dataloaders["train"]

    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )

    dataloaders["train"] = train_dataloader

    # allocated = torch.cuda.memory_allocated(device)
    # reserved = torch.cuda.memory_reserved(device)

    # print(f"{os.getpid()} Allocated memory: {allocated / (1024 ** 2):.2f} MB")
    # print(f"{os.getpid()} Reserved memory: {reserved / (1024 ** 2):.2f} MB")   

    trainer = Trainer(
        model=model,
        accelerator=accelerator,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()
    accelerator.end_training()


if __name__ == "__main__":
    main()
