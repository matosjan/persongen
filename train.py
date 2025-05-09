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
from src.model.photomaker.pipeline_orig import PhotoMakerStableDiffusionXLPipeline

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
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
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
    visual_projection_params = filter(lambda p: p.requires_grad, model.id_encoder.visual_projection.parameters())
    visual_projection_2_params = filter(lambda p: p.requires_grad, model.id_encoder.visual_projection_2.parameters())
    fuse_module_params = filter(lambda p: p.requires_grad, model.id_encoder.fuse_module.parameters())
    # other_params =  filter(lambda p: p.requires_grad, model.id_encoder.parameters())
    # id_encoder_params = filter(lambda p: p.requires_grad, model.id_encoder.parameters())

    trainable_params = [
        {'params': lora_params, 'lr': config.lr_for_lora, 'name': 'lora_params'},
        {'params': visual_projection_params, 'lr': config.lr_for_vis_proj, 'name': 'vis_proj_params'},
        {'params': visual_projection_2_params, 'lr': config.lr_for_vis_proj_2, 'name': 'vis_proj_2_params'},
        {'params': fuse_module_params, 'lr': config.lr_for_fuse_module, 'name': 'fuse_module_params'},
        # {'params': id_encoder_params, 'lr': config.lr_id_encoder, 'name': 'id_encoder_params'},
    ]

    optimizer = instantiate(config.optimizer, params=trainable_params)
    if accelerator.is_main_process:
        for i, group in enumerate(optimizer.param_groups):
            print(f"Param group <{group['name']}>:")
            print(f"  learning rate: {group['lr']}")
            print(f"  weight decay:  {group['weight_decay']}")
            print(f"  betas:  {group['betas']}")
            print(f"  eps:  {group['eps']}")

            # list the names or number of params
            print(f"  num params:    {len(group['params'])}")

    # lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer) 
    # with warmup
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer, lr_lambda=lambda step: min((step + 1) / config.lr_warmup_steps, 1.0)) 


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

    pipe = None
    if accelerator.is_main_process:
        print(config.model.pretrained_model_name_or_path)
        pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                config.model.pretrained_model_name_or_path, #'SG161222/RealVisXL_V3.0',  
                torch_dtype=torch.float16, 
                use_safetensors=True, 
                # variant="fp16"
            )

    trainer = Trainer(
        model=model,
        pipe=pipe,
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
