import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.io_utils import ROOT_PATH
from src.model.photomaker.pipeline_orig import PhotoMakerStableDiffusionXLPipeline
from src.model.ip_adapter.pipeline_orig import IPMakerStableDiffusionXLPipeline
from src.model.ipmakerbg.pipeline_orig import IPMakerBGStableDiffusionXLPipeline

from omegaconf import OmegaConf


warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    project_config = OmegaConf.to_container(config)
    # logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, None, project_config)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model)

    # get metrics
    metrics = instantiate(config.metrics)

    # save_path for model predictions
    save_path = ROOT_PATH / "inference_results" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                config.model.pretrained_model_name_or_path, #'SG161222/RealVisXL_V3.0',  
                torch_dtype=torch.float16, 
                use_safetensors=True, 
                # variant="fp16"
            )

    inferencer = Inferencer(
        model=model,
        pipe=pipe,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        writer=writer,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")
    
    with open(f'{save_path}/metrics.txt', '+w') as f:
        for part in logs.keys():
            for key, value in logs[part].items():
                full_key = part + "_" + key
                f.write(f"    {full_key:15s}: {value}\n")

if __name__ == "__main__":
    main()
