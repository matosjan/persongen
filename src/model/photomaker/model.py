import torch
from torch import nn
from torch.nn import Sequential
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
import PIL
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
PipelineImageInput = Union[
    PIL.Image.Image,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[torch.FloatTensor],
]

from diffusers.callbacks import (
    MultiPipelineCallbacks,
    PipelineCallback,
)
from diffusers.utils import (
    _get_model_file,
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    scale_lora_layers,
    unscale_lora_layers,
)
import itertools
# if is_torch_xla_available():
#     import torch_xla.core.xla_model as xm
#     XLA_AVAILABLE = True
# else:
#     XLA_AVAILABLE = False

from peft.utils import get_peft_model_state_dict
from peft import LoraConfig, set_peft_model_state_dict
from transformers import CLIPImageProcessor
from src.model.photomaker.id_encoder import PhotoMakerIDEncoder
from src.model.photomaker.orig_id_encoder import OrigPhotoMakerIDEncoder

from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

class PhotoMaker(nn.Module):
    def __init__(self, pretrained_model_name_or_path, rank, weight_dtype, device='cuda', trigger_word='img'):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.lora_rank = rank
        if weight_dtype == 'fp16':
            self.weight_dtype = torch.float16
        elif weight_dtype == 'fp32':
            self.weight_dtype = torch.float32
        self.device = device
        self.trigger_word = trigger_word

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )

        self.tokenizer_2 = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            use_fast=False,
        )

        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.pretrained_model_name_or_path, None
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.pretrained_model_name_or_path, None, subfolder="text_encoder_2"
        )

        self.text_encoder = text_encoder_cls_one.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.text_encoder_2 = text_encoder_cls_two.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder_2"
        )

        self.noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")

        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="unet",
        )

        self.id_image_processor = CLIPImageProcessor()
        # self.id_encoder = PhotoMakerIDEncoder()
        self.id_encoder = OrigPhotoMakerIDEncoder()    

        self.num_tokens = 1
        self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)

        self.prepare_for_training()

    def prepare_for_training(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.unet.requires_grad_(False)
        # self.id_encoder.requires_grad_(True)
        self.id_encoder.requires_grad_(False)
        # делаем обучаемым первый проекционный слой
        for param in self.id_encoder.visual_projection.parameters():
            param.requires_grad = True

        # делаем обучаемым второй проекционный слой
        for param in self.id_encoder.visual_projection_2.parameters():
            param.requires_grad = True

        # делаем обучаемым весь FuseModule
        for param in self.id_encoder.fuse_module.parameters():
            param.requires_grad = True


        # Move unet, vae and text_encoder to device and cast to weight_dtype
        # The VAE is in float32 to avoid NaN losses.
        self.unet.to(dtype=torch.float32)
        self.vae.to(dtype=torch.float32)
        self.text_encoder.to(dtype=self.weight_dtype)
        self.text_encoder_2.to(dtype=self.weight_dtype)
        self.id_encoder.to(dtype=torch.float32)

        unet_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        self.unet.add_adapter(unet_lora_config)

        lora_params = 0
        for name, p in self.unet.named_parameters():
            if p.requires_grad:
                lora_params += 1
                # print(p.shape)
        print(f'Lora params {lora_params}')

        encoder_params = 0
        for name, p in self.id_encoder.named_parameters():
            if p.requires_grad:
                encoder_params += 1  
        
        print(f'Encoder params {encoder_params}')


    def get_state_dict(self):
        lora_weights = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet))
        id_encoder_state_dict = self.id_encoder.state_dict()

        return {
            'lora_weights': lora_weights,
            'id_encoder': id_encoder_state_dict,
        }
    
    def load_state_dict_(self, state_dict):
        self.id_encoder.load_state_dict(state_dict['id_encoder'])
        print('loaded id encoder into model')
        lora_state_dict = state_dict['lora_weights']
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items()}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(self.unet, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
                assert 1 == 0
        print('loaded lora into model')

    # time ids
    def compute_time_ids(self, original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = [512, 512]
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(self.device, dtype=torch.float32)
        return add_time_ids
    
    def forward(self, pixel_values, caption, ref_images, original_sizes, crop_top_lefts, bbox, do_cfg=False, masked_loss=False):
        pixel_values = pixel_values.to(self.device) # нужно ли?
        model_input = self.vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * self.vae.config.scaling_factor
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)

        bsz = model_input.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device  # num_train_timesteps ???
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

        add_time_ids = torch.cat(
            [self.compute_time_ids(s, c) for s, c in zip(original_sizes, crop_top_lefts)]
        )

        #*********************************************************
        prompt_embeds_list, pooled_prompt_embeds_list, class_tokens_mask_list = [], [], []
        for prompt, refs in zip(caption, ref_images):
            prompt_embeds, pooled_prompt_embeds, class_tokens_mask = self.encode_prompt_with_trigger_word(
                prompt=prompt,
                num_id_images=len(refs),
                do_cfg=do_cfg,
            )
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
            class_tokens_mask_list.append(class_tokens_mask)
            #*********************************************************
            if do_cfg == False:
                id_pixel_values = self.id_image_processor(refs, return_tensors="pt").pixel_values.unsqueeze(0)
                id_pixel_values = id_pixel_values.to(self.device, dtype=self.id_encoder.dtype)
                prompt_embeds = prompt_embeds.to(dtype=self.id_encoder.dtype)
                prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        pooled_prompt_embeds = torch.concat(pooled_prompt_embeds_list, dim=0)

        if do_cfg == True:
            dummy = sum(p.sum() for p in self.id_encoder.parameters()) * 0
            prompt_embeds += dummy

        #*********************************************************
        bs_embed, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1).to(dtype=self.unet.dtype)
        add_text_embeds = pooled_prompt_embeds
        add_text_embeds = add_text_embeds.to(self.device, dtype=self.unet.dtype)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        model_pred = self.unet(
            noisy_model_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        target = noise
        if masked_loss:
            model_pred = list(torch.split(model_pred, 1, dim=0))
            target = list(torch.split(target, 1, dim=0))
            for i, box in enumerate(bbox):
                box[0], box[1], box[2], box[3] = int(box[0] // 8), int(box[1] // 8), int(box[2] // 8), int(box[3] // 8)
                model_pred[i] = model_pred[i][0, :, box[1]:box[3], box[0]:box[2]]
                target[i] = target[i][0, :, box[1]:box[3], box[0]:box[2]]
                
        return {
            'model_pred': model_pred,
            'target': target,
        }


    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ### Added args
        num_id_images: int = 1,
        class_tokens_mask: Optional[torch.LongTensor] = None,
        do_cfg: bool = False,
    ):
        # Find the token id of the trigger word
        image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        prompt = prompt if do_cfg == False else ''

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids 
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    print(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                if not do_cfg:
                    clean_index = 0
                    clean_input_ids = []
                    class_token_index = []
                    # Find out the corresponding class word token based on the newly added trigger word token
                    for i, token_id in enumerate(text_input_ids.tolist()[0]):
                        if token_id == image_token_id:
                            class_token_index.append(clean_index - 1)
                        else:
                            clean_input_ids.append(token_id)
                            clean_index += 1

                    if len(class_token_index) != 1:
                        raise ValueError(
                            f"PhotoMaker currently does not support multiple trigger words in a single prompt.\
                                Trigger word: {self.trigger_word}, Prompt: {prompt}."
                        )
                    class_token_index = class_token_index[0]

                    # Expand the class word token and corresponding mask
                    class_token = clean_input_ids[class_token_index]
                    clean_input_ids = clean_input_ids[:class_token_index] + [class_token] * num_id_images * self.num_tokens + \
                        clean_input_ids[class_token_index+1:]                
                        
                    # Truncation or padding
                    max_len = tokenizer.model_max_length
                    if len(clean_input_ids) > max_len:
                        clean_input_ids = clean_input_ids[:max_len]
                    else:
                        clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                            max_len - len(clean_input_ids)
                        )

                    class_tokens_mask = [True if class_token_index <= i < class_token_index+(num_id_images * self.num_tokens) else False \
                        for i in range(len(clean_input_ids))]
                    
                    text_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
                    class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)
                    class_tokens_mask = class_tokens_mask.to(self.device)                    

                prompt_embeds = text_encoder(text_input_ids.to(self.device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
            
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(self.device)
        
        bs_embed, _, _ = prompt_embeds.shape
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    
        return prompt_embeds, pooled_prompt_embeds, class_tokens_mask