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

# if is_torch_xla_available():
#     import torch_xla.core.xla_model as xm
#     XLA_AVAILABLE = True
# else:
#     XLA_AVAILABLE = False

from peft import LoraConfig, set_peft_model_state_dict
from diffusers.training_utils import cast_training_params
from transformers import CLIPImageProcessor
from src.model.photomaker.id_encoder import PhotoMakerIDEncoder

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

class PhotoMaker():
    def __init__(self, pretrained_model_name_or_path, rank, weight_dtype, trigger_word='img'):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.lora_rank = rank
        if weight_dtype == 'fp16':
            self.weight_dtype = torch.float16
        elif weight_dtype == 'fp32':
            self.weight_dtype = torch.float32
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
        self.id_encoder = PhotoMakerIDEncoder()
        
        self.num_tokens = 1
        self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)

        self.prepare_for_training()

    def prepare_for_training(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.id_encoder.requires_grad_(True)

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        # The VAE is in float32 to avoid NaN losses.
        self.unet.to("cuda", dtype=torch.float32)
        self.vae.to("cuda", dtype=torch.float32)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        self.text_encoder_2.to("cuda", dtype=self.weight_dtype)
        self.id_encoder.to("cuda", dtype=torch.float32) 

        unet_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k.0", "to_q.0", "to_v.0", "to_out.0"],
        )

        self.unet.add_adapter(unet_lora_config)
    
    # def state_dict(self):


    # time ids
    def compute_time_ids(self, original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (512, 512)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to("cuda", dtype=torch.float32)
        return add_time_ids
    
    def forward(self, pixel_values, caption, ref_images, original_sizes, crop_top_lefts):
        pixel_values = pixel_values.cuda() # нужно ли?
        print(ref_images)
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
        for prompt in caption:
            prompt_embeds, pooled_prompt_embeds, class_tokens_mask = self.encode_prompt_with_trigger_word(
                prompt=prompt,
                device='cuda',
                num_id_images=len(ref_images[0])
            )
            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
            class_tokens_mask_list.append(class_tokens_mask)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        pooled_prompt_embeds = torch.concat(pooled_prompt_embeds_list, dim=0)
        class_tokens_mask = torch.concat(class_tokens_mask_list, dim=0)
        print(prompt_embeds.shape, pooled_prompt_embeds.shape, class_tokens_mask.shape)
        #*********************************************************
        id_pixel_values_list = []
        for input_id_images in ref_images:
            print('preproccesed')
            id_pixel_values_list.append(self.id_image_processor(input_id_images, return_tensors="pt").pixel_values.unsqueeze(0))

        id_pixel_values = torch.concat(id_pixel_values_list, dim=0)
        id_pixel_values = id_pixel_values.to(device='cuda', dtype=torch.float32) # TODO: multiple prompts

        #*********************************************************
        print(id_pixel_values.shape, prompt_embeds.shape, class_tokens_mask.shape)
        prompt_embeds = prompt_embeds.to(dtype=torch.float32)
        prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)

        #*********************************************************
        bs_embed, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        add_text_embeds = pooled_prompt_embeds
        add_text_embeds = add_text_embeds.to('cuda', dtype=torch.float32)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        model_pred = self.unet(
            noisy_model_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        target = noise        
        return {
            'model_pred': model_pred,
            'target': target,
        }


    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        ### Added args
        num_id_images: int = 1,
        class_tokens_mask: Optional[torch.LongTensor] = None,
    ):
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        # if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
        #     self._lora_scale = lora_scale

        #     # dynamically adjust the LoRA scale
        #     if self.text_encoder is not None:
        #         if not USE_PEFT_BACKEND:
        #             adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
        #         else:
        #             scale_lora_layers(self.text_encoder, lora_scale)

        #     if self.text_encoder_2 is not None:
        #         if not USE_PEFT_BACKEND:
        #             adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
        #         else:
        #             scale_lora_layers(self.text_encoder_2, lora_scale)

        # prompt = [prompt] if isinstance(prompt, str) else prompt

        # if prompt is not None:
        #     batch_size = len(prompt)
        # else:
        #     batch_size = prompt_embeds.shape[0]

        # Find the token id of the trigger word
        image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                # if isinstance(self, TextualInversionLoaderMixin):
                #     prompt = self.maybe_convert_prompt(prompt, tokenizer)

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
                
                clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
                class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)

                prompt_embeds = text_encoder(clean_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
                
                print(prompt_embeds.shape)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(device=device)
        class_tokens_mask = class_tokens_mask.to(device=device) # TODO: ignoring two-prompt case
        # get unconditional embeddings for classifier free guidance
        # zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        # if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
        #     negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        #     negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        # elif do_classifier_free_guidance and negative_prompt_embeds is None:
        #     negative_prompt = negative_prompt or ""
        #     negative_prompt_2 = negative_prompt_2 or negative_prompt

        #     # normalize str to list
        #     negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        #     negative_prompt_2 = (
        #         batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
        #     )

        #     uncond_tokens: List[str]
        #     if prompt is not None and type(prompt) is not type(negative_prompt):
        #         raise TypeError(
        #             f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
        #             f" {type(prompt)}."
        #         )
        #     elif batch_size != len(negative_prompt):
        #         raise ValueError(
        #             f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
        #             f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
        #             " the batch size of `prompt`."
        #         )
        #     else:
        #         uncond_tokens = [negative_prompt, negative_prompt_2]

        #     negative_prompt_embeds_list = []
        #     for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
        #         if isinstance(self, TextualInversionLoaderMixin):
        #             negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

        #         max_length = prompt_embeds.shape[1]
        #         uncond_input = tokenizer(
        #             negative_prompt,
        #             padding="max_length",
        #             max_length=max_length,
        #             truncation=True,
        #             return_tensors="pt",
        #         )

        #         negative_prompt_embeds = text_encoder(
        #             uncond_input.input_ids.to(device),
        #             output_hidden_states=True,
        #         )
        #         # We are only ALWAYS interested in the pooled output of the final text encoder
        #         negative_pooled_prompt_embeds = negative_prompt_embeds[0]
        #         negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

        #         negative_prompt_embeds_list.append(negative_prompt_embeds)

        #     negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(device=device)
        else:
            prompt_embeds = prompt_embeds.to(device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape

        # if do_classifier_free_guidance:
        #     # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        #     seq_len = negative_prompt_embeds.shape[1]

        #     if self.text_encoder_2 is not None:
        #         negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        #     else:
        #         negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

        #     negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        #     negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        # if do_classifier_free_guidance:
        #     negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
        #         bs_embed * num_images_per_prompt, -1
        #     )

        # if self.text_encoder is not None:
        #     if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
        #         # Retrieve the original scale by scaling back the LoRA layers
        #         unscale_lora_layers(self.text_encoder, lora_scale)

        # if self.text_encoder_2 is not None:
        #     if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
        #         # Retrieve the original scale by scaling back the LoRA layers
        #         unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, pooled_prompt_embeds, class_tokens_mask