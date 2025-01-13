import torch
import torch.nn.functional as F

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


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
        # batch = self.move_batch_to_device(batch)
        # batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()
            
        #######################
        if self.model.pretrained_vae_model_name_or_path is not None:
            pixel_values = batch["pixel_values"].to(dtype=self.model.weight_dtype)
        else:
            pixel_values = batch["pixel_values"]

        # print(batch)
        pixel_values = pixel_values.cuda() # нужно ли?

        model_input = self.model.vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * self.model.vae.config.scaling_factor
        if self.model.pretrained_vae_model_name_or_path is None:
            model_input = model_input.to(self.model.weight_dtype)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)

        bsz = model_input.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.model.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.model.noise_scheduler.add_noise(model_input, noise, timesteps)

        # time ids
        def compute_time_ids(original_size, crops_coords_top_left):
            # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
            target_size = (512, 512)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to("cuda", dtype=self.model.weight_dtype)
            return add_time_ids

        add_time_ids = torch.cat(
            [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
        )

        #*********************************************************
        prompt_embeds, pooled_prompt_embeds, class_tokens_mask = self.model.encode_prompt_with_trigger_word(
            prompt=batch['caption'],
            device=self.device,
            num_id_images=len(batch['ref_images'][0])
        )
        
        # 6. Prepare the input ID images
        input_id_images = batch['ref_images'][0]
        dtype = next(self.model.id_encoder.parameters()).dtype
        if not isinstance(input_id_images[0], torch.Tensor):
            print('preproccesed')
            id_pixel_values = self.model.id_image_processor(input_id_images, return_tensors="pt").pixel_values

        id_pixel_values = id_pixel_values.unsqueeze(0).to(device=self.device, dtype=dtype) # TODO: multiple prompts
        print(torch.isnan(prompt_embeds).any(), torch.isnan(id_pixel_values).any())
        # 7. Get the update text embedding with the stacked ID embedding
        # if id_embeds is not None:
        #     id_embeds = id_embeds.unsqueeze(0).to(device=self.device, dtype=dtype)
        #     prompt_embeds = self.model.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask, id_embeds)
        # else:
        print(id_pixel_values.dtype, prompt_embeds.dtype, class_tokens_mask.dtype)
        prompt_embeds = prompt_embeds.to(dtype=torch.float32)
        prompt_embeds = self.model.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)
        prompt_embeds = prompt_embeds.to(dtype=torch.float32)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        print(id_pixel_values.shape, prompt_embeds.shape, pooled_prompt_embeds.shape)
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        add_text_embeds = pooled_prompt_embeds
        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device, dtype=torch.float32)
        add_time_ids = add_time_ids.to(self.device, dtype=torch.float32)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        print(noisy_model_input.dtype, add_text_embeds.dtype, prompt_embeds.dtype, add_time_ids.dtype)
        noisy_model_input = noisy_model_input.to(dtype=torch.float32)
        #*********************************************************
        
        # unet_added_conditions = {"time_ids": add_time_ids}
        # prompt_embeds, pooled_prompt_embeds = encode_prompt(
        #     text_encoders=[self.model.text_encoder_one, self.model.text_encoder_two],
        #     tokenizers=[self.model.tokenizer_one, self.model.tokenizer_two],
        #     prompt=batch['caption'],
        #     # text_input_ids_list=[batch["input_ids_one"], batch["input_ids_two"]],
        # )
        # unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

        # Predict the noise residual
        print(torch.isnan(prompt_embeds).any())
        print(torch.isnan(added_cond_kwargs['text_embeds']).any())
        print(torch.isnan(added_cond_kwargs['time_ids']).any())
        model_pred = self.model.unet(
            noisy_model_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if self.model.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.model.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.model.noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.model.noise_scheduler.config.prediction_type}")
        
        batch.update({
            'model_pred': model_pred,
            'target': target,
        })

        #######################
        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            print(batch['loss'])
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
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
            pass
