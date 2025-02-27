# Merge image encoder and fuse module to create an ID Encoder
# send multiple ID images, we can directly obtain the updated text encoder containing a stacked ID embedding

import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers import PretrainedConfig

VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class FuseModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        # print(prompt_embeds.shape, id_embeds.shape)
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(
        self,
        prompt_embeds,
        id_embeds,
        class_tokens_mask,
    ) -> torch.Tensor:
        # id_embeds shape: [num_inputs_in_batch, 1, 2048]
        id_embeds = id_embeds.to(prompt_embeds.dtype)
        # seq_length: 77
        batch_size, seq_length = prompt_embeds.shape[:2]
        prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        class_tokens_mask = class_tokens_mask.view(-1)
        id_embeds = id_embeds.view(-1, id_embeds.shape[-1])
        # slice out the image token embeddings
        # print(prompt_embeds.shape)
        image_token_embeds = prompt_embeds[class_tokens_mask]
        stacked_id_embeds = self.fuse_fn(image_token_embeds, id_embeds)
        # print(prompt_embeds[~class_tokens_mask].shape)
        print(f'ID embed norm: {torch.norm(id_embeds)}')
        print(f'Stacked embed norm: {torch.norm(stacked_id_embeds)}')
        print(f'Class token embed norm: {torch.norm(prompt_embeds[class_tokens_mask])}')
        print(f'Text embed norm: {torch.norm(prompt_embeds[~class_tokens_mask])}')
        assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"
        zeros_embed = torch.zeros_like(stacked_id_embeds)
        prompt_embeds.masked_scatter_(class_tokens_mask[:, None], zeros_embed.to(prompt_embeds.dtype))
        updated_prompt_embeds = prompt_embeds.view(batch_size, seq_length, -1)
        print(f'Updated embed norm: {torch.norm(updated_prompt_embeds)}')
        return updated_prompt_embeds

class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    def __init__(self):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        self.fuse_module = FuseModule(2048)

    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask):
        # print(id_pixel_values.shape, prompt_embeds.shape, class_tokens_mask.shape)
        _, num_inputs_in_batch, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(num_inputs_in_batch, c, h, w)

        shared_id_embeds = self.vision_model(id_pixel_values)[1]
        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)

        id_embeds = id_embeds.view(num_inputs_in_batch, 1, -1)
        id_embeds_2 = id_embeds_2.view(num_inputs_in_batch, 1, -1)  

        id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)
        # print(id_embeds.shape)
        updated_prompt_embeds = self.fuse_module(prompt_embeds, id_embeds, class_tokens_mask)
        print(updated_prompt_embeds.shape)
        return updated_prompt_embeds


if __name__ == "__main__":
    PhotoMakerIDEncoder()