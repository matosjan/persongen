import torch
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection

pretrained_model_state_dict = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").state_dict()
torch.save(pretrained_model_state_dict, 'clip_encoder.pth')
