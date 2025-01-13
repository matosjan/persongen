from pathlib import Path

from src.datasets.base_dataset import BaseDataset
import os
from tqdm import tqdm

class MeladzeDataset(BaseDataset):
    def __init__(self, images_dir_path=None, captions_path=None, *args, **kwargs):
        data = []
        captions = []
        with open(captions_path, 'r') as f:
            for line in f:
                captions.append(line[:-1])
        image_filenames_list = os.listdir(images_dir_path)
        img_paths = sorted([os.path.join(images_dir_path, filename) for filename in image_filenames_list], key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img_path, img_caption in zip(img_paths, captions):
            data.append({
                'img_path': img_path,
                'caption': img_caption
            })
            
        super().__init__(data, img_paths, *args, **kwargs)