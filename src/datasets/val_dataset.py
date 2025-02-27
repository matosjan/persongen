from pathlib import Path

from src.datasets.base_dataset import BaseDataset
import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import itertools

class MeladzeValDataset():
    def __init__(self, images_dir_path=None, prompts_path=None, *args, **kwargs):
        self.prompts = []
        with open(prompts_path, 'r') as f:
            for line in f:
                self.prompts.append(line[:-1])
        image_filenames_list = os.listdir(images_dir_path)
        self.img_paths = sorted([os.path.join(images_dir_path, filename) for filename in image_filenames_list], key=lambda x: int(x.split('/')[-1].split('.')[0]))
        self.ref_sets = []
        for ref_num in range(1, 4):
            self.ref_sets.extend(list(itertools.combinations(self.img_paths, ref_num)))
        # print(self.ref_sets)
    
    def __len__(self):
        return len(self.prompts) * len(self.ref_sets)
    
    def __getitem__(self, ind):
        prompt_ind = ind // len(self.ref_sets)
        ref_set_ind = ind % len(self.ref_sets)
        prompt = self.prompts[prompt_ind]
        ref_set = self.ref_sets[ref_set_ind]
        # print(ref_set, len(self.ref_sets))
        ref_images = [Image.open(path).convert('RGB') for path in ref_set]
        return {
            'prompt': prompt,
            'ref_images': ref_images,
        }