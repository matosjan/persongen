from pathlib import Path

from src.datasets.base_dataset import BaseDataset
import os
from tqdm import tqdm
import json
from src.id_utils.aligner import Aligner
from PIL import Image
import numpy as np


class MeladzeDataset(BaseDataset):
    def __init__(self, images_dir_path=None, captions_path=None, index_path=None, *args, **kwargs):
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)

        else:
            aligner = Aligner()
            index = []
            image_filenames_list = os.listdir(images_dir_path)
            img_paths = sorted([os.path.join(images_dir_path, filename) for filename in image_filenames_list], key=lambda x: int(x.split('/')[-1].split('.')[0]))
            with open(captions_path, 'r') as f:
                for i, caption in enumerate(f):
                    _, bboxes = aligner([Image.open(img_paths[i]).convert("RGB")])
                    index.append({
                        'img_path': img_paths[i],
                        'caption': caption,
                        'bbox': bboxes[0]
                    })
            with open(index_path, "w+") as f:
                json.dump(index, f, indent=2)
            
        super().__init__(index, *args, **kwargs)


    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        img_path = data_dict["img_path"]
        img = self.load_object(img_path)
        caption = data_dict["caption"]

        ref_images = []
        num_ref_images = random.randint(1, 4)
        used_indexes = {ind}
        while len(ref_images) != num_ref_images:
            new_ref_idx = random.randint(0, len(self._index) - 1)
            if new_ref_idx not in used_indexes:
                used_indexes.add(new_ref_idx)
                ref_images.append(self.load_object(self._index[new_ref_idx]['img_path']))
        
        bbox = data_dict['bbox']
        width_rescale_factor = img.width / 512
        height_rescale_factor = img.height / 512
        resized_bbox = [bbox[0] // width_rescale_factor, bbox[1] // height_rescale_factor, bbox[2] // width_rescale_factor, bbox[3] // height_rescale_factor]
        instance_data = {
            "pixel_values": img,
            "ref_images": ref_images,
            "caption": caption,
            "original_sizes": (img.height, img.width),
            "crop_top_lefts": (0, 0),
            "bbox": resized_bbox,
        }
        instance_data = self.preprocess_data(instance_data)
        return instance_data
