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
            # for i, a in enumerate(index):
            #     im = np.array(Image.open(index[i]['img_path']).convert("RGB"))
            #     b = index[i]['bbox']
            #     im = im[b[1]: b[3], b[0]: b[2]]
            #     cropped_img = Image.fromarray(im)
            #     cropped_img.save(f'/home/aamatosyan/pers-diffusion/OurPhotoMaker/meladze_set/croppes/{i}.png')

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