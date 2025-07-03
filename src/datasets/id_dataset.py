from src.datasets.base_dataset import BaseDataset
from copy import copy
import os
import json
import numpy as np
import random
import PIL
from PIL import Image
from src.logger.utils import BaseTimer
from time import time
from collections import defaultdict
from src.id_utils.aligner import Aligner
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = 933120000

DATA_PREFIX = "/home/jovyan/home/jovyan/shares/SR006.nfs2/bobkov/full_dataset"
OOD_DATA_PREFIX = "/home/jovyan/home/jovyan/shares/SR006.nfs2/matos/persongen/data/additional_val"

BASIC_CAPTIONS = [
        "Photo of a person img looking into the camera, wearing a black cloak and red hat. Daytime, 3 mountains in the background, a medieval castle can be seen on the far right mountain",
        """A photo of an angry businessman img in a yellow suit, talking on the phone and looking into the camera. He is in the street of a big city: to the left behind him is a bank building with a big sign above it saying "Neon Bank".""",
        # "A 2d cartoon-style photo of a small but very muscular dwarf img with a long red beard looking into the camera. He has a naked top and is holding a massive battleaxe in his left hand. He is in a tavern, with many tables behind him and a bar with a barman.",
        " A photo of a man img in a space suit, his face is seen very surprised, he is in the desert near Oathis with lake, palm trees and a couple of camels behind him.",
        # "A photo of man img with blue hair , looking at the viewer, dressed in a red clown suit and clown make-up seated on a bench with a windmill far behind him.",
        "A photo of a middle-aged man img in a dark green sweater looking at the viewer, he is in a room with white walls, there is a portrait behind him and a books on a shelf."
        ]

SPLITTED_CAPTIONS = [
        "Photo of a person img looking into the camera, wearing a black cloak and red hat.\n\nDaytime, 3 mountains in the background, a medieval castle can be seen on the far right mountain",
        """A photo of an angry businessman img in a yellow suit, talking on the phone and looking into the camera.\n\nHe is in the street of a big city: to the left behind him is a bank building with a big sign above it saying "Neon Bank".""",
        # "A 2d cartoon-style photo of a small but very muscular dwarf img with a long red beard looking into the camera. He has a naked top and is holding a massive battleaxe in his left hand. He is in a tavern, with many tables behind him and a bar with a barman.",
        " A photo of a man img in a space suit, his face is seen very surprised.\n\nHe is in the desert near Oathis with lake, palm trees and a couple of camels behind him.",
        # "A photo of man img with blue hair , looking at the viewer, dressed in a red clown suit and clown make-up seated on a bench with a windmill far behind him.",
        "A photo of a middle-aged man img in a dark green sweater looking at the viewer.\n\nHe is in a room with white walls, there is a portrait behind him and a books on a shelf."
        ]

SPLITTED_BODY_CAPTIONS = [
        "Photo of a person img looking into the camera.\n\nDaytime, 3 mountains in the background, a medieval castle can be seen on the far right mountain.\n\nThe person is wearing a black cloak and red hat.",
        """A photo of an angry businessman img looking into the camera.\n\nHe is in the street of a big city: to the left behind him is a bank building with a big sign above it saying "Neon Bank".\n\nHe is in a yellow suit, talking on the phone""",
        # "A 2d cartoon-style photo of a small but very muscular dwarf img with a long red beard looking into the camera. He has a naked top and is holding a massive battleaxe in his left hand. He is in a tavern, with many tables behind him and a bar with a barman.",
        " A photo of a man img, his face is seen very surprised.\n\nHe is in the desert near Oathis with lake, palm trees and a couple of camels behind him.\n\nThe man is in a space suit",
        # "A photo of man img with blue hair , looking at the viewer, dressed in a red clown suit and clown make-up seated on a bench with a windmill far behind him.",
        "A photo of a middle-aged man img looking at the viewer.\n\nHe is in a room with white walls, there is a portrait behind him and a books on a shelf.\n\nThe man is in a dark green sweater"
        ]

def get_crop_values(img_data, target_res=512):
    H, W = img_data["orig_image_size"]
    body_crop = img_data["body_crop"]
    crop_size = body_crop[1] - body_crop[0]

    coef = target_res / crop_size
    new_H = H * coef
    new_W = W * coef

    new_body_crop = np.array(body_crop) * coef
    new_body_crop = new_body_crop.astype(np.int64)

    x1 = new_body_crop[0]
    y1 = new_body_crop[2]
    return y1, x1


def get_bigger_crop(img, crop, scale=0.2):
    # to square crop 
    if crop[3] - crop[1] < crop[2] - crop[0]:
        diff = crop[2] - crop[0] - (crop[3] - crop[1])
        if diff % 2 != 0:
            crop[0] -= 1
            diff += 1
        crop[3] += diff // 2
        crop[1] -= diff // 2
    elif crop[2] - crop[0] < crop[3] - crop[1]:
        diff = crop[3] - crop[1] - (crop[2] - crop[0])
        if diff % 2 != 0:
            crop[1] -= 1
            diff += 1
        crop[2] += diff // 2
        crop[0] -= diff // 2
    assert crop[3] - crop[1] == crop[2] - crop[0], crop

    # upscale crop
    to_add = int((crop[3] - crop[1]) * scale)
    h, w, _ = np.array(img).shape
    crop = [max(0, crop[0] - to_add), max(0, crop[1] - to_add), min(w, crop[2] + to_add), min(h, crop[3] + to_add)]
    cropped_arr = np.array(img)[crop[1]:crop[3], crop[0]:crop[2]]
    return Image.fromarray(cropped_arr)



class IDDataset(BaseDataset):
    def __init__(self, data_json_pth=None, *args, **kwargs):
        with open(data_json_pth) as f:
            data_json = json.load(f)

        self.val_ids = [
                        'ji_sung',
                        'nm0000609',
                        'nm0000334',
                        'bosco_ho',
                        'nm0004925',
                        'nm0000674',
                        'nm0028413',
                        'nm0186505',
                        'nm0003683',
                        'nm0015865',
                        'nm0005452'
                        ]
        
        index = []
        self.ids = []
        for k, v in tqdm(data_json.items()):
            if k not in self.val_ids:
                index.append(v)
                self.ids.append(k)

        # to increase bs, delete it
        index = index
        self.ids = self.ids
            
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
        id_dict = self._index[ind]
        id_images = list(id_dict.keys())
        id = self.ids[ind]
        # num_ref_images = random.randint(1, 4) if self.ref_num_random==None else self.ref_num_random.randint(1, 4)
        num_ref_images = random.randint(1, 4)
        num_sampled_images = min(len(id_dict), num_ref_images + 1)
        images_indexes = np.random.choice(len(id_dict), size=num_sampled_images, replace=False)

        instance_data = {}
        ref_images = []

        for i, index in enumerate(images_indexes):
            img_name = id_images[index]
            img_data = id_dict[img_name]

            img = Image.open(f"{DATA_PREFIX}/{id}/{img_name}.jpg")
        
            if i == 0: # target image case
                instance_data["pixel_values"] =  img #img
                instance_data["caption"] = img_data["text"]

                bbox = img_data['new_face_crop']
                w_rescale_factor = img.width / 512
                h_rescale_factor = img.height / 512
                resized_bbox = [bbox[0] // w_rescale_factor, bbox[1] // h_rescale_factor, bbox[2] // w_rescale_factor, bbox[3] // h_rescale_factor]
                instance_data["bbox"] = resized_bbox

            else: # reference image
                crop = img_data['new_face_crop']
                ref_images.append(get_bigger_crop(img, crop=crop))

        instance_data['ref_images'] = ref_images
        orig_size = img_data["orig_image_size"]
        instance_data["original_sizes"] = (orig_size[1], orig_size[0])
        instance_data["crop_top_lefts"] = get_crop_values(img_data)

        instance_data = self.preprocess_data(instance_data)

        return instance_data


class IDValDataset(BaseDataset):
    def __init__(self, data_json_pth=None, *args, **kwargs):

        with open(data_json_pth) as f:
            data_json = json.load(f)

        ids_json = {
            "ji_sung" : ["58", "63"],
            "nm0001841" : ["2", "27"]
        }

        captions = SPLITTED_CAPTIONS

        index = []
        for id, id_data in ids_json.items():
            for img_name in id_data:
                for caption in captions:
                    img_data = copy(data_json[id][img_name])
                    img_data.update({"id": id, "img_name": img_name, "img_path": ""})
                    img_data.update({"caption": caption})
                    index.append(img_data)
          
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
        img_dict = self._index[ind]
        img_name = img_dict["img_name"]
        id = img_dict["id"]
        instance_data = {}

        img = self.load_object(f"{DATA_PREFIX}/{id}/{img_name}.jpg")
        instance_data["pixel_values"] = img
        instance_data["prompt"] = img_dict["caption"]

        crop = img_dict['new_face_crop']
        face_img = get_bigger_crop(img, crop=crop)

        instance_data["ref_images"] = [face_img] 

        instance_data = self.preprocess_data(instance_data)
        instance_data["id"] = id
        instance_data["image_name"] = img_name
        return instance_data
    
class OODValDataset(BaseDataset):
    def __init__(self, data_json_pth=None, *args, **kwargs):

        with open(data_json_pth) as f:
            data_json = json.load(f)
        
        self.ids = [
            'aib',
            'denis',
            'martini',
            'maxim',
            'sanya',
            'valentin',
            'vetrov'
        ]

        captions = SPLITTED_CAPTIONS

        index = []
        for id in self.ids:
            for img_name in data_json[id]:
                for caption in captions:
                    img_data = copy(data_json[id][img_name])
                    img_data.update({"id": id, "img_name": img_name, "img_path": ""})
                    img_data.update({"caption": caption})
                    index.append(img_data)
          
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
        img_dict = self._index[ind]
        img_name = img_dict["img_name"]
        id = img_dict["id"]
        instance_data = {}

        img = self.load_object(f"{OOD_DATA_PREFIX}/{id}/{img_name}.jpg")
        instance_data["pixel_values"] = img
        instance_data["prompt"] = img_dict["caption"]

        crop = img_dict['new_face_crop']
        face_img = get_bigger_crop(img, crop=crop)

        instance_data["ref_images"] = [face_img] 

        instance_data = self.preprocess_data(instance_data)
        instance_data["id"] = id
        instance_data["image_name"] = img_name
        return instance_data
    
class InTrainValDataset(BaseDataset):
    def __init__(self, data_json_pth=None, *args, **kwargs):

        with open(data_json_pth) as f:
            data_json = json.load(f)
        
        self.ids = [
            'nm0005059',
            'allen_deng',
            'gun_atthaphan_poonsawat',
            'nm0025745',
            'kim_tae_ri',
            'nm0000215',
            'nm0511088',
            'nm0005468',
            'nm0001778',
            'nm0005151'
        ]

        captions = SPLITTED_CAPTIONS

        index = []
        for id in self.ids:
            for img_name in data_json[id]:
                for caption in captions:
                    img_data = copy(data_json[id][img_name])
                    img_data.update({"id": id, "img_name": img_name, "img_path": ""})
                    img_data.update({"caption": caption})
                    index.append(img_data)
          
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
        img_dict = self._index[ind]
        img_name = img_dict["img_name"]
        id = img_dict["id"]
        instance_data = {}

        img = self.load_object(f"{DATA_PREFIX}/{id}/{img_name}.jpg")
        instance_data["pixel_values"] = img
        instance_data["prompt"] = img_dict["caption"]

        crop = img_dict['new_face_crop']
        face_img = get_bigger_crop(img, crop=crop)

        instance_data["ref_images"] = [face_img] 

        instance_data = self.preprocess_data(instance_data)
        instance_data["id"] = id
        instance_data["image_name"] = img_name
        return instance_data
    

class OutTrainValDataset(BaseDataset):
    def __init__(self, data_json_pth=None, *args, **kwargs):

        with open(data_json_pth) as f:
            data_json = json.load(f)

        self.ids = [
            'nm0000609',
            'nm0000334',
            'bosco_ho',
            'nm0004925',
            'nm0000674',
            'nm0028413',
            'nm0186505',
            'nm0003683',
            'nm0015865',
            'nm0005452'
        ]

        captions = SPLITTED_CAPTIONS

        index = []
        for id in self.ids:
            for img_name in data_json[id]:
                for caption in captions:
                    img_data = copy(data_json[id][img_name])
                    img_data.update({"id": id, "img_name": img_name, "img_path": ""})
                    img_data.update({"caption": caption})
                    index.append(img_data)
        
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
        img_dict = self._index[ind]
        img_name = img_dict["img_name"]
        id = img_dict["id"]
        instance_data = {}

        img = self.load_object(f"{DATA_PREFIX}/{id}/{img_name}.jpg")
        instance_data["pixel_values"] = img
        instance_data["prompt"] = img_dict["caption"]

        crop = img_dict['new_face_crop']
        face_img = get_bigger_crop(img, crop=crop)

        instance_data["ref_images"] = [face_img]

        instance_data = self.preprocess_data(instance_data)
        instance_data["id"] = id
        instance_data["image_name"] = img_name
        return instance_data
    
    