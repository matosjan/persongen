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

DATA_PREFIX = "/home/jovyan/shares/SR006.nfs2/free001style/final"
OOD_DATA_PREFIX = "/home/jovyan/shares/SR006.nfs2/matos/persongen_main_version/data/additional_val"


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
    def __init__(self, data_json_pth=None, references_num=1, *args, **kwargs):
        self.references_num = references_num

        with open(data_json_pth) as f:
            data_json = json.load(f)


        # self.faces = {}
        # self.images = {}
        # self.sizes = {}

        # index = []
        # self.ids = []
        # for k, v in tqdm(data_json.items()):
        #     if k in ["ji_sung", "nm0001841"]: #["nm4139037", "nm2955013", "nm8402992"]:
        #         # v.update({"img_path": "", "caption": ""})
        #         index.append(v)
        #         self.ids.append(k)
        #         self.faces[k] = {}
        #         self.images[k]  = {}
        #         self.sizes[k]  = {}

        #         for img_name in v.keys():
        #             img = Image.open(f"{DATA_PREFIX}/{k}/{img_name}.jpg")
        #             self.images[k][img_name] = img.resize((512, 512))
        #             self.sizes[k][img_name] = img.size[0]

        #             crop = v[img_name]['new_face_crop']
        #             face_img = get_bigger_crop(img, crop=crop)
        #             self.faces[k][img_name] = face_img


        index = []
        self.ids = []
        for k, v in tqdm(data_json.items()):
            if k not in ["ji_sung"]:
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


        instance_data["ref_images"] = ref_images
        instance_data["original_sizes"] = (512, 512)
        instance_data["crop_top_lefts"] = (0, 0)

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

        captions = [
        "Photo of a person img looking into the camera, wearing a black cloak and red hat. Daytime, 3 mountains in the background, a medieval castle can be seen on the far right mountain",
        """A photo of an angry businessman img in a yellow suit, talking on the phone and looking into the camera. He is in the street of a big city: to the left behind him is a bank building with a big sign above it saying "Neon Bank".""",
        # "A 2d cartoon-style photo of a small but very muscular dwarf img with a long red beard looking into the camera. He has a naked top and is holding a massive battleaxe in his left hand. He is in a tavern, with many tables behind him and a bar with a barman.",
        " A photo of a man img in a space suit, his face is seen very surprised, he is in the desert near Oathis with lake, palm trees and a couple of camels behind him.",
        # "A photo of man img with blue hair , looking at the viewer, dressed in a red clown suit and clown make-up seated on a bench with a windmill far behind him.",
        "A photo of a middle-aged man img in a dark green sweater looking at the viewer, he is in a room with white walls, there is a portrait behind him and a books on a shelf."
        ]

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
        instance_data["original_sizes"] = (512, 512)
        instance_data["crop_top_lefts"] = (0, 0)

        instance_data = self.preprocess_data(instance_data)
        instance_data["id"] = id
        instance_data["image_name"] = img_name
        return instance_data
    
class OODValDataset(BaseDataset):
    def __init__(self, data_json_pth=None, *args, **kwargs):

        with open(data_json_pth) as f:
            data_json = json.load(f)

        captions = [
        "Photo of a person img looking into the camera, wearing a black cloak and red hat. Daytime, 3 mountains in the background, a medieval castle can be seen on the far right mountain",
        """A photo of an angry businessman img in a yellow suit, talking on the phone and looking into the camera. He is in the street of a big city: to the left behind him is a bank building with a big sign above it saying "Neon Bank".""",
        # "A 2d cartoon-style photo of a small but very muscular dwarf img with a long red beard looking into the camera. He has a naked top and is holding a massive battleaxe in his left hand. He is in a tavern, with many tables behind him and a bar with a barman.",
        " A photo of a man img in a space suit, his face is seen very surprised, he is in the desert near Oathis with lake, palm trees and a couple of camels behind him.",
        # "A photo of man img with blue hair , looking at the viewer, dressed in a red clown suit and clown make-up seated on a bench with a windmill far behind him.",
        "A photo of a middle-aged man img in a dark green sweater looking at the viewer, he is in a room with white walls, there is a portrait behind him and a books on a shelf."
        ]

        index = []
        for id, id_data in data_json.items():
            for img_name in id_data.keys():
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
        instance_data["original_sizes"] = (512, 512)
        instance_data["crop_top_lefts"] = (0, 0)

        instance_data = self.preprocess_data(instance_data)
        instance_data["id"] = id
        instance_data["image_name"] = img_name
        return instance_data