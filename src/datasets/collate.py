import torch
from src.logger.utils import BaseTimer


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in dataset_items])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    original_sizes = [example["original_sizes"] for example in dataset_items]
    crop_top_lefts = [example["crop_top_lefts"] for example in dataset_items]
    caption = [example['caption'] for example in dataset_items]
    ref_images = [example['ref_images'] for example in dataset_items]
    bbox = [example['bbox'] for example in dataset_items]

    result_batch = {
        "pixel_values": pixel_values,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
        "caption": caption,
        "ref_images": ref_images,
        "bbox": bbox
    }
    return result_batch

def collate_fn_val(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    return dataset_items[0]