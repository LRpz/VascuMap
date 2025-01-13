import collections
from pathlib import Path
from typing import List

import numpy as np
import tifffile as tif
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Assuming transforms.py contains the necessary transformation functions.
from transforms import compose, hard_transforms, post_transforms, scale, contrast, new_axis

class SegmentationDataset(Dataset):
    def __init__(self, images: List[Path], masks: List[Path] = None,
                 transforms=None) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = tif.imread(image_path)
        
        image = contrast(image, .1, 99.9)
        image = scale(image)
        image = new_axis(image)

        result = {"image": image}

        if self.masks is not None:
            mask = tif.imread(self.masks[idx]).astype(int)
            result["mask"] = mask

        if self.transforms is not None:
            result = self.transforms(**result)

        result["filename"] = image_path.name

        return result


def train_datasets(images_path: Path, masks_path: Path,
                    train_transforms_fn, valid_transforms_fn,
                    seed, test_size, format='tif') -> tuple:

    all_images = sorted(images_path.glob(f"*.{format}"))
    all_masks = sorted(masks_path.glob(f"*.{format}")) if masks_path else None

    # Split the image and mask dataset into training and validation sets
    train_images, valid_images = train_test_split(all_images, test_size=test_size, random_state=seed)
    train_masks, valid_masks = train_test_split(all_masks, test_size=test_size, random_state=seed)
    
    return (SegmentationDataset(train_images, train_masks, train_transforms_fn),
            SegmentationDataset(valid_images, valid_masks, valid_transforms_fn))


def get_loaders(train_dataset: Dataset, valid_dataset: Dataset,
                batch_size: int, num_workers: int) -> dict:

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True,
                              pin_memory=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              drop_last=False,
                              pin_memory=True)

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders


def load_data(images_path_str: str, masks_path_str: str=None,
              batch_size=16, seed=0, test_size=0.1, format='tif') -> dict:

    images_path = Path(images_path_str)

    train_transforms = compose([
          hard_transforms(), 
          post_transforms()
          ])

    valid_transforms = compose([
          post_transforms()
          ])

    if masks_path_str:
        
        masks_path=Path(masks_path_str)
        
        # Create datasets with transformations applied.
        train_dataset, valid_dataset = train_datasets(
            images_path,
            masks_path,
            train_transforms,
            valid_transforms, 
            seed, 
            test_size,
            format
            )
               
        # Get data loaders for training and validation subsets.
        return get_loaders(train_dataset,valid_dataset,batch_size,num_workers=0)
         
    else:

        inference_dataset=SegmentationDataset(
            list(images_path.glob(f'*{format}')),transforms=valid_transforms)
            
        inference_loader=DataLoader(inference_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              drop_last=False,
                              pin_memory=True)
        
        return inference_loader