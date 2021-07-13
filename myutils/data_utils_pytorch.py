import os
import sys
sys.path.append(os.path.dirname(os.path.basename(__file__)))\

import pandas
from typing import Any, List, Callable, Tuple, Dict, Union, Optional
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .data_utils import load_list_data


ImageT = torch.Tensor
TargetT = Dict[str, torch.Tensor]

class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, list_image_path: List[str], list_boxes: List[Any], list_classes: List[Any],
                 transforms: Callable[[Any], torch.tensor]):
        self.list_image_path = list_image_path
        self.list_boxes = list_boxes
        self.list_classes = list_classes
        self.transforms = transforms

    def __len__(self):
        return len(self.list_image_path)

    def __getitem__(self, index) -> Tuple[ImageT, TargetT]:
        image_path = self.list_image_path[index]
        image = Image.open(image_path).convert('RGB')

        num_object = len(self.list_boxes[index])
        boxes = torch.as_tensor(self.list_boxes[index], dtype=torch.float32)
        labels = torch.as_tensor(self.list_classes[index], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # Create more target to evaluate by COCO metric
        target["image_id"] = torch.tensor([index])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((num_object,), dtype=torch.int64)

        if self.transforms is not None:
            image, target = (self.transforms(image), target)

        return image, target

def collate_fn(batch: List[Tuple[ImageT, TargetT]]) -> Tuple[Tuple[ImageT, ...], Tuple[TargetT, ...]]:
    # Convert batch (list of tensor) into tuple contain sequences of image and sequences of target
    return tuple(zip(*batch))


def load_dataset(dataframe: pandas.DataFrame, folder_image_path: str, batch_size: int, shuffle: bool = True,
                 size: Optional[Union[int, tuple]] = None) -> DataLoader:
    (list_image_path, list_boxes, list_classes) = load_list_data(dataframe, folder_image_path,
                                                                 label_off_set=0, norm=False)

    # As default, size = 1 is mean keep the original size of image
    transforms_ = transforms.ToTensor()
    if (size != None):
        transforms_ = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    dataset = ObjectDetectionDataset(list_image_path, list_boxes, list_classes,
                                     transforms_)
    dataset = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle, num_workers=2,
                         collate_fn=collate_fn)

    return dataset
