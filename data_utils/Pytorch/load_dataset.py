import os
import sys

import pandas

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from training_utils.draw_bounding_box import visualize_detection
from data_utils.data_utils import load_data_from_dataframe_to_list

import torch
import torch.utils.data
import torchvision.transforms
from typing import Any, List, Callable, Tuple, Dict
from PIL import Image

class CreateDataset(torch.utils.data.Dataset):
	'''
	Custom dataset for pytorch model 
	''' #
	def __init__(self, list_image_path: str, list_boxes: List[Any], list_classes: List[Any], transforms: Callable[[Any], torch.tensor]):
		''' 
		Args:
			list_image_path :str: List of image path  
			list_boxes :list[list[N, 4]]: List of bounding boxes in each image
			list_classes :list[N]: List of classes that corresponding to each bounding box 
			transforms :Callable[[Any], torch.tensor]: Function use to map data 
		''' #
		# Provide essential argument
		# provide data
		self.list_image_path = list_image_path
		self.list_boxes = list_boxes
		self.list_classes = list_classes
		# provide function to map into each dataset
		self.transforms = transforms

	def __len__(self):
		return len(self.list_image_path)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		'''
		It will fetching data from each elemnt (by index) in data you passed into __init__ method 
		
		Args:
			index: Index of data to fetch into dataset
		Returns:
			image, target: Image tensor and Target dictionary  
		''' #
		# Load Image
		image_path = self.list_image_path[index]
		image = Image.open(image_path).convert('RGB')
		image_id = torch.tensor([index])

		num_object = len(self.list_boxes[index])
		# Load Bounding Box
		# boxes = [resize(box, 1280, 720) for box in self.list_boxes[index]]
		boxes = torch.as_tensor(self.list_boxes[index], dtype=torch.float32)
		# area of each bouding box
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		# Load Label of Box
		labels = torch.as_tensor(self.list_classes[index], dtype=torch.int64)

		iscrowd = torch.zeros((num_object,), dtype=torch.int64)

		# Target dictionary contain essential data for label of image
		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		# addition argument to calculate loss
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			image, target = (self.transforms(image), target)

		return image, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[Tuple[torch.Tensor, Dict[str, torch.Tensor]], ...]:
	'''
	After fetching list of data from __build__ method of torch.utils.data.Dataset class, 
	  it passed into this function to collate them into necessary type
	  
	Args:
		batch :list: One batch in dataset 
		
	Note: Use this function to collate tensors that have different shape
	''' #
	print(batch[0])
	return tuple(zip(*batch))


def load_dataset(dataframe: pandas.DataFrame, folder_image_path: str, batch_size: int, shuffle: bool = True):
	'''
	Load data from dataframe to dataset for pytorch
	
	Args:
		dataframe :pandas.DataFrame: Dataframe contain info of data
		folder_image_path :str: Path of folder contain image 
		batch_size :int: Batch size to split dataset
		shuffle :bool: If True shuffle dataset 
	Returns:
		dataset: Dataset of pytorch 
	''' #
	# Load dataset from dataframe
	(list_image_path, list_boxes, list_classes) = load_data_from_dataframe_to_list(dataframe, folder_image_path, label_off_set=0, norm=False)
	# Create dataset of pytorch
	dataset = CreateDataset(list_image_path, list_boxes, list_classes,
								  torchvision.transforms.ToTensor())
	# Load dataset
	dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True,
												shuffle=shuffle, num_workers=8, collate_fn=collate_fn)
	return dataset


if __name__ == "__main__":
	pass