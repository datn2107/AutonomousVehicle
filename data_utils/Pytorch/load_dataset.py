import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from training_utils.draw_bounding_box import visualize_detection
from data_utils.data_utils import load_list_information_from_dataframe

import torch
import torch.utils.data
import torchvision.transforms
from PIL import Image

def resize(box, width, height):
	# rescale bounding box to original size
	box[0] = box[0] * width
	box[1] = box[1] * height
	box[2] = box[2] * width
	box[3] = box[3] * height
	return box

class CreateDataset(torch.utils.data.Dataset):
	'''
	Custom dataset for pytorch model 
	''' #
	def __init__(self, list_image_path, list_boxes, list_classes, transforms):
		'''
		:param list_image_path: List of image path  
		:param list_boxes: List of bounding box in each image
		:param list_classes: List of class that corresponding to each bounding box 
		:param transforms: Function use to map data 
		''' #
		# Provide essential argument
		# provide data
		self.list_image_path = list_image_path
		self.list_boxes = list_boxes
		self.list_classes = list_classes
		# provide function to map into each dataset
		self.transforms = transforms

	def __len__(self):
		'''
		:return: Length of dataset 
		''' #
		return len(self.list_image_path)

	def __getitem__(self, index):
		'''
		It will fetching data from each elemnt (by index) in data you passed into __init__ method 
		
		:param index: Index of data to fetch into dataset
		:return: 
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


def collate_fn(batch):
	'''
	After fetching list of data from __build__ method of torch.utils.data.Dataset class, 
	  it passed into this function to collate them into necessary type
	Note: Use this function to collate tensors that have different shape
	''' #
	return tuple(zip(*batch))


def load_dataset(dataframe, folder_image_path, batch_size):
	'''
	Load data from dataframe to dataset for pytorch
	
	:param dataframe: Dataframe contain info of data
	:param folder_image_path: Path of folder contain image 
	:param batch_size: Batch size to split dataset
	:return: 
	''' #
	# Load dataset from dataframe
	(list_image_path, list_boxes, list_classes) = load_list_information_from_dataframe(dataframe, folder_image_path, label_off_set=0)
	# rescale bonding box to orginal size (torch.utils.data.Dataset class not allow to do inside class, it will cause overload data)
	for i in range(len(list_boxes)):
		for j in range(len(list_boxes[i])):
			list_boxes[i][j] = resize(list_boxes[i][j], 1280, 720)

	# Create dataset of pytorch
	dataset = CreateDataset(list_image_path, list_boxes, list_classes,
								  torchvision.transforms.ToTensor())
	# Load dataset
	dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True,
												shuffle=True, num_workers=8, collate_fn=collate_fn)

	return dataset


if __name__ == "__main__":
	pass