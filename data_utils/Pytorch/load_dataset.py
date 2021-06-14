import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from training_utils.draw_bounding_box import visualize_detection
from data_utils.data_utils import load_list_information_from_dataframe

import torch
import torch.utils.data
import torchvision.transforms
from PIL import Image

class CreateDataset(torch.utils.data.Dataset):
	def __init__(self, list_image_path, list_boxes, list_classes, transforms):
		# Provide essential argument
		# provide data
		self.list_image_path = list_image_path
		self.list_boxes = list_boxes
		self.list_classes = list_classes
		# provide function to map into each dataset
		self.transforms = transforms

	def resize(self, box, width, height):
		# rescale bounding box to original size
		box[0] = box[0]*width
		box[1] = box[1]*height
		box[2] = box[2]*width
		box[3] = box[3]*height
		return box

	def __getitem__(self, index):
		# Load Image
		image_path = self.list_image_path[index]
		image = Image.open(image_path).convert('RGB')
		image_id = torch.tensor([index])

		num_object = len(self.list_boxes[index])
		width, height = image.size
		# Load Bounding Box (return to origin size)
		boxes = [self.resize(box, width, height) for box in self.list_boxes[index]]
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
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

	def __len__(self):
		return len(self.list_image_path)


def collate_fn(batch):
	'''
	Collate list of dataset in to batch 

	:param 
		
	
	''' #
	return tuple(zip(*batch))


def load_dataset(dataframe, folder_image_path, batch_size):
	# load dataset from dataframe
	(list_image_path, list_boxes, list_classes) = load_list_information_from_dataframe(dataframe, folder_image_path, label_off_set=0)

	# create dataset of pytorch
	dataset = CreateDataset(list_image_path, list_boxes, list_classes,
								  torchvision.transforms.ToTensor())
	# load dataset
	dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True,
												shuffle=True, num_workers=32, collate_fn=collate_fn)

	return dataset


if __name__ == "__main__":
	pass