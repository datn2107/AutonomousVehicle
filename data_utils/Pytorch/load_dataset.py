import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import torch
import torch.utils.data
from PIL import Image


class LoadDataset(torch.utils.data.Dataset):
	def __init__(self, list_image_path, list_boxes, list_classes, transforms):
		self.list_image_path = list_image_path
		self.list_boxes = list_boxes
		self.list_classes = list_classes
		self.transforms = transforms

	def resize(self, box, width, height):
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

		num_obj = len(self.list_boxes[index])
		width, height = image.size
		# Load Bounding Box (return to origin size)
		boxes = [self.resize(box, width, height) for box in self.list_boxes[index]]
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		# Load Label of Box
		labels = torch.as_tensor(self.list_classes[index], dtype=torch.int64)

		iscrowd = torch.zeros((num_obj,), dtype=torch.int64)

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			image, target = self.transforms(image, target)

		return image, target

	def __len__(self):
		return len(self.list_image_path)

