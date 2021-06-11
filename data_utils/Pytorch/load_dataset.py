import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from data_utils.data_utils import load_list_information_from_dataframe
from vision.references.detection.utils import collate_fn

import torch
import torch.utils.data
import torchvision.transforms
from PIL import Image

class CreateDataset(torch.utils.data.Dataset):
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
			image, target = (self.transforms(image), target)

		return image, target

	def __len__(self):
		return len(self.list_image_path)


def load_dataset(df_train, folder_image_path, batch_size):
	(train_list_image_path, train_list_boxes, train_list_classes) = load_list_information_from_dataframe(df_train, folder_image_path, label_off_set=0)

	train_dataset = CreateDataset(train_list_image_path, train_list_boxes, train_list_classes,
								  torchvision.transforms.ToTensor())
	train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
												shuffle=True, num_workers=10, collate_fn=collate_fn)

	return train_dataset

import pandas as pd

if __name__ == "__main__":
	folder_image_path = r'D:\Autonomous Driving\Data\Object Detection\image'
	folder_label_path = r'D:\Autonomous Driving\Data\Object Detection\label'

	#df_train = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))
	df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))

	#train_dataset = load_dataset(df_train, os.path.join(folder_image_path, 'train'), 1)
	test_dataset = load_dataset(df_test, os.path.join(folder_image_path, 'test'), 1)

	for (image, label) in test_dataset:
		print(image, label)
		break