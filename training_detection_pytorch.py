import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from data_utils.data_utils import load_list_information_from_dataframe

import torch
import pandas as pd
import numpy as np
import argparse
from PIL import Image


class LoadDataset(object):
	def __init__(self, list_image_path, list_boxes, list_classes, transforms):
		self.list_image_path = list_image_path
		self.list_boxes = list_boxes
		self.list_classes = list_classes
		self.transfroms = transforms

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
		target = []
		target["boxes"] = boxes
		target["labels"] = labels
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			img, target = self.transforms(image, target)

		return image, target

	def __len__(self):
		return len(self.list_image_path)



def main():
	df_train = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))
	df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))

	(train_list_image_path, train_list_boxes, train_list_classes) = load_list_information_from_dataframe(df_train, os.path.join(folder_image_path, 'train'), label_off_set=0)
	(test_list_image_path, test_list_boxes, test_list_classes) = load_list_information_from_dataframe(df_test, os.path.join(folder_image_path, 'test'), label_off_set=0)

	print(train_list_image_path[0], train_list_boxes[0], train_list_classes[0])


if __name__ == '__main__':
	''' Create args to feed argument from terminal '''
	parser = argparse.ArgumentParser()
	# Folder Image Path argument
	parser.add_argument('--fip', type=str, help='Folder Image Path')
	parser.set_defaults(fip=r'D:\Autonomous Driving\Data\Object Detection\image')
	# Folder Label Path argument
	parser.add_argument('--flp', type=str, help='Folder Label Path')
	parser.set_defaults(flp=r'D:\Autonomous Driving\Data\Object Detection\label')
	# Batch Size argument
	parser.add_argument('--bs', type=int, help='Batch size to split image dataset')
	parser.set_defaults(bs=8)

	''' Take the values from args '''
	args = parser.parse_args()
	folder_image_path = args.fip
	folder_label_path = args.flp
	batch_size = args.bs


	main()