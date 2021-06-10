import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from data_utils.data_utils import load_list_information_from_dataframe
from data_utils.Pytorch.load_dataset import LoadDataset
from training_utils.Pytorch.training_utils import load_model
from training_utils.Pytorch.training_utils import train_loop
from training_utils.Pytorch.training_utils import test_loop

import torch
import torch.utils.data
import torchvision.utils
import torchvision.transforms
from torch import nn
import pandas as pd
import numpy as np
import argparse
from PIL import Image


def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	df_train = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))
	df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))

	(train_list_image_path, train_list_boxes, train_list_classes) = load_list_information_from_dataframe(df_train, os.path.join(folder_image_path, 'train'), label_off_set=0)
	(test_list_image_path, test_list_boxes, test_list_classes) = load_list_information_from_dataframe(df_test, os.path.join(folder_image_path, 'test'), label_off_set=0)

	train_dataset = LoadDataset(train_list_image_path, train_list_boxes, train_list_classes, torchvision.transforms.ToTensor())
	train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
											 shuffle=True, num_workers=10)
	test_dataset = LoadDataset(test_list_image_path, test_list_boxes, test_list_classes, torchvision.transforms.ToTensor())
	train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
											 shuffle=True, num_workers=10)

	model = load_model(num_class=13)
	model.to(device)

	epochs = 30
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

	for t in range(epochs):
		print(f"Epoch {t + 1}\n-------------------------------")
		train_loop(train_data, model, loss_fn, optimizer)
		test_loop(train_data, model, loss_fn)
	print("Done!")



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