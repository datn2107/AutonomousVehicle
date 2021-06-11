import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from data_utils.Pytorch.load_dataset import load_dataset
from training_utils.Pytorch.training_utils import load_model
from vision.references.detection.engine import train_one_epoch, evaluate


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

	train_dataset = load_dataset(df_train, os.path.join(folder_image_path,'train'), batch_size)
	test_dataset = load_dataset(df_test, os.path.join(folder_image_path, 'test'), batch_size)

	model = load_model(num_class=13)
	model.to(device)

	epochs = 30
	optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

	for epoch in range(epochs):
		print(f"Epoch {epoch + 1}\n-------------------------------")
		train_one_epoch(model, optimizer, train_dataset, device, epoch, print_freq=500)
		evaluate(model, test_dataset, device=device)
		torch.save(model.state_dict(), os.path.join(checkpoint_path, str(epoch + 1) + '.pt'))
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
	# Checkpoint Path argument
	parser.add_argument('--cp', type=str, help='Save Checkpoint Path')
	parser.set_defaults(cp=r'D:\Autonomous Driving\SourceCode\checkpoint_fasterrcnn_resmet50_pytorch')

	''' Take the values from args '''
	args = parser.parse_args()
	folder_image_path = args.fip
	folder_label_path = args.flp
	batch_size = args.bs
	checkpoint_path = args.cp


	main()