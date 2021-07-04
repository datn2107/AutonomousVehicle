import os
import sys

from data_utils.load_dataset_pytorch import load_dataset
from training_utils.training_utils_pytorch import initialize_FasterRCNN_model
from training_utils.training_utils_pytorch import train_one_epoch
# from vision.references.detection.engine import train_one_epoch
from vision.references.detection.engine import evaluate
from training_utils.draw_bounding_box import visualize_detection
from data_utils.data_utils import load_data_from_dataframe_to_list
from data_utils.data_utils import clean_error_bounding_box_in_datafrane

import torch
import torch.utils.data
import pandas as pd
import argparse
import numpy as np


def load_model(ckpt=None):
	if ckpt == None:
		ckpt = checkpoint_path

	model = initialize_FasterRCNN_model(num_class=13)
	if ".pt" in os.path.basename(ckpt) and os.path.exists(ckpt):
		model.load_state_dict(torch.load(ckpt))
	model.to(device)

	return model


def metric():
	## Load dataframe
	df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))
	df_test = clean_error_bounding_box_in_datafrane(df_test)

	## Load dataset
	test_dataset = load_dataset(df_test, os.path.join(folder_image_path, 'test'), batch_size, shuffle=False)

	file_name = "epoch_{index}.pt"
	for index in range(0,40):
		ckpt = os.path.join(checkpoint_dir, file_name.format(index=index))
		print("Evaluate " + file_name)
		model = load_model(ckpt)
		evaluate(model, test_dataset, device=device)


def train():
	## Load dataframe
	df_train = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))
	df_train = clean_error_bounding_box_in_datafrane(df_train)

	## Load dataset
	train_dataset = load_dataset(df_train, os.path.join(folder_image_path,'train'),
								 batch_size, shuffle=True)

	## Load model
	model = load_model()

	## Setup essential parameter for model
	epochs = 40
	optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)


	## Training
	for epoch in range(epochs):
		print(f"Epoch {epoch}\n-------------------------------")
		train_one_epoch(model, optimizer, train_dataset, device, print_freq=1000)
		# train_one_epoch(model, optimizer, train_dataset, device, epoch, print_freq=500) # torchvision version
		torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'epoch_' + str(epoch) + '.pt'))
	print("Done!")


def visualize_result():
	## Load dataframe
	df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))
	df_test = clean_error_bounding_box_in_datafrane(df_test)

	model = load_model()

	## Visualize Detection
	(list_image_path, list_boxes, list_classes) = load_data_from_dataframe_to_list(df_test, folder_image_path,
																				   label_off_set=0, norm=False)
	test_dataset = load_dataset(df_test, os.path.join(folder_image_path, 'test'), 1, shuffle=False)

	for index, (image, target) in enumerate(test_dataset):
		# Find loss of detection
		target = [{key: value.to(device) for key, value in target[0].items()}]
		loss = model(image[0].unsqueeze(0).to(device), target)
		# calculate average loss
		avg_loss = sum(val for val in loss.values())/len(loss.values())

		# Visualize image that have too many error
		if avg_loss > 0.5:
			# turn of training mode
			model.eval()

			with torch.no_grad():
				# get list_box and list_class in prediction
				prediction = model(image[0].unsqueeze(0).to(device))
				list_box = []
				list_class = []
				for dict in prediction:
					boxes, classes, scores = dict.values()
				for id in range(len(boxes)):
					if scores[id] > 0.75:
						list_box.append(boxes[id])
						list_class.append(classes[id].cpu().data.numpy())

				# visualize prediction and ground true by image
				visualize_detection(image=np.array(image[0].numpy()), boxes=list_box, classes=list_class,
									image_name='prediction_' + str(index) + '.png')
				visualize_detection(image_path=list_image_path[index], boxes=list_boxes[index], classes=list_classes[index],
									image_name='groundth_true_' + str(index) + '.png')

		if index > 20:
			break


if __name__ == '__main__':
	## Create args to feed argument from terminal
	parser = argparse.ArgumentParser()
	# Folder Image Path argument
	parser.add_argument('--fip', type=str, help='Folder Image Path')
	parser.set_defaults(fip=r'D:\Machine Learning Project\Autonomous Driving\Data\Object Detection\image')
	# Folder Label Path argument
	parser.add_argument('--flp', type=str, help='Folder Label Path')
	parser.set_defaults(flp=r'D:\Machine Learning Project\Autonomous Driving\Data\Object Detection\label')
	# Batch Size argument
	parser.add_argument('--bs', type=int, help='Batch size to split image dataset')
	parser.set_defaults(bs=8)
	# Checkpoint Path argument
	parser.add_argument('--cp', type=str, help='Save Checkpoint Path (File)')
	parser.set_defaults(cp=r'D:\Machine Learning Project\Autonomous Driving\SourceCode\epoch_19.pt')

	## Take the values from args
	args = parser.parse_args()
	folder_image_path = args.fip
	folder_label_path = args.flp
	batch_size = args.bs
	checkpoint_path = args.cp
	checkpoint_dir = os.path.dirname(checkpoint_path)
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	metric()
