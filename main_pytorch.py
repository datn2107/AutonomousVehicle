import os
import sys
sys.path.append(os.path.dirname(os.path.basename(__file__)))

import pandas as pd
import argparse
import numpy as np
import logging

import torch
import torch.utils.data

from utils.data_utils_pytorch import load_dataset
from utils.models_pytorch import ssd300_vgg16
from utils.models_pytorch import faster_rcnn
from utils.train_pytorch import train_one_epoch
# from vision.references.detection.engine import train_one_epoch
from vision.references.detection.engine import evaluate
from utils.draw_bounding_box import visualize_detection
from utils.data_utils import load_list_data
from utils.data_utils import create_yolo_labels



def load_model(ckpt=None):
	if ckpt == None:
		ckpt = checkpoint_path

	model = None
	if model_name == 'faster_rcnn':
		model = faster_rcnn(num_class=13)
	elif model_name == 'ssd':
		model = ssd300_vgg16(num_class=13)
	else:
		logging.getLogger().error("No model name: " + model_name)

	if os.path.exists(ckpt):
		model.load_state_dict(torch.load(ckpt))
	model.to(device)

	return model


def eval():
	## Load dataframe
	df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))

	## Load dataset
	test_dataset = load_dataset(df_test, os.path.join(folder_image_path, 'test'), batch_size, shuffle=False)

	file_name = "epoch_{index}.pt"
	for index in range(0,40):
		ckpt = os.path.join(checkpoint_dir, file_name.format(index=index))
		print("Evaluate " + file_name)
		model = load_model(ckpt)
		evaluate(model, test_dataset, device=device)


def train():
	df_train = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))
	train_dataset = load_dataset(df_train, os.path.join(folder_image_path,'train'),
								 batch_size, shuffle=True, size=(244,244))

	model = load_model()
	epochs = 40
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	## Training
	for epoch in range(epochs):
		print(f"Epoch {epoch}\n-------------------------------")
		# train_one_epoch(model, optimizer, train_dataset, device)
		train_one_epoch(model, optimizer, train_dataset, device, epoch, print_freq=500) # torchvision version
		torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'epoch_' + str(epoch) + '.pt'))
	print("Done!")


def visualize_result():
	## Load dataframe
	df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))

	model = load_model()

	## Visualize Detection
	(list_image_path, list_boxes, list_classes) = load_list_data(df_test, folder_image_path,
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
	parser.add_argument('--images', type=str, help='Folder Image Path')
	parser.set_defaults(labels=r'D:\Machine Learning Project\Autonomous Driving\Data\Object Detection\images')
	# Folder Label Path argument
	parser.add_argument('--labels', type=str, help='Folder Label Path')
	parser.set_defaults(labels=r'D:\Machine Learning Project\Autonomous Driving\Data\Object Detection\labels')
	# Batch Size argument
	parser.add_argument('--batch', type=int, help='Batch size to split image dataset')
	parser.set_defaults(batch=8)
	# Checkpoint Path argument
	parser.add_argument('--checkpoint', type=str, help='Save Checkpoint Path (File)')
	parser.set_defaults(checkpoint=r'D:\Machine Learning Project\Autonomous Driving\SourceCode\epoch_19.pt')
	# Select Model
	parser.add_argument('--model', type=str, help='Model you want to use')
	parser.set_defaults(model=r'yolo')
	# Select Mode
	parser.add_argument('--mode', type=str)
	parser.set_defaults(mode=r'train')


	args = parser.parse_args()
	folder_image_path = args.images
	folder_label_path = args.labels
	batch_size = args.batch
	checkpoint_path = args.checkpoint
	checkpoint_dir = os.path.dirname(checkpoint_path)
	model_name = args.model
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	mode = args.mode


	if model_name == 'yolo':
		create_yolo_labels(folder_image_path, folder_label_path)
	else:
		if mode == 'train':
			train()
		elif mode == 'eval':
			eval()
