import os
import sys
sys.path.append(os.path.dirname(os.path.basename(__file__)))

from utils.data_utils_pytorch import load_dataset
from utils.models_pytorch import ssd300_vgg16
from utils.models_pytorch import faster_rcnn
from utils.train_pytorch import train_one_epoch
# from vision.references.detection.engine import train_one_epoch
from vision.references.detection.engine import evaluate
from utils.draw_bounding_box import visualize_detection
from utils.data_utils import load_list_data
from utils.data_utils import create_yolo_labels

import torch
import torch.utils.data
import pandas as pd
import argparse
import numpy as np


def load_model(ckpt=None):
	if ckpt == None:
		ckpt = checkpoint_path

	model = faster_rcnn(num_class=13)
	if os.path.exists(ckpt):
		model.load_state_dict(torch.load(ckpt))
	model.to(device)

	return model


def metric():
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
	## Load dataframe
	df_train = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))

	## Load dataset
	train_dataset = load_dataset(df_train, os.path.join(folder_image_path,'train'),
								 batch_size, shuffle=True, size=(244,244))

	## Load model
	model = load_model()

	## Setup essential parameter for model
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
	parser.add_argument('--fip', type=str, help='Folder Image Path')
	parser.set_defaults(fip=r'D:\Machine Learning Project\Autonomous Driving\Data\Object Detection\images')
	# Folder Label Path argument
	parser.add_argument('--flp', type=str, help='Folder Label Path')
	parser.set_defaults(flp=r'D:\Machine Learning Project\Autonomous Driving\Data\Object Detection\labels')
	# Batch Size argument
	parser.add_argument('--bs', type=int, help='Batch size to split image dataset')
	parser.set_defaults(bs=8)
	# Checkpoint Path argument
	parser.add_argument('--cp', type=str, help='Save Checkpoint Path (File)')
	parser.set_defaults(cp=r'D:\Machine Learning Project\Autonomous Driving\SourceCode\epoch_19.pt')


	args = parser.parse_args()
	folder_image_path = args.fip
	folder_label_path = args.flp
	batch_size = args.bs
	checkpoint_path = args.cp
	checkpoint_dir = os.path.dirname(checkpoint_path)
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


	# Create labels for yolo
	# for dir in ["train", "test"]:
	# 	if (not os.path.isdir(os.path.join(folder_label_path, dir))):
	# 		os.mkdir(os.path.join(folder_label_path, dir))
	# 		dataframe = pd.read_csv(os.path.join(folder_label_path, dir + ".csv"))
	# 		create_yolo_labels(dataframe, os.path.join(folder_image_path, dir), os.path.join(folder_label_path, dir))


	# train()
