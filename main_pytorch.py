import os
import sys
sys.path.append(os.path.dirname(os.path.basename(__file__)))

import time
import pandas as pd
import argparse
import numpy as np
import re
from PIL import Image

import torch
import torch.utils.data
import torchvision.transforms as transforms

from myutils.data_utils_pytorch import load_dataset
from mymodels.models_pytorch import load_model
from mymodels.models_pytorch import faster_rcnn
# from utils.train_pytorch import train_one_epoch
from vision.references.detection.engine import train_one_epoch
from vision.references.detection.engine import evaluate
from myutils.draw_bounding_box import plot_detection
from myutils.draw_bounding_box import plot_image
from myutils.data_utils import load_list_data
from myutils.data_utils import create_yolo_labels


def eval():
	df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))
	test_dataset = load_dataset(df_test, os.path.join(folder_image_path, 'test'), batch_size=1, shuffle=False)

	file_name = "epoch_{index}.pt"
	for index in range(0,40):
		ckpt = os.path.join(checkpoint_dir, file_name.format(index=index))
		print("Evaluate " + file_name)
		model = load_model(model_name, ckpt)
		evaluate(model, test_dataset, device=device)


def train():
	df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))
	test_dataset = load_dataset(df_test, os.path.join(folder_image_path, 'test'), batch_size=1, shuffle=False)

	df_train = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))
	train_dataset = load_dataset(df_train, os.path.join(folder_image_path,'train'),
								 batch_size, shuffle=True)

	start_epoch = int(re.findall(r'\d+', checkpoint_path)[-1]) + 1
	model = load_model(model_name, checkpoint_path)
	epochs = 40
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	## Training
	for epoch in range(start_epoch, epochs):
		print(f"Epoch {epoch}\n-------------------------------")
		# train_one_epoch(model, optimizer, train_dataset, device)
		train_one_epoch(model, optimizer, train_dataset, device, epoch, print_freq=4000) # torchvision version
		torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'epoch_' + str(epoch) + '.pt'))
		evaluate(model, test_dataset, device=device)
	print("Done!")


def visualize_result(thresh_hold):
	save_result_path = os.path.join(os.path.dirname(__file__),'VisualizeResult')
	if not os.path.exists(save_result_path):
		os.makedirs(os.path.join(os.path.dirname(__file__),'VisualizeResult'))

	df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))
	(list_image_path, list_boxes, list_classes) = load_list_data(df_test, os.path.join(folder_image_path, "test"),
																 label_off_set=0, norm=False)
	test_dataset = load_dataset(df_test, os.path.join(folder_image_path, 'test'), 1, shuffle=False)
	model = faster_rcnn(num_class=13)
	model.load_state_dict(torch.load(checkpoint_path))
	model.to(device)

	for index, (image, target) in enumerate(test_dataset):
		target = [{key: value.to(device) for key, value in target[0].items()}]
		loss = model(image[0].unsqueeze(0).to(device), target)
		avg_loss = sum(val for val in loss.values())/len(loss.values())

		if avg_loss > 0.5:
			model.eval()
			with torch.no_grad():
				prediction = model(image[0].unsqueeze(0).to(device))
				list_box = []
				list_class = []
				for dict in prediction:
					boxes, classes, scores = dict.values()
					print(dict.values())
				for id in range(len(boxes)):
					if scores[id] > 0.75:
						list_box.append(boxes[id])
						list_class.append(classes[id].cpu().data.numpy())

				# visualize prediction and ground true by image
				# visualize_detection(image=np.array(image[0].numpy()), boxes=list_box, classes=list_class,
				# 					image_name='prediction_' + str(index) + '.png')
				# visualize_detection(image_path=list_image_path[index], boxes=list_boxes[index], classes=list_classes[index],
				# 					image_name='groundth_true_' + str(index) + '.png')
			break

		if index > 20:
			break


if __name__ == '__main__':
	## Create args to feed argument from terminal
	parser = argparse.ArgumentParser()
	# Folder Image Path argument
	parser.add_argument('--images', type=str, help='Folder Image Path')
	parser.set_defaults(images=r'D:\Machine Learning Project\Autonomous Driving\Data\Object Detection\images')
	# Folder Label Path argument
	parser.add_argument('--labels', type=str, help='Folder Label Path')
	parser.set_defaults(labels=r'D:\Machine Learning Project\Autonomous Driving\Data\Object Detection\labels')
	# Batch Size argument
	parser.add_argument('--batch', type=int, help='Batch size to split image dataset')
	parser.set_defaults(batch=8)
	# Checkpoint Path argument
	parser.add_argument('--checkpoint', type=str, help='Save Checkpoint Path (File)')
	parser.set_defaults(checkpoint=r'D:\Machine Learning Project\Autonomous Driving\SourceCode\epoch_0.pt')
	# Select Model
	parser.add_argument('--model', type=str, help='Model you want to use')
	parser.set_defaults(model=r'faster_rcnn')


	args = parser.parse_args()
	folder_image_path = args.images
	folder_label_path = args.labels
	batch_size = args.batch
	checkpoint_path = args.checkpoint
	checkpoint_dir = os.path.dirname(checkpoint_path)
	model_name = args.model
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# visualize_result()

	if model_name == 'yolo':
		create_yolo_labels(folder_image_path, folder_label_path)
	else:
		visualize_result(0.75)
