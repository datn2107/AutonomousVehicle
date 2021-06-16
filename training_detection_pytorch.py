import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from data_utils.Pytorch.load_dataset import load_dataset
from training_utils.Pytorch.training_utils import load_model
from vision.references.detection.engine import train_one_epoch, evaluate
from training_utils.draw_bounding_box import visualize_detection
from data_utils.data_utils import load_list_information_from_dataframe

import torch
import torch.utils.data
import pandas as pd
import argparse
import numpy as np


def main():

	'''Load dataset'''
	# df_train = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))
	df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))
	# train_dataset = load_dataset(df_train, os.path.join(folder_image_path,'train'), batch_size, shuffle=True)
	test_dataset = load_dataset(df_test, os.path.join(folder_image_path, 'test'), 1, shuffle=False)

	'''Load model'''
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model = load_model(num_class=13)
	model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'epoch_6.pt')))
	model.to(device)

	'''Setup essential parameter for model'''
	epochs = 30
	optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

	'''Training'''
	# for epoch in range(epochs):
	# 	print(f"Epoch {epoch + 1}\n-------------------------------")
	# 	train_one_epoch(model, optimizer, train_dataset, device, epoch, print_freq=500)
	# 	evaluate(model, test_dataset, device=device)
	# 	torch.save(model.state_dict(), os.path.join(checkpoint_path, 'epoch_' + str(epoch) + '.pt'))
	# print("Done!")

	'''Visualize Detection'''
	(list_image_path, list_boxes, list_classes) = load_list_information_from_dataframe(df_test, folder_image_path,
																					   label_off_set=0, norm=False)
	for index, (image, target) in enumerate(test_dataset):
		# Find loss of detection
		for key in target[0].keys():
			target[0][key] = target[0][key].to(device)
		loss = model(image[0].unsqueeze(0).to(device), target)
		# calculate average loss
		avg_loss = 0
		for key in loss.keys():
			print(key)
			avg_loss += loss[key].cpu().data.numpy()
		avg_loss /= 4
		# Set thresh hold to
		if (avg_loss > 0.6):
			model.eval()
			with torch.no_grad():
				predictions = model(image[0].unsqueeze(0).to(device))
				list_box = []
				list_class = []
				for dict in predictions:
					boxes, classes, scores = dict.values()
				for id in range(len(boxes)):
					if scores[id] > 0.6:
						list_box.append(boxes[id])
						list_class.append(classes[id].cpu().data.numpy())
				visualize_detection(image=np.array(image[0].numpy()), boxes=list_box, classes=list_class,
									image_name='prediction.png')
				visualize_detection(image_path=list_image_path[index], boxes=list_boxes[index], classes=list_classes[index],
									image_name='groundth_true.png')
			break


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