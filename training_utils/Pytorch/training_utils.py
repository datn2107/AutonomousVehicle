import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import torch
import torchvision
from torch import nn
from typing import Callable, Any
from vision.torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from vision.torchvision.models.detection.ssd import ssd300_vgg16
from vision.torchvision.models.detection.ssd import SSDClassificationHead

def initialize_FasterRCNN_model(num_class: int) -> nn.Module:
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	input_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_class+1)

	return model

def initialize_SSD300_VGG16_model(num_class: int) -> nn.Module:
	model = ssd300_vgg16(pretrained=True)
	out_channels = model.head.in_channels
	num_anchors = model.head.num_anchors
	model.head.classification_head = SSDClassificationHead(out_channels, num_anchors, num_class+1)
	return model

# def train_loop(dataset, model, loss_fn, optimizer):
# 	for batch, (image, label) in enumerate(dataset):
# 		# Compute prediction and loss
# 		loss = model(image, label)
#
# 		# Backpropagation
# 		optimizer.zero_grad()
# 		loss.backward()
# 		optimizer.step()
#
# 		if batch % 100 == 0:
# 			loss, current = loss.item(), batch * len(X)
# 			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def test_loop(dataloader, model, loss_fn):
# 	size = len(dataloader.dataset)
# 	test_loss, correct = 0, 0
#
# 	with torch.no_grad():
# 		for X, y in dataloader:
# 			pred = model(X)
# 			test_loss += loss_fn(pred, y).item()
# 			correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#
# 	test_loss /= size
# 	correct /= size
# 	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")