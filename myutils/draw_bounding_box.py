import cv2
import json

import numpy
import numpy as np
from typing import List, Union

BBox = List[Union[int, float]]

def plot_detection(image: numpy.array = None, image_path: str = None, boxes: List[BBox] = None, classes: List[int] = None) -> np.array:
	# file = open('D:\Machine Learning Project\Autonomous Driving\Data\Object Detection\labels\id_category_dict.json', 'r')
	# category = json.load(file)
	# category = dict([(v,k) for k, v in category.items()])
	if image_path != None:
		image = cv2.imread(image_path)
	else:
		if image.shape[0] == 3:
			# convert to cv2 format if it in PIL format
			image = np.array(np.moveaxis(image,0,-1))

	for type, box in zip(classes, boxes):
		x_min = int(box[0])
		y_min = int(box[1])
		x_max = int(box[2])
		y_max = int(box[3])
		image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
		image = cv2.putText(image, str(type), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

	return image


def plot_image(image: numpy.array, image_name: str = "test.png"):
	print(image_name)
	cv2.imwrite(image_name, image)
