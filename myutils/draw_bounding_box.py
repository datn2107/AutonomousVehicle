import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_detection(image=None, image_path=None, boxes=None, classes=None, image_name='test.jpg'):
	'''
	Draw bounding box on image from image path
	
	:param image: image as numpy array (can be None and PIL either cv2 format)
	:param image_path: path to image (can be None)
	:param boxes: list of list coordinate for bouding box in image 
	:return: 
		Image was drawn bounding box 
	''' #
	if image_path != None:
		image = cv2.imread(image_path)
	else:
		if image.shape[0] == 3:
			# convert to cv2 format if it in PIL format
			image = np.multiply(np.array(np.moveaxis(image,0,-1)), 255).astype(int)

	# add each bounding box to image
	for type, box in zip(classes, boxes):
		x_min = int(box[0])
		y_min = int(box[1])
		x_max = int(box[2])
		y_max = int(box[3])
		image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
		cv2.putText(image, str(type), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 1)

	cv2.imwrite(image_name, image)

