import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_detection(boxes, image=None, image_path=None):
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
			image = np.multiply(np.array(image.moveaxis(image,0,-1)), 255).astype(int)

	# add each bounding box to image
	for box in boxes:
		x_min = int(box[0])
		y_min = int(box[1])
		x_max = int(box[2])
		y_max = int(box[3])
		image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

	cv2.imwrite('test.jpg', image)

