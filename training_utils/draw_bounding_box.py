# from google.colab.patches import cv2_imshow
import cv2
import numpy as np

def visualize_detection(image_path, boxes):
	image = cv2.imread(image_path)

	for box in boxes:
		w, h = (image.shape[1], image.shape[0])
		x_min = int(box[0]*w)
		y_min = int(box[1]*h)
		x_max = int(box[2]*w)
		y_max = int(box[3]*h)
		image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,0,255), 1)

	#cv2_imshow(image)



