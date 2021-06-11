import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np


@tf.function(experimental_relax_shapes=True)
def detect(model, image_tensor):
	"""Run detection on an input image.

	:param
		image_tensor: Tensor of image you want to detect (shape = [1, height, width, 3])

	:return:
		A dict containing 3 Tensors: detection_boxes, detection_classes, detection_scores.
	""" #
	preprocessed_image, shapes = model.preprocess(image_tensor)
	prediction_dict = model.predict(preprocessed_image, shapes)

	# use the detection model's postprocess() method to get the the final detections
	detections = model.postprocess(prediction_dict, shapes)

	return detections
