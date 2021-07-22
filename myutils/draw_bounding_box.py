import cv2
import json

import numpy
import numpy as np
from typing import List, Union

BBox = List[Union[int, float]]


def plot_detection(image: numpy.array = None, image_path: str = None, boxes: List[BBox] = None,
                   classes: List[int] = None, image_type: str = None) -> np.array:
    category = {0: "light", 1: "sign", 2: "car", 3: "pedestrian", 4: "bus", 5: "truck", 6: "rider", 7: "bicycle",
                8: "motorcycle", 9: "train", 10: "other", 11: "other", 12: "trailer"}

    if image_path != None:
        image = cv2.imread(image_path)
    else:
        if image.shape[0] == 3:
            # convert to cv2 format if it in PIL format
            image = np.array(np.moveaxis(image, 0, -1))
        if isinstance(image.item(0), int):
            image = np.multiply(image, 255).astype(int)


    height, width = image.shape[0], image.shape[1]
    for label, box in zip(classes, boxes):
        if all(x <= 1. for x in box):
            box[0], box[2] = box[0]*width, box[2]*width
            box[1], box[3] = box[1]*height, box[3]*height
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        image = cv2.putText(image, category[int(label)], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if image_type != None:
        image = cv2.putText(image, image_type, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)

    return image


def plot_image(image: numpy.array, image_name: str = "test.png"):
    print(image_name)
    cv2.imwrite(image_name, image)
