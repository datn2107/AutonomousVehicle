import cv2
import json

import numpy
import numpy as np
from typing import List, Union

BBox = List[Union[int, float]]


def plot_detection(image: numpy.array = None, image_path: str = None, boxes: List[BBox] = None,
                   classes: List[int] = None, image_type: str = None) -> np.array:
    category = {1: "light", 2: "sign", 3: "car", 4: "pedestrian", 5: "bus", 6: "truck", 7: "rider", 8: "bicycle",
                9: "motorcycle", 10: "train", 11: "other", 12: "other", 13: "trailer"}

    if image_path != None:
        image = cv2.imread(image_path)
    else:
        if image.shape[0] == 3:
            # convert to cv2 format if it in PIL format
            image = np.multiply(np.array(np.moveaxis(image, 0, -1)), 255)

    for type, box in zip(classes, boxes):
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        image = cv2.putText(image, category[type], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if image_type != None:
        print(image_type)
        image = cv2.putText(image, image_type, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)

    return image


def plot_image(image: numpy.array, image_name: str = "test.png"):
    print(image_name)
    cv2.imwrite(image_name, image)
