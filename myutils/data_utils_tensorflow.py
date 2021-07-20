import os
import sys
from typing import List, Union

import pandas as pd
import tensorflow as tf

from .data_utils import load_list_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# def load_image(height, width):
#     def load_image(path):
#         image = tf.io.read_file(path)
#         image = tf.image.decode_jpeg(image, channels=3)
#         image = tf.cast(image, dtype=tf.float32)
#         image = tf.image.resize(image, (height, width))
#
#         return image
#
#     return load_image


BBox = List[Union[float]]


def convert_to_one_hot(list_class: List[int], num_class: int):
    classes_tensor = tf.one_hot(list_class, num_class)
    classes_tensor = tf.cast(classes_tensor, dtype=tf.float32)

    return classes_tensor


def load_image_dataset(list_image_path: List[str], height: int, width: int, batch_size: int = 16,
                       norm_image: bool = False):
    def load_image(path: str):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, dtype=tf.float32)
        image = tf.image.resize(image, (height, width))
        if (norm_image):
            tf.image.per_image_standardization(image)

        return image

    list_image_path = tf.data.Dataset.from_tensor_slices(list_image_path)
    image_dateset = list_image_path.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    image_dateset = image_dateset.batch(batch_size=batch_size)
    image_dateset = image_dateset.prefetch(tf.data.AUTOTUNE)

    return image_dateset


def load_dataset(dataframe: pd.DataFrame, folder_image_path: str, num_class: int, height: int, width: int,
                 batch_size: int, norm_box: bool = True, norm_image: bool = False):
    (list_image_path, list_boxes, list_classes) = load_list_data(dataframe, folder_image_path, label_off_set=1,
                                                                 norm_box=norm_box)

    image_dateset = load_image_dataset(list_image_path, height=height, width=width,
                                       batch_size=batch_size, norm_image=norm_image)
    list_boxes = list(map(tf.constant, list_boxes))
    list_classes = list(map(convert_to_one_hot, list_classes, len(list_classes) * [num_class]))

    return (image_dateset, list_boxes, list_classes)
