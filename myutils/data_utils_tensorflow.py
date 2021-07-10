import pandas as pd
import tensorflow as tf

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from data_utils import load_list_data


def load_image(height, width):
    def load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, dtype=tf.float32)
        image = tf.image.resize(image, (height, width))

        return image

    return load_image

def convert_to_one_hot(list_class, num_class):
    classes_tensor = tf.one_hot(list_class, num_class)
    classes_tensor = tf.cast(classes_tensor, dtype=tf.float32)

    return classes_tensor

def load_image_dataset(list_image_path, height, width, batch_size=16):
    list_image_path = tf.data.Dataset.from_tensor_slices(list_image_path)
    image_dateset = list_image_path.map(load_image(height, width), num_parallel_calls=tf.data.AUTOTUNE)
    image_dateset = image_dateset.batch(batch_size=batch_size, drop_remainder=True)
    image_dateset = image_dateset.prefetch(tf.data.AUTOTUNE)

    return image_dateset


def convert_to_tensor(list_image_path, list_bboxes, list_classes, height, width, num_class=13, batch_size=16):
    image_dateset = load_image_dataset(list_image_path, height=height, width=width,
                                       batch_size=batch_size)
    list_classes = list(map(convert_to_one_hot, list_classes, len(list_classes) * [num_class]))
    list_bboxes = list(map(tf.constant, list_bboxes))

    return (image_dateset, list_bboxes, list_classes)


def load_dataset(dataframe, folder_image_path, num_class, height, width, batch_size):
    (list_image_path, list_bboxes, list_classes) = load_list_data(dataframe, folder_image_path)
    (image_dataset, list_bboxes, list_classes) = convert_to_tensor(list_image_path,
                                                                   list_bboxes, list_classes,
                                                                   height=height, width=width,
                                                                   num_class=num_class,
                                                                   batch_size=batch_size)

    return (image_dataset, list_bboxes, list_classes)


if __name__ == "__main__":
    pass