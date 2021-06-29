import pandas as pd
import tensorflow as tf

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from data_utils.data_utils import load_list_information_from_dataframe


def load_image(height, width):
    ''' Wrapping function to add (height, width) parameter in function when using tf.data.Dataset.map() '''  #

    def load_image(path):
        '''
        Convert JPEG or JPG to Tensor 

        :param path: Path of image 

        :return 
            image: Image Tensor
        '''  #
        # read image
        image = tf.io.read_file(path)
        # decode image to unit8 tensor
        image = tf.image.decode_jpeg(image)
        # convert to tf.float32
        image = tf.cast(image, dtype=tf.float32)
        # resize it if necessary
        image = tf.image.resize(image, (height, width))

        return image

    return load_image


def load_dataset_from_list_image_path(list_image_path, height, width, batch_size=16):
    '''
    From image path list load data and create dataset for it  


    :param image_path_list: list of image path 
    :param batch_size: batch_size for dataset

    :return: 
        dataset: contain image tensor of each image in list (separated by batch size)
    '''  #
    list_image_path = tf.data.Dataset.from_tensor_slices(list_image_path)
    image_dateset = list_image_path.map(load_image(height, width), num_parallel_calls=tf.data.AUTOTUNE)
    image_dateset = image_dateset.batch(batch_size=batch_size, drop_remainder=True)
    image_dateset = image_dateset.prefetch(tf.data.AUTOTUNE)

    return image_dateset


def convert_to_one_hot(classes_list, num_class):
    '''
    Convert classes to one-hot tensor (only one element equal 1 one-hot)

    :param classes_list: list of bounding box's annotation
    :param num_class: number of class (type of annotation)

    :return:
        one-hot tensor create through classes_list
    '''  #
    classes = tf.one_hot(classes_list, num_class)
    classes = tf.cast(classes, dtype=tf.float32)

    return classes


def load_data_from_created_list(list_image_path, list_bboxes, list_classes, num_class, height, width, batch_size=16):
    '''
    Convert all list take from csv to dataset and list tensor

    :param list_image_path: list of image path 
    :param list_classes: list contain list of class for each bounding box in each image  
    :param list_bboxes: list of bounding box in each image 
    :param num_class: number class for create one-hot tensor
    :param batch_size: batch size for split data in image dataset

    :return: 
        image_dateset: dataset contain image tensor of each image (split by batch size)
        list_classes: list of one_hot tensor (shape = [num_box, num_class]) 
        list_bboxes: list contain a tensor for list of bounding box (shape = [num_box, 4])
    '''  #
    image_dateset = load_dataset_from_list_image_path(list_image_path, height=height, width=width,
                                                      batch_size=batch_size)
    list_classes = list(map(convert_to_one_hot, list_classes, len(list_classes) * [num_class]))
    list_bboxes = list(map(tf.constant, list_bboxes))

    return (image_dateset, list_bboxes, list_classes)


def load_data_from_dataframe(dataframe, folder_image_path, num_class, height, width, batch_size):
    '''
    Load data from dataframe to image dataset and list of bounding box and their class

    :param dataframe: Dataframe contain name image, coordinate of bounding box and class of these bounding box 
    :param folder_image_path: The path of folder contain images which their name contain in dataframe 
    :param num_class: Number of category to classify object in each bounding box 
    :param height: Height of image will be resized 
    :param width: Width of image will be resized 
    :param batch_size: Size of each batch which image dataset will be split into

    :return:
        image_dataset: Dataset contain image that split into batch
        list_bboxes: List contain bounding boxes in each image 
        list_claases: List contain classes of each bounding box in each image 
    '''  #
    (list_image_path, list_bboxes, list_classes) = load_list_information_from_dataframe(dataframe, folder_image_path)
    (image_dataset, list_bboxes, list_classes) = load_data_from_created_list(list_image_path,
                                                                             list_bboxes, list_classes,
                                                                             num_class=num_class,
                                                                             height=height, width=width,
                                                                             batch_size=batch_size)

    return (image_dataset, list_bboxes, list_classes)


if __name__ == "__main__":
    pass