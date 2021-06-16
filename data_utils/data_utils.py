import pandas as pd
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def split_dataframe_for_training_validation_testing(folder_label_path):
    '''
    Base on folder label, create 3 dataframe one for training, one for validation and one for testing 

    :param FOLDER_LABEL_PATH: Folder contain label 

    :return: 
        3 dataframe for training, validation and testing
    '''  #
    # Prepare dataframe for training and validation
    df_train_val = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))
    # Split training dataframe to training and validation
    list_name = df_train_val['name'].unique()
    list_train, list_val = train_test_split(list_name, test_size=0.14, shuffle=False)
    df_train = df_train_val.set_index('name').loc[list_train[:]].reset_index()
    df_val = df_train_val.set_index('name').loc[list_val[:]].reset_index()
    # Prepare dataframe for testing
    df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))

    return (df_train, df_val, df_test)


def load_list_information_from_dataframe(dataframe, folder_image_path, label_off_set=1, norm=True):
    '''
    Load data from dataframe to lists data

    :param dataframe: dataframe contain image name and label 
    :param folder_image_path: path of folder contain image 
    :param label_off_set: shift all the id_category to certain index 

    :return
        list_image_path: list of image path 
        list_classes: list of classes of all bbox in each image
        list_boxes: list of bounding boxes in each image   
    '''  #

    ## Clean Dataframe
    # clean error bounding box (area <= 50)
    dataframe['area'] = (dataframe['x2']-dataframe['x1'])*(dataframe['y2']-dataframe['y1'])
    dataframe = dataframe[dataframe['area']>50].reset_index(drop=True)
    # convert image name to path to that image
    dataframe['name'] = dataframe['name'].apply(lambda name: os.path.join(folder_image_path, name))
    if norm:
        # normalize length of each edge in bounding box
        dataframe['x1'] = dataframe['x1'] / dataframe['width']
        dataframe['x2'] = dataframe['x2'] / dataframe['width']
        dataframe['y1'] = dataframe['y1'] / dataframe['height']
        dataframe['y2'] = dataframe['y2'] / dataframe['height']
    # create new column contain list of coordinate of each bounding box
    dataframe['bbox'] = dataframe.iloc[:][['x1', 'y1', 'x2', 'y2']].values.tolist()
    # shift id_category to left 1 to start at 0
    dataframe['id_category'] = dataframe['id_category'] - label_off_set

    ## Load data from dataframe to list
    # group all image name into one (group bounding box in same image)
    list_image_path = dataframe.groupby(['name'])['name'].apply(list)
    list_image_path = list_image_path.index.tolist()
    # group all id category by specific image into one list
    list_classes = dataframe.groupby(['name'])['id_category'].apply(list)
    list_classes = list_classes.values.tolist()
    # group all bounding box by specific image into one list
    list_boxes = dataframe.groupby(['name'])['bbox'].apply(list)
    list_boxes = list_boxes.values.tolist()

    return (list_image_path, list_boxes, list_classes)


if __name__ == "__main__":
    pass