import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import pandas
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


def split_dataframe_for_training_validation_testing(folder_label_path: str, shuffle: bool = True) \
        -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
    Base on folder label contain 2 csv file for training and testing create 3 dataframe for training, validation and testing

    Args: 
        folder_label_path :str: Path to folder contain 2 csv file for training and testing (train.csv and test.csv)
        shuffle :bool: If True shuffle infomation in training and validation dataframe
    Return:
        df_train, df_val, df_test - 3 dataframe contain information of training, validation and testing 
    """ #
    # Split training dataframe to training and validation by train.csv
    df_train_val = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))
    list_name = df_train_val['name'].unique()
    list_train, list_val = train_test_split(list_name, test_size=0.14, shuffle=shuffle)
    df_train = df_train_val.set_index('name').loc[list_train[:]].reset_index()
    df_val = df_train_val.set_index('name').loc[list_val[:]].reset_index()
    # Prepare dataframe for testing by test.csv
    df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))

    return df_train, df_val, df_test


def load_data_from_dataframe_to_list(dataframe: pandas.DataFrame, folder_image_path: str,
                                     label_off_set: int = 1, norm: bool = True)\
                                     -> Tuple[list, list, list]:
    """
    Load data from dataframe to three lists data
    
    Args:
        dataframe :pandas.DataFrame: Each row of dataframe contain one bounding box, 
                                      it contain ['name'(image_name), 'x1', 'y1', 'x2', 'y2', 'id_category', 'height', 'width']
        folder_image_path :str: Path of folder contain image 
        label_off_set :int: Shift all the id_category to certain index
        norm :bool: If True coordinates of bounding box will be normalized 
    Returns:   
        (list_image_path, list_boxes, list_classes) - Tuple contain 3 lists of information 
    """ #
    ## Clean Dataframe
    # convert image name to path to that image
    dataframe['name'] = dataframe['name'].apply(lambda name: os.path.join(folder_image_path, name))
    if norm:
        # normalize coordinate of each bounding box
        dataframe['x1'] = dataframe['x1'] / dataframe['width']
        dataframe['x2'] = dataframe['x2'] / dataframe['width']
        dataframe['y1'] = dataframe['y1'] / dataframe['height']
        dataframe['y2'] = dataframe['y2'] / dataframe['height']
    # create new column contain list of coordinate of each bounding box
    dataframe['bbox'] = dataframe.iloc[:][['x1', 'y1', 'x2', 'y2']].values.tolist()
    # shift id_category the left label_off_set position
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

    return list_image_path, list_boxes, list_classes


def clean_error_bounding_box_in_datafrane(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    # Clean error bounding box (area <= 100)
    dataframe = dataframe[(dataframe['x2']-dataframe['x1'])*(dataframe['y2']-dataframe['y1'])>100].reset_index(drop=True)
    return dataframe

if __name__ == "__main__":
    pass