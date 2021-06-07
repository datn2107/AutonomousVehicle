import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from data_utils.Tensorflow.load_dataset import load_data_from_dataframe
from data_utils.data_utils import split_dataframe_for_training_validation_testing
from training_utils import load_model_from_config
from training_utils import load_checkpoint_for_model
from training_utils import define_fine_tune_list
from training_utils import train_step_fn
from training_utils import evaluate_loss


''' Create args to feed argument from terminal '''
parser = argparse.ArgumentParser()
# Folder Image Path argument
parser.add_argument('--fip', type=str, help='Folder Image Path')
# Folder Label Path argument
parser.add_argument('--flp', type=str, help='Folder Label Path')
# Batch Size argument
parser.add_argument('--bs', type=int, help='Batch size to split image dataset')
# Model config path
parser.add_argument('--mcp', type=str, help='Path to model config')
# Model checkpoint path
parser.add_argument('--cp', type=str, help='Number class of each object')


def main_tensorflow():

    ''' Take the values from args '''
    args = parser.parse_args()
    folder_image_path = args.fip
    folder_label_path = args.flp
    #height = args.h
    #width = args.w
    batch_size = args.bs
    #num_class = args.nc
    model_config_path = args.mcp
    checkpoint_path = args.cp
    #learning_rate = args.lr
    #num_epoch = args.ne

    #folder_image_path = r'D:\Autonomous Driving\Data\Object Detection\image'
    #folder_label_path = r'D:\Autonomous Driving\Data\Object Detection\label'
    height = 640
    width = 640
    batch_size = 8
    num_class = 13
    #model_config_path = r'D:\Autonomous Driving\SourceCode\models\research\object_detection\configs\tf2\ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
    #checkpoint_path = r'D:\Autonomous Driving\SourceCode\checkpoint\ckpt-37'
    learning_rate = 0.01
    num_epoch = 100

    ''' Prepare Data '''
    ## Load data from dataframe for training, validation and testing
    #(df_train, df_val, df_test) = split_dataframe_for_training_validation_testing(folder_label_path)
    df_train = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))
    df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))

    (train_image_dataset, train_list_bboxes, train_list_classes) = load_data_from_dataframe(dataframe=df_train,
                                                                                            folder_image_path=os.path.join(folder_image_path, 'train'),
                                                                                            height=height, width=width,
                                                                                            batch_size=batch_size,
                                                                                            num_class=num_class)

    #(val_image_dataset, val_list_bboxes, val_list_classes) = load_data_from_dataframe(dataframe=df_val,
    #                                                                                  folder_image_path=os.path.join(folder_image_path, 'train'),
    #                                                                                  height=height, width=width,
    #                                                                                  batch_size=batch_size,
    #                                                                                  num_class=num_class)

    (test_image_dataset, test_list_bboxes, test_list_classes) = load_data_from_dataframe(dataframe=df_test,
                                                                                         folder_image_path=os.path.join(folder_image_path, 'test'),
                                                                                         height=height, width=width,
                                                                                         batch_size=batch_size,
                                                                                         num_class=num_class)

    ''' Prepare model '''
    model = load_model_from_config(model_config_path, num_class)
    model = load_checkpoint_for_model(model, checkpoint_path, first_time=False)
    to_fine_tune = define_fine_tune_list(model)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)


    ''' Start Training'''
    print('Start fine-tuning!', flush=True)

    checkpoint = tf.train.Checkpoint(root=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)
    for epoch in range(num_epoch):
        train_loss = 0
        num_batch = 0
        for image_batch in train_image_dataset:
            # Get the ground truth
            groundtruth_boxes_list = [train_list_bboxes[num_batch*batch_size+id] for id in range(batch_size)]
            groundtruth_classes_list = [train_list_classes[num_batch*batch_size+id] for id in range(batch_size)]
            num_batch = num_batch + 1

            # Training step (forward pass + backwards pass)
            total_loss = train_step_fn(image_batch,
                                       groundtruth_boxes_list,
                                       groundtruth_classes_list,
                                       model,
                                       optimizer,
                                       to_fine_tune)

            train_loss += total_loss.numpy()


        print('epoch ' + str(epoch) + ' of ' + str(num_epoch)
              + ', train_loss=' + str(train_loss/num_batch), flush=True)

        save_path = manager.save()
        print('Save checkpoint at ' + save_path, flush=True)

    print('Done fine-tuning!')


if __name__ == "__main__":
    main_tensorflow()
