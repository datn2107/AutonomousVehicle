import os

import tensorflow as tf
import pandas as pd
import argparse

from utils.data_utils_tensorflow import load_data_from_dataframe
from model.training_utils_tensorflow import load_model_from_config
from model.training_utils_tensorflow import load_checkpoint_for_model
from model.training_utils_tensorflow import define_fine_tune_list
from model.detection_utils_tensorflow import detect
from model.detection_utils_tensorflow import visualize_detection


def training_by_lower_api(train_image_dataset, train_list_boxes, train_list_classes):
    ''' Prepare model '''
    model = load_model_from_config(model_config_path, num_class)
    model = load_checkpoint_for_model(model, checkpoint_path, batch_size, initiation_model=initiation_model)
    to_fine_tune = define_fine_tune_list(model)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    # ''' Start Training'''
    # print('Start fine-tuning!', flush=True)
    # checkpoint = tf.train.Checkpoint()
    # manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)
    # for epoch in range(num_epoch):
    #     train_loss = 0
    #     num_batch = 0
    #     for image_batch in train_image_dataset:
    #         # Get the ground truth
    #         groundtruth_boxes_list = [train_list_boxes[num_batch*batch_size+id] for id in range(batch_size)]
    #         groundtruth_classes_list = [train_list_classes[num_batch*batch_size+id] for id in range(batch_size)]
    #         num_batch = num_batch + 1
    #         # Training step (forward pass + backwards pass)
    #         total_loss = train_step_fn(image_batch,
    #                                    groundtruth_boxes_list,
    #                                    groundtruth_classes_list,
    #                                    model,
    #                                    optimizer,
    #                                    to_fine_tune)
    #         # Sum the losses
    #         train_loss += total_loss.numpy()
    #         if num_batch % 5000 == 0:
    #             print('batch ' + str(num_batch)
    #                   + ', loss = ' + str(train_loss / num_batch), flush=True)
    #     # Display loss
    #     print('epoch ' + str(epoch) + ' of ' + str(num_epoch)
    #           + ', train_loss=' + str(train_loss / num_batch), flush=True)
    #     # Save path after each epoch
    #     save_path = manager.save()
    #     print('Save checkpoint at ' + save_path, flush=True)
    # print('Done fine-tuning!')

    return model

def detection_by_lower_api(model, test_image_dataset):
    ''' Detection '''
    for image in test_image_dataset:
        detections = detect(model, image)
        detections_boxes = tf.squeeze(detections['detection_boxes']).numpy()
        detection_scores = tf.squeeze(detections['detection_scores']).numpy()

        list_boxes = []
        for id in range(detection_scores.shape[0]):
            list_boxes.append(detections_boxes[id])

        visualize_detection(tf.squeeze(image).numpy(), list_boxes)
        break


def main():
    ''' Prepare Data '''
    ## Load data from dataframe for training, validation and testing
    #(df_train, df_val, df_test) = split_dataframe_for_training_validation_testing(folder_label_path)
    df_train = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))
    df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))
    (train_image_dataset, train_list_boxes, train_list_classes) = load_data_from_dataframe(dataframe=df_train,
                                                                                           folder_image_path=os.path.join(folder_image_path, 'train'),
                                                                                           height=height, width=width,
                                                                                           batch_size=batch_size,
                                                                                           num_class=num_class)
    # (val_image_dataset, val_list_boxes, val_list_classes) = load_data_from_dataframe(dataframe=df_val,
    #                                                                                  folder_image_path=os.path.join(folder_image_path, 'train'),
    #                                                                                  height=height, width=width,
    #                                                                                  batch_size=batch_size,
    #                                                                                  num_class=num_class)
    (test_image_dataset, test_list_boxes, test_list_classes) = load_data_from_dataframe(dataframe=df_test,
                                                                                        folder_image_path=os.path.join(folder_image_path, 'test'),
                                                                                        height=height, width=width,
                                                                                        batch_size=1,
                                                                                        num_class=num_class)

    '''Training'''
    model = training_by_lower_api(train_image_dataset, train_list_boxes, train_list_classes)

    '''Detection'''
    detection_by_lower_api(model, test_image_dataset)



if __name__ == "__main__":
    ''' Create args to feed argument from terminal '''
    parser = argparse.ArgumentParser()
    # Folder Image Path argument
    parser.add_argument('--fip', type=str, help='Folder Image Path')
    parser.set_defaults(fip=r'D:\Autonomous Driving\Data\Object Detection\image')
    # Folder Label Path argument
    parser.add_argument('--flp', type=str, help='Folder Label Path')
    parser.set_defaults(flp=r'D:\Autonomous Driving\Data\Object Detection\label')
    # Batch Size argument
    parser.add_argument('--bs', type=int, help='Batch size to split image dataset')
    parser.set_defaults(bs=8)
    # Model config path
    parser.add_argument('--mcp', type=str, help='Path to model config')
    parser.set_defaults(mcp=r'D:\Autonomous Driving\SourceCode\models\research\object_detection\configs\tf2\ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config')
    # Model checkpoint path
    parser.add_argument('--cp', type=str, help='Number class of each object')
    parser.set_defaults(cp=r'D:\Autonomous Driving\SourceCode\checkpoint_ssd_resnet50_tensorflow')

    ''' Take the values from args '''
    args = parser.parse_args()
    folder_image_path = args.fip
    folder_label_path = args.flp
    model_config_path = args.mcp
    checkpoint_path = args.cp
    batch_size = args.bs

    '''Initialize for essensial variable'''
    height = 640
    width = 640
    num_class = 13
    learning_rate = 0.01
    num_epoch = 30
    initiation_model = True

    main()
