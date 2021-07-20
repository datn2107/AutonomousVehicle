import argparse
import os

import tensorflow as tf
import pandas as pd

# from mymodels.detection_utils_tensorflow import detect
from mymodels.models_tensorflow import SSDModel
from mytrain.train_tensorflow import train_step_fn
from myutils.data_utils_tensorflow import load_dataset


def load_model():
    pass


def split_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def training():
    df_train = pd.read_csv(os.path.join(folder_label_path, 'train.csv'))
    (image_dataset, list_boxes, list_classes) = load_dataset(dataframe=df_train,
                                                             folder_image_path=os.path.join(
                                                                 folder_image_path, 'train'),
                                                             height=height, width=width,
                                                             batch_size=batch_size,
                                                             num_class=num_class)
    list_boxes = split_list(list_boxes, batch_size)
    list_classes = split_list(list_classes, batch_size)

    builder = SSDModel(model_config_path)
    builder.load_model(num_class)
    builder.load_checkpoint(checkpoint_path, height, width, batch_size, initiation_model=True)
    builder.load_optimizer()
    fine_tune_layer = builder.get_fine_tune_layer(train_all=True)
    optimizer = builder.optimizer
    model = builder.model

    # ''' Start Training'''
    print('Start fine-tuning!', flush=True)
    # checkpoint = tf.train.Checkpoint()
    # manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)
    for epoch in range(num_epoch):
        train_loss = 0
        for batch, (image_batch, boxes_batch, class_batch) in enumerate(
                zip(image_dataset, list_boxes, list_classes)):
            total_loss = train_step_fn(image_batch,
                                       boxes_batch,
                                       class_batch,
                                       model,
                                       optimizer,
                                       fine_tune_layer)
            train_loss += total_loss.numpy()
            if batch % 5000 == 0:
                print('batch ' + str(batch)
                      + ', loss = ' + str(train_loss / batch), flush=True)
        # Display loss
        print('epoch ' + str(epoch) + ' of ' + str(num_epoch)
              + ', train_loss=' + str(train_loss / num_batch), flush=True)
        # Save path after each epoch
        # save_path = manager.save()
        # print('Save checkpoint at ' + save_path, flush=True)
    print('Done fine-tuning!')

    return model


# def detection_by_lower_api(model, test_image_dataset):
#     ''' Detection '''
#     for image in test_image_dataset:
#         detections = detect(model, image)
#         detections_boxes = tf.squeeze(detections['detection_boxes']).numpy()
#         detection_scores = tf.squeeze(detections['detection_scores']).numpy()
#
#         list_boxes = []
#         for id in range(detection_scores.shape[0]):
#             list_boxes.append(detections_boxes[id])
#
#         # visualize_detection(tf.squeeze(image).numpy(), list_boxes)
#         break


# def train():
    # df_test = pd.read_csv(os.path.join(folder_label_path, 'test.csv'))
    # (test_image_dataset, test_list_boxes, test_list_classes) = load_dataset(dataframe=df_test,
    #                                                                         folder_image_path=os.path.join(
    #                                                                             folder_image_path, 'test'),
    #                                                                         height=height, width=width,
    #                                                                         batch_size=1,
    #                                                                         num_class=num_class)
    #
    # detection_by_lower_api(model, test_image_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Folder Image Path argument
    parser.add_argument('--images', type=str, help='Folder Image Path')
    parser.set_defaults(images=r'D:\Machine Learning Project\Autonomous Driving\Data\Object Detection\images')
    # Folder Label Path argument
    parser.add_argument('--labels', type=str, help='Folder Label Path')
    parser.set_defaults(labels=r'D:\Machine Learning Project\Autonomous Driving\Data\Object Detection\labels')
    # Batch Size argument
    parser.add_argument('--batch', type=int, help='Batch size to split image dataset')
    parser.set_defaults(batch=8)
    # Model config path
    parser.add_argument('--config', type=str, help='Path to model config')
    parser.set_defaults(
        config=r'D:\Machine Learning Project\Autonomous Driving\efficientdet_d4_coco17_tpu-32\pipeline.config')
    # Model checkpoint path
    parser.add_argument('--checkpoint', type=str, help='Number class of each object')
    parser.set_defaults(
        checkpoint=r'D:\Machine Learning Project\Autonomous Driving\efficientdet_d4_coco17_tpu-32\checkpoint')

    args = parser.parse_args()
    folder_image_path = args.images
    folder_label_path = args.labels
    batch_size = args.batch
    model_config_path = args.config
    checkpoint_path = args.checkpoint

    height = 800
    width = 1333
    num_class = 13
    num_epoch = 30

    training()
