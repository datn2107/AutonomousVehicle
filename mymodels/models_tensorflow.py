import tensorflow as tf

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from models.research.object_detection.utils import config_util
from models.research.object_detection.builders import model_builder


def load_model_from_config(model_config_path, num_class):
    # load the configuration file into a dictionary
    configs = config_util.get_configs_from_pipeline_file(model_config_path)
    # get model from configs
    model_config = configs['model']
    # setup model config
    model_config.ssd.num_classes = num_class
    model_config.ssd.freeze_batchnorm = True
    # build model from that config
    model = model_builder.build(model_config=model_config, is_training=True)

    return model


def load_checkpoint_for_model(model, checkpoint_path, batch_size, initiation_model=True):
    ## Load checkpoint of necessary part
    if initiation_model:
        # only load box_predictor and feature_extractor part
        tmp_box_predictor_checkpoint = tf.train.Checkpoint(_base_tower_layers_for_heads=model._box_predictor._base_tower_layers_for_heads,
                                                           _box_prediction_head=model._box_predictor._box_prediction_head)
        tmp_model_checkpoint = tf.train.Checkpoint(_feature_extractor=model._feature_extractor,
                                                   _box_predictor=tmp_box_predictor_checkpoint)
        checkpoint = tf.train.Checkpoint(model=tmp_model_checkpoint)
    else:
        # load all part of model
        checkpoint = tf.train.Checkpoint(model=model)
    # restore checkpoint for model
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

    ## Pass dummy matrix to model for loading weight
    tmp_image, tmp_shapes = model.preprocess(tf.zeros([batch_size, 640, 640, 3]))
    tmp_prediction_dict = model.predict(tmp_image, tmp_shapes)
    tmp_detections = model.postprocess(tmp_prediction_dict, tmp_shapes)

    return model


def define_fine_tune_list(model):
    to_fine_tune = []
    prefixes_to_train = ['WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead',
                         'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead']

    for layer in model.trainable_variables:
        if any(layer.name.startswith(prefix) for prefix in prefixes_to_train):
            to_fine_tune.append(layer)

    return to_fine_tune


@tf.function(experimental_relax_shapes=True)
def train_step_fn(images_batch,
                  groundtruth_boxes_list,
                  groundtruth_classes_list,
                  model,
                  optimizer,
                  to_fine_tune):

    # Provide groundtruth
    model.provide_groundtruth(groundtruth_boxes_list=groundtruth_boxes_list,
                              groundtruth_classes_list=groundtruth_classes_list)
    true_shape_tensor = tf.constant(images_batch.shape[0] * [[640, 640, 3]], dtype=tf.int32)

    with tf.GradientTape() as tape:
        # Preprocess the images
        preprocessed_image = model.preprocess(images_batch)[0]
        # Make a prediction
        prediction_dict = model.predict(preprocessed_image, true_shape_tensor)

        # Calculate the total loss (sum of both losses)
        losses_dict = model.loss(prediction_dict, true_shape_tensor)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

        # Calculate the gradients
        gradients = tape.gradient(total_loss, to_fine_tune)
        # Optimize the model's selected variables
        optimizer.apply_gradients(zip(gradients, to_fine_tune))

        model.provide_groundtruth(groundtruth_boxes_list=[],
                                  groundtruth_classes_list=[])

    return total_loss


@tf.function(experimental_relax_shapes=True)
def evaluate_loss(image_batch,
                  groundtruth_boxes_list,
                  groundtruth_classes_list,
                  model):

    true_shape_tensor = tf.constant(1 * [[640, 640, 3]], dtype=tf.int32)

    # Provide groundtruth
    model.provide_groundtruth(groundtruth_boxes_list=groundtruth_boxes_list,
                              groundtruth_classes_list=groundtruth_classes_list)

    # Preprocess the images
    preprocessed_image = model.preprocess(image_batch)[0]
    # Make a prediction
    prediction_dict = model.predict(preprocessed_image, true_shape_tensor)

    # Calculate the total loss (sum of both losses)
    losses_dict = model.loss(prediction_dict, true_shape_tensor)
    total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

    return total_loss