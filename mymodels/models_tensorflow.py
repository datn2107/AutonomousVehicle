import os
import sys

import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from models.research.object_detection.utils import config_util
from models.research.object_detection.builders import model_builder
from models.research.object_detection.builders import optimizer_builder


class SSDModel():
    def __init__(self, model_config_path):
        self.configs = config_util.get_configs_from_pipeline_file(model_config_path)
        self.optimizer = None
        self.model = None

    def load_model(self, num_class):
        model_config = self.configs['model']
        model_config.ssd.num_classes = num_class
        model_config.ssd.freeze_batchnorm = True
        self.model = model_builder.build(model_config=model_config, is_training=True)

    def load_optimizer(self):
        training_config = self.configs['train_config']
        optimizer_config = training_config.optimizer
        self.optimizer = optimizer_builder.build(optimizer_config)[0]

    def load_checkpoint(self, checkpoint_path, height, width, batch_size):
        if (tf.train.latest_checkpoint(checkpoint_path).split("-")[-1]) == 0:
            # only load box_predictor and feature_extractor part
            tmp_box_predictor_checkpoint = tf.train.Checkpoint(
                _base_tower_layers_for_heads=self.model._box_predictor._base_tower_layers_for_heads,
                _box_prediction_head=self.model._box_predictor._box_prediction_head)
            tmp_model_checkpoint = tf.train.Checkpoint(_feature_extractor=self.model._feature_extractor,
                                                       _box_predictor=tmp_box_predictor_checkpoint)
            checkpoint = tf.train.Checkpoint(model=tmp_model_checkpoint)
        else:
            checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

        tmp_image, tmp_shapes = self.model.preprocess(tf.zeros([batch_size, height, width, 3]))
        tmp_prediction_dict = self.model.predict(tmp_image, tmp_shapes)
        _ = self.model.postprocess(tmp_prediction_dict, tmp_shapes)

    def get_fine_tune_layer(self, train_all=False):
        to_fine_tune = []
        if train_all:
            prefixes_to_train = ['WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead',
                                 'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead']
            for layer in self.model.trainable_variables:
                if any(layer.name.startswith(prefix) for prefix in prefixes_to_train):
                    to_fine_tune.append(layer)
        else:
            to_fine_tune = self.model.trainable_variables

        return to_fine_tune
