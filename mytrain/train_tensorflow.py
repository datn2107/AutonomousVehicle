import tensorflow as tf


@tf.function(experimental_relax_shapes=True)
def train_step_fn(images_batch,
                  height, width,
                  groundtruth_boxes_list,
                  groundtruth_classes_list,
                  model,
                  optimizer,
                  to_fine_tune):
    model.provide_groundtruth(groundtruth_boxes_list=groundtruth_boxes_list,
                              groundtruth_classes_list=groundtruth_classes_list)
    true_shape_tensor = tf.constant(images_batch.shape[0] * [[height, width, 3]], dtype=tf.int32)

    with tf.GradientTape() as tape:
        preprocessed_image = model.preprocess(images_batch)[0]
        prediction_dict = model.predict(preprocessed_image, true_shape_tensor)

        losses_dict = model.loss(prediction_dict, true_shape_tensor)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

        gradients = tape.gradient(total_loss, to_fine_tune)
        optimizer.apply_gradients(zip(gradients, to_fine_tune))

        model.provide_groundtruth(groundtruth_boxes_list=[],
                                  groundtruth_classes_list=[])

    return total_loss


@tf.function(experimental_relax_shapes=True)
def evaluate_loss(image_batch,
                  height, width,
                  groundtruth_boxes_list,
                  groundtruth_classes_list,
                  model):
    true_shape_tensor = tf.constant(1 * [[1024, 1024, 3]], dtype=tf.int32)

    model.provide_groundtruth(groundtruth_boxes_list=groundtruth_boxes_list,
                              groundtruth_classes_list=groundtruth_classes_list)

    preprocessed_image = model.preprocess(image_batch)[0]
    prediction_dict = model.predict(preprocessed_image, true_shape_tensor)

    losses_dict = model.loss(prediction_dict, true_shape_tensor)
    total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

    return total_loss

@tf.function
def detect(model, input_tensor):
    preprocessed_image, shapes = model.preprocess(input_tensor)
    prediction_dict = model.predict(preprocessed_image, shapes)

    detections = model.postprocess(prediction_dict, shapes)

    return detections