# Copyright 2018-2020 Ivan Alles. See also the LICENSE file.

import enum
import json
import os

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

from localizer import utils


class Object:
    """ A predicted object. Duck-type compatible with dataset.Object. """
    def __init__(self, x, y, angle, category, confidence):
        self.origin = np.array([x, y], dtype=np.float32)
        self.angle = angle  # Angle in radiance
        self.category = int(category)
        self.confidence = confidence

    def __str__(self):
        return f'{self.origin[0]:.2f} {self.origin[1]:.2f} {np.rad2deg(self.angle):.2f} ' \
               f'{self.category} {self.confidence:.2f}'


class TrainingModelChannels(enum.IntEnum):
    """ Indices of output channels. """
    Y = 0  # y-coordinate.
    X = 1  # x-coordinate.
    SA = 2  # sin(angle).
    CA = 3  # cos(angle).
    COUNT = 4  # Number of channels.


class PredictionModelOutputs(enum.IntEnum):
    """ Indices of prediction model outputs. """
    OBJECTS = 0  # Objects, a 2d array num_objects x (x, y, angle, category, confidence)
    # The rest is used for diagnostics
    MODEL = 1  # Training model output.
    CONFIDENCE = 2  # Confidence level.
    AVERAGE_POS = 3  # Average x, y positions.
    AVERAGE_ANGLE = 4  # Average sin(angle), cos(angle).
    OUTPUT_WINDOW_POS = 5  # Window function output.


class Localizer:
    """
    Predicts object positions, orientations, and categories on an image.
    """

    def __init__(self, config_file_name):
        with open(os.path.join(config_file_name), encoding='utf-8') as f:
            self._cfg = json.load(f)

        self._diag = False
        self._diag_dir = 'localizer_diag'

        self._model_dir = os.path.dirname(config_file_name)
        self._model_path = os.path.join(self._model_dir, 'model.tf')

        self._sigma = self._cfg['sigma']

        self._confidence_ksize = self._cfg['confidence_ksize']
        self._pos_ksize = self._cfg['pos_ksize']
        self._angle_ksize = self._cfg['angle_ksize']

        self._model_scale = float(self._cfg['model_scale'])
        self._input_scale = self._cfg['input_scale']

        self._image_shape = None

    @property
    def cfg(self):
        return self._cfg

    @property
    def diag(self):
        return self._diag

    @diag.setter
    def diag(self, value):
        self._diag = value
        if self._diag:
            os.makedirs(self._diag_dir, exist_ok=True)

    @property
    def diag_dir(self):
        return self._diag_dir

    @diag_dir.setter
    def diag_dir(self, value):
        self._diag_dir = value
        if self._diag:
            os.makedirs(self._diag_dir, exist_ok=True)

    def _update_input_image_parameters(self, image_shape):
        if self._image_shape == image_shape:
            return

        self._image_shape = image_shape

        input_size = tuple(np.ceil(np.array(image_shape[:2]) * self._input_scale).astype(int))

        color_channels = image_shape[2] if len(image_shape) > 2 else 1
        self._model_input_shape = tuple(input_size) + (color_channels,)

        self._create_model()

        # Transform to convert from image to input.
        self._input_t_image = utils.make_transform2(self._input_scale)
        input_t_output = utils.make_transform2(1 / self._model_scale)
        self._image_t_output = np.dot(np.linalg.inv(self._input_t_image), input_t_output)

    @staticmethod
    def _compute_pos_kernels(size, sigma):
        """
        Compute kernels for positions.

        pos = (y, x)
        gaussian = exp(-0.5 * (pos/sigma)**2).
        :param size - size of the kernel. Must be an odd integer.
        :return:  pos tensor (size, size, 2), gaussian tensor (size, size).
        """

        if size % 2 != 1:
            raise ValueError('Kernel size must be odd')

        hs = size // 2
        row = -np.array(range(-hs, hs + 1), dtype=np.float32)
        pos = np.zeros((size, size, 2), dtype=np.float32)
        pos[:, :, 1] = np.broadcast_to(row, (size, size))
        pos[:, :, 0] = pos[:, :, 1].T

        n2 = ((pos / sigma) ** 2).sum(axis=2)
        gaussian = np.exp(-0.5 * n2)
        return pos, gaussian

    def _create_model(self):
        model = keras.models.load_model(self._model_path)

        if self._diag:
            keras.utils.plot_model(model, to_file=os.path.join(self._diag_dir, 'model.svg'), dpi=50, show_shapes=True)

        # Compute model with dummy values to let it compute the output shape.
        dummy_input = np.zeros((1,) + self._model_input_shape)
        self._model_output_shape = model.predict(dummy_input).shape[1:]

        self._category_count = model.output.shape[1]

        output_pos = model.output[:, :, :, :, TrainingModelChannels.Y:TrainingModelChannels.X + 1]
        output_angle = model.output[:, :, :, :, TrainingModelChannels.SA:TrainingModelChannels.CA + 1]

        scale = np.full(2, 1. / self._sigma)

        def make_window(v):
            sr2 = tf.reduce_sum(tf.square(v * scale), axis=-1, keepdims=True)
            gaussian = tf.math.exp(-0.5 * sr2)
            window = v * scale * gaussian
            return window

        output_window_pos = make_window(output_pos)

        def show_kernel(k, name, factor):
            for i in range(k.shape[2]):
                im = cv2.resize(k[:, :, i] * factor, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_AREA)
                cv2.imshow(name + f'.i{i}', utils.red_green(im))
            cv2.waitKey(0)

        p, g = self._compute_pos_kernels(self._confidence_ksize, self._sigma)
        k = p / self._sigma * np.expand_dims(g, 2)
        # Normalize so that the convolution to itself gives 1.
        s = (k * k).sum(axis=(0, 1), keepdims=True) * 2
        if (s == 0).any():
            raise ValueError('Confidence kernel is zero, increase kernel size')
        confidence_kernel = k / s

        if False:  # Test code
            show_kernel(confidence_kernel, 'confidence_kernel', 20)

        confidence_map = tf.nn.conv2d(output_window_pos, np.expand_dims(confidence_kernel, 3),
                                 strides=1,
                                 padding='SAME', name='confidence_map')

        confidence_loc_max = (tf.nn.max_pool(confidence_map, 3, 1, 'SAME', 'NCDHW') - confidence_map) == 0
        confidence_thr = tf.math.logical_and(confidence_loc_max, confidence_map >= self._cfg['confidence_thr'])
        confidence = confidence_map * tf.cast(confidence_thr, tf.float32)

        def make_average_function(output, ksize, name):
            _, k = self._compute_pos_kernels(ksize, self._sigma)
            k /= k.sum()
            if False:  # Test code
                show_kernel(np.expand_dims(k, 2), name + '_kernel', 10)

            kernel = np.zeros((ksize, ksize, 2, 2))
            for i in range(2):
                kernel[:, :, i, i] = k

            average_func = tf.nn.conv2d(output, kernel,
                                        strides=1,
                                        padding='SAME',
                                        name=name)
            return average_func

        average_pos = make_average_function(output_pos, self._pos_ksize, name='average_pos')
        average_angle = make_average_function(output_angle, self._angle_ksize, name='average_angle')

        confidence = tf.squeeze(confidence, -1)
        local_max_idx = tf.where(confidence)
        category = tf.cast(local_max_idx[:, 1], tf.float32)
        yx = tf.gather_nd(average_pos, local_max_idx) + tf.cast(local_max_idx[:, 2:], tf.float32)
        sc = tf.gather_nd(average_angle, local_max_idx)
        angle = tf.math.atan2(sc[:, 0], sc[:, 1])
        conf = tf.gather_nd(confidence, local_max_idx)
        objects = tf.stack([yx[:, 1], yx[:, 0], angle, category, conf], 1)

        # Add outputs in the order defined by PredictionModelOutputs.
        all_outputs = [objects]
        if self._diag:
            all_outputs += [model.output, confidence, average_pos, average_angle, output_window_pos]
        self._model = keras.Model(inputs=model.input, outputs=all_outputs)

        # This stopped working with TF 2.4.
        # if self._diag:
        #     tf.keras.utils.plot_model(self._model, to_file=os.path.join(self._diag_dir, 'prediction_model.svg'),
        #                               dpi=50, show_shapes=True)

        if self._cfg.get('save_prediction_model', False):
            self._model.save(os.path.join(self._model_dir, 'prediction_model.tf'))

    def predict(self, image):
        """
        param: image. Must have the correct number of input channels (RGB or grayscale).
        return: a list of predicted objects.
        """

        image = image.astype(np.float32)

        def pad_size(size, pad_to):
            return pad_to - size % pad_to if size % pad_to > 0 else 0

        pad_to = self._cfg.get('pad_to', 0)
        if pad_to:
            image = np.pad(image, (
                (0, pad_size(image.shape[0], pad_to)),
                (0, pad_size(image.shape[1], pad_to)),
                (0, 0)))

        self._update_input_image_parameters(image.shape)

        batch_size = 1

        input = cv2.warpAffine(image, self._input_t_image[:2, :],
                               (self._model_input_shape[1::-1]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        class Batch:
            pass

        batch = Batch()
        batch.inputs = np.expand_dims(input, 0)

        result = self._model.predict(batch.inputs)

        if self._diag:
            batch.output_on_image = np.zeros((batch_size, self._category_count, 2) + image.shape)

            for bi in range(batch_size):
                confidence = result[PredictionModelOutputs.CONFIDENCE][bi]
                average_pos = result[PredictionModelOutputs.AVERAGE_POS][bi]
                average_angle = result[PredictionModelOutputs.AVERAGE_ANGLE][bi]

                if self._diag:
                    batch.outputs = np.expand_dims(result[PredictionModelOutputs.MODEL][bi], 0)
                    batch.output_window_pos = np.expand_dims(result[PredictionModelOutputs.OUTPUT_WINDOW_POS][bi], 0)
                    batch.confidence = np.expand_dims(confidence, 0)
                    batch.average_pos = np.expand_dims(average_pos, 0)
                    batch.average_angle = np.expand_dims(average_angle, 0)

                    def blend(img, data, factor):
                        alpha = 0.5
                        data = cv2.warpAffine(data, self._image_t_output[:2, :],
                                              (image.shape[1], image.shape[0]),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        data = utils.red_green(data * factor)
                        return alpha * data + (1 - alpha) * img

                    for cat in range(self._category_count):
                        batch.output_on_image[bi][cat][0] = blend(image, batch.outputs[0, cat, :, :, 0], 1)
                        batch.output_on_image[bi][cat][1] = blend(image, batch.outputs[0, cat, :, :, 1], 1)

                    utils.save_batch_as_images(batch, self._diag_dir, prefix='{:02}'.format(bi), fmt='{0}{2}.png')

        if type(result) == list:
            objects = result[PredictionModelOutputs.OBJECTS]
        else:
            objects = result

        pos = np.hstack([objects[:, :2], np.ones((objects.shape[0], 1))])
        pos = np.dot(pos, self._image_t_output.T)[:, :2]

        predictions = []
        for i in range(len(objects)):
            predictions.append(Object(pos[i][0], pos[i][1], objects[i][2], objects[i][3], objects[i][4]))

        return predictions
