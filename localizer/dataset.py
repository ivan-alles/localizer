# Copyright 2018-2020 Ivan Alles. See also the LICENSE file.

import json
import os

import cv2
import numpy as np

from robogym import geometry

from localizer import predict
from localizer import utils


class Dataset:
    """
    A dataset consisting of labeled images.
    """
    def __init__(self, file_name, cfg):
        self._file_name = file_name
        with open(file_name) as f:
            data = json.load(f)

        self.image_mean = None
        self.data_elements = []
        root_dir = os.path.dirname(self._file_name)
        for i, data_element_data in enumerate(data):
            self.data_elements.append(DataElement(i, cfg, root_dir, data_element_data))

    def save(self, file_name):
        data_elements = []

        for data_element in self.data_elements:
            objects = []
            for obj in data_element.objects:
                image_t_bounding_box = np.linalg.inv(obj.bounding_box_t_image)
                objects.append({
                    'category': obj.category,
                    'origin': {
                        'x': obj.origin[0],
                        'y': obj.origin[1],
                        'angle': obj.angle
                    },
                    'bounding_box': {
                        'x': image_t_bounding_box[0, 2],
                        'y': image_t_bounding_box[1, 2],
                        'size_x': obj.bounding_box_sx,
                        'size_y': obj.bounding_box_sy,
                        'angle': np.arctan2(image_t_bounding_box[1, 0], image_t_bounding_box[0, 0])
                    },
                })
            data_elements.append(
                {
                    'image': os.path.relpath(data_element.path, os.path.dirname(file_name)),
                    'objects': objects
                }
            )

        with open(file_name, 'w') as f:
            json.dump(data_elements, f, indent=1)

    def precompute_training_data(self, data_element_indices):
        mean_sum = np.zeros(3)
        mean_count = np.zeros(3)
        for i in data_element_indices:
            self.data_elements[i].precompute_training_data(mean_sum, mean_count)

        self.image_mean = (mean_sum / mean_count).astype(np.float32)
        print(f'Image mean {self.image_mean}')


class DataElement:
    """
    An element of a dataset, an image with labeled objects.
    """
    def __init__(self, id, cfg, root_dir, data):
        self.id = id
        self._cfg = cfg
        self.rel_path = data['image']
        self.full_path = os.path.join(root_dir, self.rel_path)
        self.objects = []

        for i, obj_label in enumerate(data['objects']):
            self.objects.append(Object(i, obj_label))

    def read_image(self):
        image = cv2.imread(self.full_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read image {self.full_path}')
        dtype = image.dtype
        image = image.astype(np.float32)
        if dtype == np.uint8:
            image *= 1 / 255
        elif dtype == np.uint16:
            image *= 1 / 65535
        else:
            raise ValueError(f'Unsupported image type {dtype} for {self.full_path}')
        return image

    def precompute_training_data(self, mean_sum, mean_count):
        output_dir = self._cfg['runtime']['output_dir']

        image = self.read_image()

        mean_sum += image.sum(axis=(0, 1))
        mean_count += image.shape[0] * image.shape[1]

        object_thumbnail_dir = os.path.abspath(os.path.join(output_dir, 'objects'))
        os.makedirs(object_thumbnail_dir, exist_ok=True)

        for obj in self.objects:
            directory = os.path.join(object_thumbnail_dir, f"{obj.category}")
            os.makedirs(directory, exist_ok=True)
            fn = os.path.splitext(os.path.basename(self.full_path))[0]
            file_name = f'{fn}-{obj.id:03d}.png'
            data_element_image = obj.get_bounding_box_patch(image) * 255
            cv2.imwrite(os.path.join(directory, file_name), data_element_image.astype(np.uint8))

        image_shape = image.shape[:2]
        target_scale = self._cfg['input_scale'] * self._cfg['model_scale']
        target_shape = tuple(np.ceil(np.array(image_shape, dtype=np.float32) * target_scale).astype(int))

        output_dir = os.path.dirname(self.data_file)
        os.makedirs(output_dir, exist_ok=True)

        # A tensor with homogeneous target pixel coordinates
        target_xy = utils.make_xy_tensor(target_shape)

        image_t_target = utils.make_transform2(1 / target_scale)
        image_xy = np.dot(target_xy, image_t_target.T)

        category_count = self._cfg['runtime']['category_count']

        object_map = np.full((category_count,) + target_shape, -1, dtype=int)
        dist_to_obj = []

        for cat in range(category_count):
            for i, obj in enumerate(self.objects):
                if obj.category != cat:
                    continue
                obj_t_image = obj.bounding_box_t_image
                scaled_obj_t_image = np.dot(
                    np.diag([2 / obj.bounding_box_sx, 2 / obj.bounding_box_sy, 1]),
                    obj_t_image)

                dist_to_obj.append(np.linalg.norm(image_xy[:, :, :2] - obj.origin, axis=2))

                obj_xy = np.abs(np.dot(image_xy, scaled_obj_t_image.T))
                obj_map_one = np.logical_and(obj_xy[:, :, 0] < 1, obj_xy[:, :, 1] < 1)

                object_map[cat] = object_map[cat] * (1 - obj_map_one) + obj_map_one * i

        nearest_object = np.argmin(np.stack(dist_to_obj), axis=0)

        # Save tensors
        np.savez(self.data_file,
                 image=image,
                 object_map=object_map,
                 nearest_object=nearest_object)

        # Save diagnostic images
        cv2.imwrite(os.path.join(output_dir, 'image.png'), image * 255)
        cv2.imwrite(os.path.join(output_dir, 'nearest_object.png'),
                    ((nearest_object + 1) * 255 / (len(self.objects) + 1.001)).astype(np.uint8))
        for cat in range(category_count):
            cv2.imwrite(os.path.join(output_dir, f'{cat:03d}-object_map.png'),
                        ((object_map[cat] + 1) * 255 / (len(self.objects) + 1.001)).astype(np.uint8))

    def make_training_data(self, batch, batch_index, rng):
        show_diag_images = False  # Set to true to see diag images.
        data_element_tensors = np.load(self.data_file)

        image = data_element_tensors['image']

        category_count = batch.weight.shape[1]
        input_shape = batch.input.shape[1:]
        output_shape = batch.target_window.shape[1:]

        random_angle, random_center = self._select_random_pose(image, rng)

        # Test code
        # random_center = np.array([319.5 + 0, 239.5 - 0])
        # random_angle = 0

        daug_t_pose = self._make_data_augmentation_transform(rng)

        # Transform to the selected pose
        pose_t_image = np.dot(
            utils.make_transform2(1, -random_angle),
            utils.make_transform2(1, 0, -random_center[0], -random_center[1]))

        input_center = np.array(input_shape[1::-1]) * .5 - .5
        # Transform to the center of the input.
        input_t_daug = utils.make_transform2(self._cfg['input_scale'], 0, input_center[0], input_center[1])

        input_t_image = np.linalg.multi_dot((input_t_daug, daug_t_pose, pose_t_image))

        assert np.allclose(np.dot(input_t_image, np.append(random_center, 1))[:2], input_center), \
            'Training example center must go to the input center'

        input = cv2.warpAffine(image, input_t_image[:2, :3],
                               (input_shape[1], input_shape[0]),
                               flags=cv2.INTER_LINEAR)
        input *= rng.uniform(*self._cfg['data_augmentation_color'])

        batch.input[batch_index] = input

        if show_diag_images:
            cv2.imshow('input', input)

        target_t_input = utils.make_transform2(self._cfg['model_scale'])
        target_t_image = np.dot(target_t_input, input_t_image)

        # We store the target tensors in target resolution, but input_t_image is in image resolution,
        # so scale back to image first, than do the same transform as for the input,
        # and finally scale to target.
        target_t_precomputed = np.dot(
            target_t_image,
            utils.make_transform2(1 / self._cfg['model_scale'] / self._cfg['input_scale']))

        # A tensor with homogeneous target pixel coordinates
        target_xy = utils.make_xy_tensor(output_shape[1:3])

        batch.weight[batch_index] = self._cfg['background_weight']

        for cat in range(category_count):
            for i, obj in enumerate(self.objects):
                if obj.category != cat:
                    continue
                pos = np.dot(
                    target_t_image,
                    np.append(obj.origin, 1))

                t = pos.reshape(1, 1, -1) - target_xy
                a = np.full(output_shape[1:3], geometry.normalize_angle(obj.angle - random_angle))
                wx, wy, wsa, wca = make_window(t[:, :, 0], t[:, :, 1], a, self._cfg['sigma'])
                batch.target_window[batch_index, cat, :, :, predict.TrainingModelChannels.X] += wx
                batch.target_window[batch_index, cat, :, :, predict.TrainingModelChannels.Y] += wy
                batch.target_window[batch_index, cat, :, :, predict.TrainingModelChannels.SA] += wsa
                batch.target_window[batch_index, cat, :, :, predict.TrainingModelChannels.CA] += wca

                obj_weight = np.zeros(output_shape[1:3])
                c = tuple(int(x) for x in np.round(pos[:2]))
                r = int(np.round(self._cfg['object_weight_sigma_factor'] * self._cfg['sigma'] + 0.5))
                cv2.circle(obj_weight, c, r, 1, cv2.FILLED)
                batch.weight[batch_index] = np.maximum(batch.weight[batch_index], np.expand_dims(obj_weight, 2))

                if show_diag_images:
                    cv2.imshow(f'{i}-wsa', utils.red_green(wsa))
                    cv2.imshow(f'{i}-wca', utils.red_green(wca))
                    cv2.imshow(f'{i}-tx', utils.red_green(wx) * 0.5)
                    cv2.imshow(f'{i}-ty', utils.red_green(wy) * 0.5)

        # Zero out weight in border areas, where the model kernel does not fit.
        # The border shall be a half of the model kernel size plus optionally some pixels for
        # convnet padding.
        border = self._cfg['weight_border']
        batch.weight[batch_index, :, :border, :, :] = 0
        batch.weight[batch_index, :, -border:, :, :] = 0
        batch.weight[batch_index, :, :, :border, :] = 0
        batch.weight[batch_index, :, :, -border:, :] = 0

        if show_diag_images:
            cv2.waitKey(0)

    def _select_random_pose(self, image, rng):
        obj = self.objects[rng.randint(len(self.objects))]
        if rng.uniform(0, 1) <= self._cfg['random_background_probability']:
            random_center = np.array(
                [
                    rng.uniform(0, image.shape[1]),
                    rng.uniform(0, image.shape[0])
                ])
            random_angle = rng.uniform(-np.pi, np.pi)
        else:
            # It is enough to vary the position within model scale radius to cover all possible
            # object placements for the convnet.
            offset = rng.uniform(-.5, .5, size=(2,)) / self._cfg['model_scale']
            random_center = obj.origin + offset
            random_angle = obj.angle + rng.uniform(-np.pi, np.pi)
        return random_angle, random_center

    def _make_data_augmentation_transform(self, rng):
        # Compute data augmentation transformation
        #
        shear_scale = rng.uniform(*self._cfg['data_augmentation_shear_scale'])
        shear_rot = rng.uniform(*self._cfg['data_augmentation_shear_rotation'])
        scale = rng.uniform(*self._cfg['data_augmentation_scale'])
        # Test code
        # scale = 1
        # shear_scale_x = 2
        # shear_scale_y = 1
        # shear_rot = 0.5
        t_shear_rot = utils.make_transform2(1, shear_rot)  # Rotation to make shear scaling
        t_shear_scale = utils.make_transform2(shear_scale)  # Shear scaling
        t_scale = utils.make_transform2(scale)  # Uniform scaling
        daug_t_pose = np.linalg.multi_dot([t_scale, t_shear_rot.T, t_shear_scale, t_shear_rot])
        # Test code
        # daug_t_pose = np.eye(3)
        return daug_t_pose

    @property
    def data_file(self):
        return os.path.abspath(os.path.join(self._cfg['runtime']['output_dir'], 'data_elements', f'{self.id:05d}',
                                            'tensors.npz'))


class Object:
    """
    A ground-truth object with labeled coordinate system (origin, angle) and an oriented bounding box.

    Angle values are in radians.
    """
    def __init__(self, id, data):
        self.id = id
        self.category = data['category']

        self.origin = np.array([data['origin']['x'], data['origin']['y']])
        self.angle = data['origin']['angle']

        image_t_bb = utils.make_transform2(
            1,
            data['bounding_box']['angle'],
            data['bounding_box']['x'],
            data['bounding_box']['y'])
        self.bounding_box_t_image = np.linalg.inv(image_t_bb)
        self.bounding_box_sx = data['bounding_box']['size_x']
        self.bounding_box_sy = data['bounding_box']['size_y']

    def __str__(self):
        return f'{self.origin[0]:.2f} {self.origin[1]:.2f} {np.rad2deg(self.angle):.2f} {self.category}'

    def get_bounding_box_patch(self, image, warp_affine_kwargs={'flags': cv2.INTER_LINEAR}):
        """
        Gets an image patch corresponding to the bounding box.
        :param image: full image.
        :return:
        """
        size = (int(self.bounding_box_sx), int(self.bounding_box_sy))
        t = np.dot(np.array([[1.0, 0, size[0] / 2], [0, 1, size[1] / 2], [0, 0, 1]]),
                   self.bounding_box_t_image)
        patch = cv2.warpAffine(image, t[:2, :3], size, **warp_affine_kwargs)

        return patch


def make_window(x, y, angle, sigma):
    g = np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(sigma))
    wx = x / sigma * g
    wy = y / sigma * g
    wsa = np.sin(angle) * g
    wca = np.cos(angle) * g
    return wx, wy, wsa, wca
