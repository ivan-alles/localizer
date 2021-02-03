# Copyright 2018-2020 Ivan Alles. See also the LICENSE file.

import os
import shutil

import cv2
import numpy as np


def make_clean_directory(path):
    """
    Creates an empty directory.
    If it exists, delete its content.
    :param path: path to the directory.
    """
    need_create = True
    try:
        if os.path.isdir(path):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            need_create = False
        elif os.path.isfile(path):
            os.remove(path)
    except PermissionError:
        # This can be caused by Windows Explorer indexing images, so just ignore it.
        pass
    if need_create:
        os.makedirs(path)


def make_transform2(scale, angle=0, tx=0, ty=0):
    """
    Rotate, scale (about origin) and then translate 2d points.
    """
    ca = np.cos(angle) * scale
    sa = np.sin(angle) * scale
    t = np.array([ca, -sa, tx, sa, ca, ty, 0, 0, 1]).reshape(3, 3)
    return t


def save_tensor_as_images(tensor, dst_dir, prefix='', suffix='',
                          delete_old_files=False, mode=None,
                          file_name_format='{0}{1:0>3}{2}-{3:1d}.png'):
    """
    Save tensor as images.

    :param tensor: an array of shape (batch, c, h, w) or (c, h, w) (will be reshaped to (1, c, h, w)).
    :param prefix a prefix for image name. Can be a string or an array-like of strings for each batch element.
    :param suffix a suffix for image name. Can be a string or an array-like of strings for each batch element.
    :param mode: one of
        None: default
        'rgb': save as rgb if it has 3 channels.
        'rg': save negative values as red, positive as green.
    :param file_name_format string for the file name. Arguments are
        0: prefix
        1: index in the 0th axis of the tensor
        2: suffix.
        3: channel (index in the 1st axis of the tensor).
    """
    if tensor.ndim == 3:
        tensor.reshape((1,) + tensor.shape)
    elif tensor.ndim != 4:
        raise ValueError("Unsupported tensor shape")

    os.makedirs(dst_dir, exist_ok=True)
    if delete_old_files:
        for f in os.listdir(dst_dir):
            file_path = os.path.join(dst_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def save_image(i, image, channel):
        if image.ndim == 2 and mode == 'rg':
            image = red_green(image)

        file_prefix = prefix if type(prefix) == str else prefix[i]
        file_suffix = suffix if type(suffix) == str else suffix[i]

        file_name = file_name_format.format(file_prefix, i, file_suffix, channel)
        cv2.imwrite(os.path.join(dst_dir, file_name), image.astype(np.uint8))

    for i in range(0, tensor.shape[0]):
        if tensor[i].shape[2] == 3 and mode == 'rgb':
            save_image(i, tensor[i], 0)
        elif tensor[i].shape[2] == 1:
            save_image(i, tensor[i][:, :, 0], 0)
        else:
            for ci in range(tensor[i].shape[2]):
                save_image(i, tensor[i][:, :, ci], ci)


def red_green(image):
    """
    Convert a 1 channel image of positive and negative numbers into a color image.
    Red corresponds to negative numbers, green to positive ones.
    :param image: one channel image (H, W) or (H, W, 1).
    :return: a color image.
    """
    if np.ndim == 2:
        image = np.squeeze(image, axis=2)
    pos = (image >= 0) * image
    neg = (image <= 0) * image
    blue = np.zeros(image.shape, dtype=image.dtype)
    image = np.stack([blue, pos, -neg], axis=-1)
    return image


def save_batch_as_images(batch, dst_dir, fmt, prefix=''):
    make_clean_directory(dst_dir)

    maxcol = 230

    if hasattr(batch, 'input'):
        save_tensor_as_images(batch.input * maxcol, dst_dir, prefix=prefix, suffix='_i',
                              delete_old_files=False,
                              mode='rgb',
                              file_name_format=fmt)

    if hasattr(batch, 'output_on_image'):
        for cat in range(batch.output_on_image.shape[1]):
            save_tensor_as_images(batch.output_on_image[:, cat, 0] * maxcol, dst_dir,
                                  prefix=prefix, suffix=f'_{cat}_oyoi', mode='rgb', file_name_format=fmt)
            save_tensor_as_images(batch.output_on_image[:, cat, 1] * maxcol, dst_dir,
                                  prefix=prefix, suffix=f'_{cat}_oxoi', mode='rgb', file_name_format=fmt)

    def save_variable(batch, name, suffix, channel_names, factors, mode='rg'):
        if not hasattr(batch, name):
            return
        tensor = getattr(batch, name)
        if tensor is None:
            return
        tensor = np.array(tensor)
        for cat in range(tensor.shape[1]):
            for ch in range(tensor.shape[-1]):
                save_tensor_as_images(tensor[:, cat, :, :, ch: ch + 1] * factors[ch], dst_dir,
                                      prefix=prefix,
                                      suffix=f'_{cat:d}_{suffix}{channel_names[ch]}',
                                      mode=mode,
                                      file_name_format=fmt)

    pos_factor = maxcol * 0.05
    # Training only
    save_variable(batch, 'output_window', 'ow', ['y', 'x', 's', 'c'], [maxcol, maxcol, maxcol, maxcol])
    save_variable(batch, 'target_window', 'tw', ['y', 'x', 's', 'c'], [maxcol, maxcol, maxcol, maxcol])
    save_variable(batch, 'weight', 'w', [''], [maxcol])
    # Training & prediction
    save_variable(batch, 'output', 'o', ['y', 'x', 's', 'c'], [pos_factor, pos_factor, maxcol, maxcol])
    # Prediction only
    save_variable(batch, 'output_window_pos', 'ow', ['y', 'x'], [maxcol, maxcol])
    save_variable(batch, 'match_pos', 'm', [''], [maxcol])
    save_variable(batch, 'average_pos', 'a', ['y', 'x'], [maxcol, maxcol])
    save_variable(batch, 'average_angle', 'a', ['s', 'c'], [maxcol, maxcol])


def make_xy_tensor(shape):
    """
    Compute a tensor with homogeneous pixel coordinates.
    :param shape: tensor shape.
    :return: a tensor of shape: shape + (2,)
    """
    xy = np.ones((3,) + shape, dtype=np.int32)
    xy[0] = np.arange(0, shape[1])  # x
    xy[1] = np.arange(0, shape[0]).T.reshape(-1, 1)  # y
    return np.moveaxis(xy, 0, 2)


def draw_objects(image, objects, axis_length=10, thickness=1, scale=1, category_colors=None):
    """
    Draw poses on an image.
    :param image: the image.
    :param objects: ground truth or predicted objects.
    :param axis_length: a number or a tuple (lx, ly) of axis lengths.
    :param thickness: line thickness.
    :param scale: scale factor.
    :param category_colors: an array of colors for categories.
    :return:
    """
    maxcol = 1
    if image.dtype == np.uint8:
        maxcol = 255

    axis_length = np.broadcast_to(axis_length, (2,))

    if category_colors is None:
        category_colors = [[0, 0, 1],
                           [0, 1, 0],
                           [1, 0, 0],
                           [1, 0, 1],
                           [1, 1, 0],
                           [1, 0, 1],
                           [1, 1, 1]
                           ]

    category_colors = np.array(category_colors, dtype=np.float32) * maxcol

    for obj in objects:
        conf = obj.confidence if hasattr(obj, 'confidence') else 1
        conf = conf * 0.5 + 0.5
        color = category_colors[obj.category % len(category_colors)] * conf
        color = tuple([float(c) for c in color])
        t = make_transform2(1, obj.angle, obj.origin[0] * scale, obj.origin[1] * scale)
        points = np.array([
            [0, 0, 1],
            [axis_length[0], 0, 1],
            [0, 0, 1],
            [0, axis_length[1], 1]
        ], dtype=np.float32)

        points = np.dot(points, t.T)[:, :2].astype(int)

        cv2.line(image, tuple(points[0]), tuple(points[1]), color, thickness)
        cv2.line(image, tuple(points[2]), tuple(points[3]), color, thickness)
