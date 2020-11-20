# Copyright 2018-2020 Ivan Alles. See also the LICENSE file.

import os
import shutil

import cv2
import numpy as np


def make_clean_directory(path):
    """
    Creates an empty directory.

    If it exists, delete its content.
    If the directory is opened in Windows Explorer, may throw PermissionError,
    although the directory is usually cleaned. The caller may catch this exception to avoid program termination.
    :param path: path to the directory.
    """
    need_create = True
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


# See https://gist.github.com/seberg/3866040
def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):  # noqa: C901
    """
    Create a view of `array` which for every point gives the n-dimensional neighbourhood of size window.

    New dimensions are added at the end of
    `array` or after the corresponding original dimension.

    Parameters
    ----------
    array : array_like
        Array to which the rolling window is applied.
    window : int or tuple
        Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a
        to ignore a dimension in the window.
    asteps : tuple
        Aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows. (Equivalent to slicing result)
    wsteps : int or tuple (same size as window)
        steps for the added window dimensions. These can be 0 to repeat values
        along the axis.
    axes: int or tuple
        If given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2.
    toend : bool
        If False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array.

    Returns
    -------
    A view on `array` which is smaller to fit the windows and has windows added
    dimensions (0s not counting), ie. every point of `array` is an array of size
    window.

    Examples
    --------
    >>> a = np.arange(9).reshape(3,3)
    >>> rolling_window(a, (2,2))
    array([[[[0, 1],
             [3, 4]],
            [[1, 2],
             [4, 5]]],
           [[[3, 4],
             [6, 7]],
            [[4, 5],
             [7, 8]]]])

    Or to create non-overlapping windows, but only along the first dimension:
    >>> rolling_window(a, (2,0), asteps=(2,1))
    array([[[0, 3],
            [1, 4],
            [2, 5]]])

    Note that the 0 is discared, so that the output dimension is 3:
    >>> rolling_window(a, (2,0), asteps=(2,1)).shape
    (1, 3, 2)

    This is useful for example to calculate the maximum in all (overlapping)
    2x2 submatrixes:
    >>> rolling_window(a, (2,2)).max((2,3))
    array([[4, 5],
           [7, 8]])

    Or delay embedding (3D embedding with delay 2):
    >>> x = np.arange(10)
    >>> rolling_window(x, 3, wsteps=2)
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6],
           [3, 5, 7],
           [4, 6, 8],
           [5, 7, 9]])
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)  # maybe crude to cast to int...

    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps

        if np.any(asteps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1  # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape  # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps

    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _

        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)


def find_local_max(x, window_shape, threshold):
    if not all([s % 2 == 1 for s in window_shape]):
        raise ValueError("all window_shape dimensions must be odd")

    result = x.copy()
    pad_size = tuple(s // 2 for s in window_shape)

    pad_axes_count = len(window_shape)
    nonpad_axes_count = result.ndim - pad_axes_count
    pad_width_arg = tuple((0, 0) for _ in range(nonpad_axes_count))
    pad_width_arg += tuple((pad_size[i:i + 1] * 2) for i in range(pad_axes_count))

    padded = np.pad(result, pad_width_arg, 'constant')

    windows = rolling_window(padded, window_shape, axes=range(nonpad_axes_count, x.ndim))

    last_axes = tuple(range(windows.ndim - len(window_shape), windows.ndim))
    max_pooling = np.apply_over_axes(np.max, windows, axes=last_axes)
    max_pooling = max_pooling.squeeze(axis=last_axes)

    result = result - max_pooling
    result = result == 0

    result[x < threshold] = 0

    return result


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
        conf_a = 0.5
        conf = obj.confidence if hasattr(obj, 'confidence') else 1
        conf = conf * conf_a + (1 - conf_a)
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
