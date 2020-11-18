# Copyright 2018-2020 Ivan Alles. See also the LICENSE file.

import os
import cv2

import numpy as np

from localizer import Localizer
import utils

MODEL_DIR = r'data\models\tools'
IMAGE_DIR = r'data\datasets\tools'


localizer = Localizer(MODEL_DIR)

for file in os.listdir(IMAGE_DIR):
    if os.path.splitext(file)[1].lower() not in ['.png', '.jpg', '.jpeg']:
        continue
    image = cv2.imread(os.path.join(IMAGE_DIR, file))
    image = image.astype(np.float32) / 255

    localizer.diag = False  # Set to True to see diagnostic images
    localizer.diag_dir = os.path.join(MODEL_DIR, '.temp', 'localizer_diag', file)

    predictions = localizer.predict(image)

    result_image = np.copy(image)
    utils.draw_objects(result_image, predictions, axis_length=20, thickness=2)
    result_image = (np.clip(result_image * 255, 0, 255)).astype(np.uint8)

    cv2.imshow('result', result_image)
    cv2.waitKey(500)

