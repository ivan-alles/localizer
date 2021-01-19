import enum
import json
import os
import sys

import cv2
import numpy as np

from localizer import dataset
from localizer import predict
from localizer import utils

class CameraDemo:

    class Mode(enum.Enum):
        DETECT = 0
        TRAIN = 1

    def __init__(self, camera_id):
        self._template_cfg_path = os.path.join(os.path.dirname(__file__), 'camera_demo.json')
        self._model_dir = os.path.join('.temp', 'demo_model')
        self._dataset_path = os.path.join(self._model_dir, 'dataset.json')
        with open(self._dataset_path, 'w') as f:
            json.dump([], f)  # Create an empty dataset
        self._scale_factor = None
        self._mode = self.__class__.Mode.DETECT
        self._localizer = predict.Localizer(r'models\.archive\tools_tl-20210118-145200/config.json')
        self._dataset = dataset.Dataset(self._dataset_path, self._template_cfg_path)
        self._object_size = self._localizer.cfg['object_size']
        self._camera = cv2.VideoCapture(camera_id)
        self._key = -1
        os.makedirs(self._model_dir, exist_ok=True)

    def run(self):
        while True:
            self._key = cv2.waitKey(1)
            if self._key == ord('q'):
                break
            elif self._key == ord('t'):
                self._mode = self.__class__.Mode.TRAIN
            elif self._key == ord('d'):
                self._mode = self.__class__.Mode.DETECT

            ret, camera_image = self._camera.read()
            if camera_image is None:
                print('Cannot read image')
                continue

            np.fliplr(camera_image)

            if self._scale_factor is None:
                actual_length = np.max(camera_image.shape[:2])
                desired_length = self._object_size * 6
                self._scale_factor = desired_length / actual_length

            camera_image = cv2.resize(camera_image, (0, 0), fx=self._scale_factor, fy=self._scale_factor)

            if self._mode == self.__class__.Mode.DETECT:
                self._detect(camera_image)
            else:
                self._train(camera_image)

            cv2.imshow('camera', camera_image)

    def _detect(self, camera_image):
        image = camera_image.astype(np.float32) / 255
        predictions = self._localizer.predict(image)
        utils.draw_objects(camera_image, predictions, axis_length=20, thickness=2)

    def _train(self, camera_image):
        x = camera_image.shape[1] / 2
        y = camera_image.shape[0] / 2
        cv2.circle(camera_image, (int(x), int(y)), self._object_size // 2, (0, 255, 0))
        if self._key == ord(' '):
            image_path = os.path.join(self._model_dir, 'image.png')
            cv2.imwrite(image_path, camera_image)
            data_element = {
              "image": image_path,
              "objects": [
                   {
                    "category": 0,
                    "origin": {
                     "x": x,
                     "y": y,
                     "angle": 0
                    }
                   }
                ]
            }
            self._dataset.add(data_element)
            self._dataset.save(self._dataset_path)


if __name__ == '__main__':
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    app = CameraDemo(camera_id)
    app.run()
