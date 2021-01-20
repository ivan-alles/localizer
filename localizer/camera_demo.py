import datetime
import enum
import json
import os
import shutil
import sys

import cv2
import numpy as np

from localizer import train
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
        self._dataset = []
        self._scale_factor = None
        self._mode = self.__class__.Mode.DETECT
        # self._localizer = predict.Localizer(r'models\.archive\tools_tl-20210118-145200/config.json')
        self._localizer = predict.Localizer(r"D:\ivan\projects\localizer-transfer-learning\.temp\demo_model\config.json")
        self._object_size = self._localizer.cfg['object_size']
        self._camera = cv2.VideoCapture(camera_id)
        self._input_image = None
        self._view_image = None
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

            self._input_image = cv2.resize(camera_image, (0, 0), fx=self._scale_factor, fy=self._scale_factor)
            self._view_image = np.copy(self._input_image)

            if self._mode == self.__class__.Mode.DETECT:
                self._detect()
            else:
                self._train()

            cv2.imshow('camera', self._view_image)

    def _detect(self):
        input = self._input_image.astype(np.float32) / 255
        predictions = self._localizer.predict(input)
        utils.draw_objects(self._view_image, predictions, axis_length=20, thickness=2)

    def _train(self):
        x = self._input_image.shape[1] / 2
        y = self._input_image.shape[0] / 2
        cv2.circle(self._view_image, (int(x), int(y)), self._object_size // 2, (0, 255, 0))
        if self._key == ord(' '):
            image_file = datetime.datetime.now().strftime('image-%Y%m%d-%H%M%S-%f.png')
            image_path = os.path.join(self._model_dir, image_file)
            cv2.imwrite(image_path, self._input_image)
            data_element = {
              "image": image_file,
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
            self._dataset.append(data_element)
            with open(self._dataset_path, 'w') as f:
                json.dump(self._dataset, f, indent=' ')

            cfg_path = os.path.join(self._model_dir, 'config.json')
            shutil.copyfile(self._template_cfg_path, cfg_path)

            train.configure_logging(cfg_path)
            trainer = train.Trainer(cfg_path)
            trainer.run()
            self._mode = self.__class__.Mode.DETECT


if __name__ == '__main__':
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    app = CameraDemo(camera_id)
    app.run()
