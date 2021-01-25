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
        DETECT = 0  # Detect using current model
        NEW_MODEL = 1  # Create a new dataset and train a new model

    def __init__(self, camera_id):
        self._template_cfg_path = os.path.join(os.path.dirname(__file__), 'camera_demo.json')
        self._model_dir = os.path.join('.temp', 'demo_model')
        self._dataset_path = os.path.join(self._model_dir, 'dataset.json')
        self._cfg_path = os.path.join(self._model_dir, 'config.json')

        self._dataset = None
        self._scale_factor = None
        self._mode = self.__class__.Mode.DETECT
        self._localizer = None
        self._object_size = None
        self._camera_image = None
        self._view_image = None
        self._key = -1
        os.makedirs(self._model_dir, exist_ok=True)

        self._camera = cv2.VideoCapture(camera_id)
        self._load_model()


    def run(self):
        while True:
            self._key = cv2.waitKey(1)
            if self._key == ord('q'):
                break
            elif self._key == ord('n'):
                self._mode = self.__class__.Mode.NEW_MODEL
            elif self._key == ord('d'):
                self._mode = self.__class__.Mode.DETECT
            elif self._key == ord('r'):
                self._train()  # Retrain model on existing data, useful for tests.

            ret, camera_frame = self._camera.read()
            if camera_frame is None:
                print('Cannot read camera frame')
                continue

            camera_frame = np.fliplr(camera_frame)

            if self._scale_factor is None:
                actual_length = np.max(camera_frame.shape[:2])
                desired_length = self._object_size * 6
                self._scale_factor = desired_length / actual_length

            self._camera_image = cv2.resize(camera_frame, (0, 0), fx=self._scale_factor, fy=self._scale_factor)
            self._view_image = np.copy(self._camera_image)

            if self._mode == self.__class__.Mode.DETECT:
                self._detect()
            else:
                self._new_model()

            cv2.imshow('camera', self._view_image)

    def _draw_pose(self, image, x, y, angle, color=(0, 255, 0), draw_object_size=True):
        arrow = np.array([0, 0, 0, -1, 0, -1, .1, -.8, 0, -1, -.1, -0.8]).reshape(-1, 2)
        t = utils.make_transform2(self._object_size / 2, angle, x, y)
        arrow = np.dot(np.append(arrow, np.ones((arrow.shape[0], 1)), axis=1), t.T)[:, :2].astype(int)
        for i in range(0, len(arrow), 2):
            cv2.line(image, tuple(arrow[i]), tuple(arrow[i + 1]), color)
        if draw_object_size:
            cv2.circle(self._view_image, (int(x), int(y)), self._object_size // 2, color)

    def _detect(self):
        input = self._make_input(self._camera_image)
        predictions = self._localizer.predict(input)
        utils.draw_objects(self._view_image, predictions, axis_length=20, thickness=2)
        for object in predictions:
            self._draw_pose(self._view_image, object.origin[0], object.origin[1], object.angle)

    def _make_input(self, image):
        image = image.astype(np.float32) / 255
        image = np.power(image, 0.5)  # A slight gamma-correction
        return image

    def _load_model(self):
        self._localizer = predict.Localizer('.temp/demo_model/config.json')
        self._object_size = self._localizer.cfg['object_size']
        self._mode = self.__class__.Mode.DETECT

    def _new_model(self):
        self._dataset = []
        x = self._camera_image.shape[1] / 2
        y = self._camera_image.shape[0] / 2
        cv2.circle(self._view_image, (int(x), int(y)), self._object_size // 2, (0, 255, 0))
        if self._key == ord(' '):
            image_file = datetime.datetime.now().strftime('image-1.png')
            image_path = os.path.join(self._model_dir, image_file)
            input_image = self._make_input(self._camera_image) * 255
            cv2.imwrite(image_path, input_image)
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

            shutil.copyfile(self._template_cfg_path, self._cfg_path)

            self._train()

    def _train(self):
        train.configure_logging(self._cfg_path)
        trainer = train.Trainer(self._cfg_path)
        trainer.run()

        self._load_model()


if __name__ == '__main__':
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    app = CameraDemo(camera_id)
    app.run()
