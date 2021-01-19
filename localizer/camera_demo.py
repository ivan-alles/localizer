import enum
import sys

import cv2
import numpy as np

from localizer import predict
from localizer import utils


class CameraDemo:

    class Mode(enum.Enum):
        DETECT = 0
        TRAIN = 1

    def __init__(self, camera_id):
        self._scale_factor = None
        self._mode = self.__class__.Mode.DETECT
        self._localizer = predict.Localizer(r'models\.archive\tools_tl-20210118-145200/config.json')
        self._object_size = self._localizer.cfg['object_size']
        self._camera = cv2.VideoCapture(camera_id)

    def run(self):
        while True:
            if cv2.waitKey(1) == ord('q'):
                break
            ret, camera_image = self._camera.read()
            if camera_image is None:
                print('Cannot read image')
                continue

            if self._scale_factor is None:
                actual_length = np.max(camera_image.shape[:2])
                desired_length = self._object_size * 6
                scale_factor = desired_length / actual_length

            camera_image = cv2.resize(camera_image, (0, 0), fx=scale_factor, fy=scale_factor)

            if self._mode == self.__class__.Mode.DETECT:
                self._detect(camera_image)
            else:
                self._train(camera_image)

            cv2.imshow('camera', camera_image)


    def _detect(self, camera_image):
        image = camera_image.astype(np.float32) / 255
        predictions = self._localizer.predict(image)
        cv2.circle(camera_image,
                   (camera_image.shape[1] // 2, camera_image.shape[0] // 2),
                   self._object_size // 2, (0, 255, 0))
        utils.draw_objects(camera_image, predictions, axis_length=20, thickness=2)

    def _train(self, camera_image):
        pass



if __name__ == '__main__':
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    app = CameraDemo(camera_id)
    app.run()
