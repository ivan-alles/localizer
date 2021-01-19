import sys

import cv2
import numpy as np

from localizer import predict
from localizer import utils


def run(camera_id):
    scale_factor = None

    localizer = predict.Localizer(r'models\.archive\tools_tl-20210118-145200/config.json')
    object_size = localizer.cfg['object_size']
    camera = cv2.VideoCapture(camera_id)

    while True:
        if cv2.waitKey(1) == ord('q'):
            break
        ret, camera_image = camera.read()
        if camera_image is None:
            print('Cannot read image')
            continue

        if scale_factor is None:
            actual_length = np.max(camera_image.shape[:2])
            desired_length = object_size * 6
            scale_factor = desired_length / actual_length

        camera_image = cv2.resize(camera_image, (0, 0),
                                  fx=scale_factor,
                                  fy=scale_factor)
        image = camera_image.astype(np.float32) / 255
        predictions = localizer.predict(image)
        cv2.circle(camera_image,
                   (camera_image.shape[1] // 2, camera_image.shape[0] // 2),
                   object_size // 2, (0, 255, 0))
        utils.draw_objects(camera_image, predictions, axis_length=20, thickness=2)
        cv2.imshow('camera', camera_image)




if __name__ == '__main__':
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(camera_id)
