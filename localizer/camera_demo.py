import cv2
import sys

def run(camera_id):
    camera = cv2.VideoCapture(camera_id)

    while True:
        ret, frame = camera.read()
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(camera_id)
