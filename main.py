import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolov5n.pt")
    model.train(epochs=10)
    model.val()

    url = 'http://10.0.0.34:4747/video'

    cap = cv2.VideoCapture(url)

    first = True

    while True:
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        results = model.predict(frame, verbose=False, show=True, conf=0.5)

        cv2.imshow('WOW', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()