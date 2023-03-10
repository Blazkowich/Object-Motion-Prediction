import cv2
import numpy as np
from Red_Point_detector import RedPointDetector


class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]], np.float32)
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y


dispW = 640
dispH = 480
flip = 2

cap = cv2.VideoCapture(0)

od = RedPointDetector()

kf = KalmanFilter()

outVid = cv2.VideoWriter('videos/RedPoint.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (dispW, dispH))
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret is False:
        break
    redpoint_bbox = od.detect(frame)
    x, y, x2, y2 = redpoint_bbox
    cx = int((x + x2) / 2)
    cy = int((y + y2) / 2)

    predicted = kf.predict(cx, cy)
    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
    cv2.circle(frame, (predicted[0], predicted[1]), 10, (255, 0, 0), 4)

    cv2.imshow("Frame", frame)
    outVid.write(frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
outVid.release()
cv2.destroyAllWindows()
