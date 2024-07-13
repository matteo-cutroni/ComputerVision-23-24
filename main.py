from ultralytics import YOLO
import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
import math
from functions import assign_team, camera_movement

model = YOLO("yolo-Weights/yolov9e.pt") 

cap = cv.VideoCapture('barella.mov')

team_colors = []
player_color = {}
old_frame = None

while cap.isOpened():
    success, frame = cap.read()

    if success:
        old_frame_original = frame

        results = model.track(frame, persist=True)

        assign_team(frame, results, model, player_color, team_colors, cap)

        if old_frame is not None:
            cam_xy = camera_movement(old_frame, frame)
             
            cv.putText(frame, f'cam x: {cam_xy[0]}', (10,30), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            cv.putText(frame, f'cam y: {cam_xy[1]}', (10,50), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        old_frame = old_frame_original
        cv.imshow('bboxes', frame)


        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
    