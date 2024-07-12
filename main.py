from ultralytics import YOLO
import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
import math
from functions import assign_team, top_view

model = YOLO("yolo-Weights/yolov9e.pt") 

cap = cv.VideoCapture('barella.mov')

team_colors = []
player_color = {}


while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        assign_team(frame, results, model, player_color, team_colors, cap)
        top_view()

        cv.imshow('bboxes', frame)


        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
    