from ultralytics import YOLO
import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
import math
from functions import assign_team, camera_movement, show_birdview, get_player_pos

model = YOLO("yolo-Weights/yolov9e.pt") 

cap = cv.VideoCapture('barella.mov')

team_colors = []
player_color = {}
old_frame = None
cam_xy = [0,0]


l = 105/2 - 16.5
h = 68/2

vertici = np.array([[649, 277], [1413,317], [376,494], [1398,317]]).astype(np.float32)

rettangolo = np.array([[0,0], [l,0], [0,h], [l,h]]).astype(np.float32)

transform = cv.getPerspectiveTransform(vertici, rettangolo)


while cap.isOpened():
    success, frame = cap.read()

    if success:
        old_frame_original = frame

        results = model.track(frame, persist=True)

        if old_frame is not None:
            cam_xy = camera_movement(old_frame, frame)
             
            cv.putText(frame, f'cam x: {cam_xy[0]}', (10,30), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            cv.putText(frame, f'cam y: {cam_xy[1]}', (10,50), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


        if results[0].boxes.id is not None:
            transformed_positions = get_player_pos(results, cam_xy, transform)
            

            if len(results[0].boxes.xywh) > 1:
                assign_team(frame, results, model, player_color, team_colors, cap)

        
        old_frame = old_frame_original
        cv.imshow('bboxes', frame)


        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
    