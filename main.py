from ultralytics import YOLO
import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

model = YOLO("yolo-Weights/yolov9e.pt") 

cap = cv.VideoCapture('barella.mov')

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        for i, xywh in enumerate(results[0].boxes.xywh):
            x, y, w, h = [int(xywh[j]) for j in range(4)]
            left = int(x-w/2)
            top = int(y-h/2)
            crop = results[0].orig_img[top:top+h, left:left+w]

            # prendo solo parte alta (per prendere il colore della maglietta)
            top_crop = crop[:int(crop.shape[0]/2), :]

            # modifico in array 2d diviso per colori di pixel
            flat_crop = top_crop.reshape(-1, 3)

            kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
            kmeans.fit(flat_crop)

            labels = kmeans.labels_

            # il colore della maglietta è quello della label meno presente
            tot = sum(labels)
            if tot > flat_crop.shape[0]/2:
                player_color = kmeans.cluster_centers_[0]
            else:
                player_color = kmeans.cluster_centers_[1]

            cv.rectangle(frame, (left, top), (left+w, top+h), player_color, 2)
            

        cv.imshow('bboxes', frame)


        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
    