from ultralytics import YOLO
import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np

model = YOLO("yolo-Weights/yolov9e.pt") 

results = model.track("barella.mov", save=True, stream=True)

for frame, result in enumerate(results):
    histograms = []
    players_hist_dict = {}
    im = result.orig_img
    for i, xywh in enumerate(result.boxes.xywh):
        x, y, w, h = [int(xywh[j]) for j in range(4)]
        left = int(x-w/2)
        top = int(y-h/2)
        crop = result.orig_img[top:top+h, left:left+w]

        hist = cv.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        players_hist_dict[result.boxes.id[i]] = hist.flatten()
        histograms.append(hist.flatten())

    histograms = np.array(histograms)
    print(histograms.shape)
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=1)
    kmeans.fit(histograms)

    
    for key, value in players_hist_dict.items():
        value = np.array(value.reshape(1, -1))
        player_team = kmeans.predict(value)
        team_hist = kmeans.cluster_centers_[player_team]






    