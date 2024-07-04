from ultralytics import YOLO


model = YOLO("yolo-Weights/yolov9e.pt") 

results = model.predict("calcio.mp4", save = True)


