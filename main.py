from ultralytics import YOLO


model = YOLO("yolo-Weights/yolov9t.pt") 

results = model.track("barella.mov", save=True, stream=True)

for result in results:
    im = result.orig_img
    for xywh in result.boxes.xywh:
        x, y, w, h = [int(xywh[i]) for i in range(4)]
        left = int(x-w/2)
        top = int(y-h/2)
        crop = result.orig_img[top:top+h, left:left+w]