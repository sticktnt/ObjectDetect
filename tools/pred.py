from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("../models/yolo/yolov11/yolo11x.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("../images/bus.jpg", save=True, imgsz=640, conf=0.5,device="cuda:0")

# -------------------------------------------
# image 1/1 D:\work\python\ObiectDetect\tools\..\images\bus.jpg: 640x480 4 persons, 1 bus, 92.9ms
# Speed: 2.0ms preprocess, 92.9ms inference, 65.7ms postprocess per image at shape (1, 3, 640, 480)
# -------------------------------------------