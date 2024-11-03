from ultralytics import YOLO

model = YOLO("../models/yolo/yolov11/yolo11x.pt")
model.export(format="onnx",imgsz=640)