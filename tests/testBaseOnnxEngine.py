import cv2
import numpy as np
from deploy.onnx.BaseOnnxEngine import BaseOnnxEngine

if __name__ == "__main__":
    # 加载模型
    model_path = "../models/yolo/yolov11/yolo11x.onnx"
    image_path = "../images/bus_letterbox.jpg"
    engine = BaseOnnxEngine(model_path)
    # 构造输入
    inputs = cv2.imread(image_path).astype(np.float32)
    inputs = np.transpose(inputs, (2, 0, 1))
    inputs = np.expand_dims(inputs, axis=0)
    # 模型推理
    for i in range(10):
        outputs = engine.inference(inputs)
    # outputs = engine.inference(inputs)
    # print(outputs)