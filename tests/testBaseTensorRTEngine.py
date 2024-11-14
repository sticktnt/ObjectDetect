import numpy as np
import cv2

from deploy.tensorrt.BaseTensorRTEngine import BaseTrtEngine

if __name__ == '__main__':
    model_path = "../models/yolo/yolov11/yolo11x.engine"
    image_path = "../images/bus_letterbox.jpg"

    engine = BaseTrtEngine(model_path)
    # 构造输入
    inputs = cv2.imread(image_path).astype(np.float32)
    inputs = np.transpose(inputs, (2, 0, 1))
    inputs = np.expand_dims(inputs, axis=0)
    for i in range(10):
        outputs = engine.inference(inputs)
