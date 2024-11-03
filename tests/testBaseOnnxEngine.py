import numpy as np
from deploy.onnx.BaseOnnxEngine import BaseOnnxEngine

if __name__ == "__main__":
    # 加载模型
    model_path = "../models/yolo/yolov11/yolo11x.onnx"
    engine = BaseOnnxEngine(model_path)
    # 构造输入
    inputs = np.random.randn(1, 3, 640, 640).astype(np.float32)
    # 模型推理
    for i in range(10):
        outputs = engine.inference(inputs)
    # outputs = engine.inference(inputs)
    # print(outputs)