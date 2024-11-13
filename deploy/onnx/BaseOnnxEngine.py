import onnxruntime as ort
import numpy as np

from tools.cost_time import cost_time


class BaseOnnxEngine:
    """
    Base class for ONNX engine
    """
    @cost_time
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'])
        # 默认YOLO模型输入输出均为一个节点
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @cost_time
    def inference(self, inputs):
        """
        onnx 模型推理
        :param inputs: 模型输入，输入形状要和模型输入节点形状一致
        :return: 模型推理的结果
        """
        outputs = self.session.run([self.output_name], {self.input_name: inputs})
        return outputs