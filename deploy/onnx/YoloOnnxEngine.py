import cv2
from .BaseOnnxEngine import BaseOnnxEngine


class YoloOnnxEngine(BaseOnnxEngine):
    """
    YOLOv11 onnx 模型推理
    """

    def __init__(self, model_path, confidence_threshold=0.25, iou_threshold=0.5):
        super(YoloOnnxEngine, self).__init__(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def letterbox_resize(self, img, new_shape=(640, 640)):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad)
        top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
        left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border
        return img
