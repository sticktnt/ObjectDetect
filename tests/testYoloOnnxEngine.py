import cv2
from deploy.onnx.YoloOnnxEngine import YoloOnnxEngine



if __name__ == '__main__':
    test_yolo_onnx_model = YoloOnnxEngine("../models/yolo/yolov11/yolo11x.onnx")
    img = cv2.imread("../images/bus.jpg")

    # 测试letterbox_resize
    img = test_yolo_onnx_model.letterbox_resize(img)
    cv2.imwrite("../images/bus_letterbox.jpg", img)
