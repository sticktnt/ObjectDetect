import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

from tools.cost_time import cost_time


class BaseTrtEngine:
    """
    Base class for TensorRT engine
    """
    def __init__(self, engine_path):
        """
        初始化TensorRT推理引擎
        :param engine_path: TensorRT引擎文件路径(.engine)
        """
        # 创建logger和runtime
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        
        # 加载引擎文件
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 分配GPU内存
        self.stream = cuda.Stream()
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.bindings = []
        
        # 为每个输入输出分配内存
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配CPU和GPU内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

    @cost_time
    def inference(self, inputs):
        """
        执行TensorRT模型推理
        :param inputs: 输入数据，numpy数组
        :return: 推理结果，numpy数组
        """
        # 将输入数据复制到GPU
        for host_input, device_input, input_data in zip(self.host_inputs, self.device_inputs, inputs):
            np.copyto(host_input, input_data.ravel())
            cuda.memcpy_htod_async(device_input, host_input, self.stream)
            
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 将结果从GPU复制回CPU
        outputs = []
        for host_output, device_output in zip(self.host_outputs, self.device_outputs):
            cuda.memcpy_dtoh_async(host_output, device_output, self.stream)
            outputs.append(host_output)
            
        # 同步流
        self.stream.synchronize()
        
        return outputs

