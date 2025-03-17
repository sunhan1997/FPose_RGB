import numpy as np
import cv2
from utils_function import load_image
import os
import platform
from loguru import logger
import time

if platform.system() != "Darwin":
    import warnings
    import pycuda.driver as cuda
    import tensorrt as trt

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    TRT_LOGGER = trt.Logger()
    TRT_LOGGER.min_severity = trt.Logger.Severity.ERROR
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

class HostDeviceMem:
    """
    Host 和 Device 内存包装类
    """
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class Dpt:
    """Depth Anything v2 推理类"""
    def __init__(self, engine_path) -> None:
        # 设定输入图像的尺寸
        self.reshape_size = [518, 518]
        # TensorRT 引擎文件路径
        self.model_path = engine_path
        # 初始化 TensorRT 模型
        self.__trt_init__(
            self.model_path,
            dynamic_shape=False,
            batch_size=1,
        )

    def __trt_init__(self, trt_file=None, dynamic_shape=False, gpu_idx=0, batch_size=1):
        """
        初始化 TensorRT。
        :param trt_file:    TensorRT 文件路径。
        :param dynamic_shape: 是否使用动态形状。
        :param gpu_idx: GPU 索引。
        :param batch_size: 批处理大小。
        """
        cuda.init()
        self._batch_size = batch_size
        self._device_ctx = cuda.Device(gpu_idx).make_context()
        self._engine = self._load_engine(trt_file)
        self._context = self._engine.create_execution_context()
        if not dynamic_shape:
            (
                self._input,
                self._output,
                self._bindings,
                self._stream,
            ) = self._allocate_buffers(self._context)

        logger.info("Dpt model <loaded>...")

    def _load_engine(self, trt_file):
        """
        加载 TensorRT 引擎。
        :param trt_file:    TensorRT 文件路径。
        :return: ICudaEngine
        """
        with open(trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def _allocate_buffers(self, context):
        """
        为数据分配设备内存空间。
        :param context:
        :return:
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self._engine:
            size = (
                trt.volume(self._engine.get_binding_shape(binding))
                * self._engine.max_batch_size
            )
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            # 分配主机和设备缓冲区
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # 将设备缓冲区追加到设备绑定中
            bindings.append(int(device_mem))
            # 追加到相应的列表中
            if self._engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def trt_infer(self, data):
        """
        实际推理过程。
        :param data: 预处理后的数据。
        :return: 推理输出。
        """
        # 复制数据到输入内存缓冲区
        [np.copyto(_inp.host, data.ravel()) for _inp in self._input]
        # 推送到设备
        self._device_ctx.push()
        # 将输入数据传输到 GPU
        [
            cuda.memcpy_htod_async(inp.device, inp.host, self._stream)
            for inp in self._input
        ]
        # 运行推理
        self._context.execute_async_v2(
            bindings=self._bindings, stream_handle=self._stream.handle
        )
        # 将预测结果从 GPU 传回主机
        [
            cuda.memcpy_dtoh_async(out.host, out.device, self._stream)
            for out in self._output
        ]
        # 同步流
        self._stream.synchronize()
        # 弹出设备
        self._device_ctx.pop()

        return [out.host.reshape(self._batch_size, -1) for out in self._output[::-1]]

    def preprocess(self, im):
        """
        预处理核心
        :param im: numpy.ndarray 图像。
        :return: 预处理后的图像和原始尺寸信息。
        """
        im, (orig_h, orig_w) = load_image(im)
        return im, (orig_w, orig_h)

    def inference(self, input_frame_array) -> np.ndarray:
        """
        推理核心
        :param input_frame_array: numpy.ndarray 输入图像。
        :return: 网络输出。
        """
        trt_inputs = [input_frame_array]
        trt_inputs = np.vstack(trt_inputs)
        result = self.trt_infer(trt_inputs)
        net_out = result[0].reshape(self.reshape_size[0], self.reshape_size[1])
        return net_out

    def postprocess(self, shape_info: tuple, depth: np.ndarray) -> np.ndarray:
        """
        后处理核心
        :param shape_info: 原始尺寸信息。
        :param depth: 深度图数组。
        :return: 处理后的深度图。
        """
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (shape_info[0], shape_info[1]))
        return depth

    def run(self, input_frame_array: np.ndarray) -> np.ndarray:
        """
        运行推理过程。
        :param input_frame_array: numpy.ndarray 输入图像。
        :return: 推理得到的深度图。
        """
        input_frame_array, shape_info = self.preprocess(input_frame_array)
        # 测量推理时间
        start_time = time.time()  # 记录开始时间
        depth_res = self.inference(input_frame_array)
        end_time = time.time()  # 记录结束时间
        # 计算并打印推理时间
        inference_time = end_time - start_time
        logger.info(f"Inference time for one image: {inference_time:.4f} seconds")

        depth_img = self.postprocess(shape_info, depth_res)

        return depth_img

    def __del__(self):
        self._device_ctx.pop()


if __name__ == "__main__":
    """
        engine_path: TensorRT 引擎路径。
        input_image_path: 输入图像路径。
        output_directory: 输出目录，用于保存深度图。
        grayscale: 是否保存为灰度图。
    """
    input_image_path = r"/home/sunh/6D_ws/Fpose_rgb/demo_data/bj/rgb/000081-color.png"
    output_directory = r"/home/sunh/6D_ws/Fpose_rgb/demo_data/vis_depth"
    engine_path = r"/home/sunh/6D_ws/Fpose_rgb/feature2/depth_anything_v2_vitb.engine"
    grayscale = False

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 加载输入图像
    image = cv2.imread(input_image_path)

    # 加载并运行模型
    model = Dpt(engine_path)
    depth_img = model.run(image)

    # 保存推理结果
    img_name = os.path.basename(input_image_path)
    # 如果为灰度图，保存并显示灰度深度图
    if grayscale:
        output_path = f'{output_directory}/{img_name[:img_name.rfind(".")]}_depth.png'
        cv2.imwrite(output_path, depth_img)  # 保存灰度深度图
        cv2.imshow("Depth Map (Grayscale)", depth_img)  # 显示灰度深度图
    else:
        # 使用 INFERNO 颜色映射生成彩色深度图
        colored_depth = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
        output_path = f'{output_directory}/{img_name[:img_name.rfind(".")]}_depth.png'
        cv2.imwrite(output_path, colored_depth)  # 保存彩色深度图
        cv2.imshow("Depth Map (Colored)", colored_depth)  # 显示彩色深度图

    logger.info(f"Depth map saved to {output_path}")

    # 等待用户按下任意键后关闭显示窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


