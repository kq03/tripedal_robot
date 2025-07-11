import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化CUDA
import os
import onnx

class PolicyModel:
    def __init__(self, model_path, device="cuda", fp16=True, max_batch_size=1):
        """初始化策略网络模型（TensorRT版）
        Args:
            model_path: .onnx模型文件路径或.engine文件路径
            device: 推理设备，目前只支持"cuda"
            fp16: 是否使用FP16精度推理（Jetson推荐使用）
            max_batch_size: 最大批次大小
        """
        self.device = device
        self.fp16 = fp16
        self.max_batch_size = max_batch_size
        self.input_dim = 45  # 观测向量维度

        if device != "cuda":
            raise ValueError("TensorRT只支持CUDA设备")
        
        # 根据文件扩展名判断是否需要转换
        if model_path.endswith('.onnx'):
            print(f"检测到ONNX文件，将转换为TensorRT Engine...")
            engine_path = model_path.replace('.onnx', '.engine')
            self.engine_path = engine_path
            
            # 检查是否已存在engine文件
            if os.path.exists(engine_path):
                print(f"发现现有Engine文件: {engine_path}，直接加载")
                self.engine = self._load_engine(engine_path)
            else:
                self.engine = self._convert_onnx_to_tensorrt(model_path, engine_path)
        
        elif model_path.endswith('.engine') or model_path.endswith('.trt'):
            print(f"加载现有TensorRT Engine文件...")
            self.engine_path = model_path
            self.engine = self._load_engine(model_path)
        else:
            raise ValueError("不支持的模型文件格式，请使用.onnx或.engine文件")
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 分配GPU内存
        self._allocate_buffers()
        
        print(f"TensorRT模型已加载到设备: {device}")
        print(f"使用精度: {'FP16' if fp16 else 'FP32'}")

    def _convert_onnx_to_tensorrt(self, onnx_path, engine_path):
        """将ONNX模型转换为TensorRT Engine"""
        print("开始ONNX到TensorRT转换...")
        
        # 创建TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # 创建builder
        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()
        
        # 设置最大工作空间大小（1GB）
        config.max_workspace_size = 1 << 30
        
        # 如果支持且启用FP16，设置FP16模式
        if self.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("启用FP16优化")
        
        # 创建网络定义
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # 解析ONNX模型
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("解析ONNX模型失败:")
                for error in range(parser.num_errors):
                    print(f"  {parser.get_error(error)}")
                raise RuntimeError("ONNX解析失败")
        
        print("ONNX模型解析成功")
        
        # 设置输入形状（动态batch大小支持）
        input_tensor = network.get_input(0)
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, 
                         (1, self.input_dim),      # min
                         (self.max_batch_size, self.input_dim),  # opt
                         (self.max_batch_size, self.input_dim))  # max
        config.add_optimization_profile(profile)
        
        # 构建引擎
        print("开始构建TensorRT引擎...")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("TensorRT引擎构建失败")
        
        # 保存引擎文件
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        print(f"TensorRT引擎已保存到: {engine_path}")
        
        return engine

    def _load_engine(self, engine_path):
        """加载TensorRT引擎"""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError(f"无法加载TensorRT引擎: {engine_path}")
        
        print(f"TensorRT引擎加载成功: {engine_path}")
        return engine

    def _allocate_buffers(self):
        """分配GPU和CPU内存缓冲区"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            # 获取绑定形状，处理动态形状
            shape = self.engine.get_binding_shape(binding)
            if -1 in shape:  # 动态维度，使用max_batch_size
                shape = tuple(self.max_batch_size if dim == -1 else dim for dim in shape)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配CPU和GPU内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def preprocess_observation(self, obs_dict):
        """预处理观测数据，输出numpy数组"""
        observation = []
        observation.extend(obs_dict["quaternion"])     # 4
        observation.extend(obs_dict["linear_vel"])     # 3
        observation.extend(obs_dict["angular_vel"])    # 3
        observation.extend(obs_dict["joint_pos"])      # 6
        observation.extend(obs_dict["joint_vel"])      # 6
        observation.append(obs_dict["height"])         # 1
        observation.append(obs_dict["prev_height"])    # 1
        observation.extend(obs_dict["prev_joint_vel"]) # 6
        observation.extend(obs_dict["joint_torque"])   # 6
        observation.extend(obs_dict["prev_action"])    # 6
        observation.extend(obs_dict["commands"])       # 3
        return np.array(observation, dtype=np.float32)

    def predict(self, observation):
        """TensorRT模型推理"""
        if isinstance(observation, dict):
            observation = self.preprocess_observation(observation)
        
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]  # 增加batch维
        
        # 将输入数据复制到GPU
        np.copyto(self.inputs[0]['host'], observation.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 设置动态形状
        self.context.set_binding_shape(0, observation.shape)
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 将结果从GPU复制回CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        # 解析输出
        output_shape = self.context.get_binding_shape(1)
        action = self.outputs[0]['host'][:np.prod(output_shape)].reshape(output_shape)
        
        return action.squeeze()

    def set_mode(self, mode):
        if mode not in ["walk", "jump"]:
            raise ValueError("模式必须是'walk'或'jump'")
        self.mode = mode

    def get_engine_info(self):
        """获取引擎信息"""
        info = {
            "engine_path": self.engine_path,
            "max_batch_size": self.max_batch_size,
            "fp16_enabled": self.fp16,
            "num_bindings": self.engine.num_bindings,
            "input_shapes": [],
            "output_shapes": []
        }
        
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            binding_shape = self.engine.get_binding_shape(i)
            binding_dtype = self.engine.get_binding_dtype(i)
            
            if self.engine.binding_is_input(i):
                info["input_shapes"].append({
                    "name": binding_name,
                    "shape": binding_shape,
                    "dtype": str(binding_dtype)
                })
            else:
                info["output_shapes"].append({
                    "name": binding_name, 
                    "shape": binding_shape,
                    "dtype": str(binding_dtype)
                })
        
        return info

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine
