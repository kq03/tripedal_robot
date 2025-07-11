import serial
import time
import numpy as np
import sys
# 导入Yesense解码库
sys.path.append("../library_funcs")  # 确保能找到IMU解码库
from Yesense_Decode import decode_data, open_serial

class SensorInterface:
    def __init__(self, imu_port='/dev/ttyACM1', imu_baud=460800, height_port='/dev/ttyACM0', height_baud=921600, height_offset=0.03):
        """初始化传感器接口
        
        Args:
            imu_port: IMU串口端口
            imu_baud: IMU串口波特率
            height_port: 高度传感器串口端口
            height_baud: 高度传感器串口波特率
            height_offset: 高度传感器安装偏置(m) - 相对于训练时的高度差
        """
        # IMU数据存储
        self.quaternion = [1.0, 0.0, 0.0, 0.0]  # [q0, q1, q2, q3]
        self.angular_velocity = [0.0, 0.0, 0.0]  # [gyro_x, gyro_y, gyro_z]
        self.linear_acceleration = [0.0, 0.0, 0.0]  # [acc_x, acc_y, acc_z]
        
        # 高度数据存储
        self.height = 0.0
        self.prev_height = 0.0
        self.height_measurements = []
        self.height_port = height_port
        self.height_baud = height_baud
        
        # 高度传感器安装偏置补偿
        self.height_offset = height_offset  # 默认0.03m (30mm)
        print(f"高度传感器安装偏置: {self.height_offset:.3f}m ({self.height_offset*1000:.0f}mm)")
        
        # IMU解析缓存
        self.yis_out = {'tid':1, 'roll':0.0, 'pitch':0.0, 'yaw':0.0, 
                        'q0':1.0, 'q1':0.0, 'q2':0.0, 'q3':0.0, 
                        'sensor_temp':25.0, 'acc_x':0.0, 'acc_y':0.0, 'acc_z':1.0, 
                        'gyro_x':0.0, 'gyro_y':0.0, 'gyro_z':0.0,
                        'norm_mag_x':0.0, 'norm_mag_y':0.0, 'norm_mag_z':0.0}
        
        # 初始化传感器
        self.imu_port = imu_port
        self.imu_baud = imu_baud
        self.setup_imu()
        self.setup_height_sensor()
        
    def setup_imu(self):
        """初始化IMU传感器"""
        try:
            self.imu_conn = open_serial(self.imu_port, self.imu_baud)
            print(f"IMU初始化成功：端口={self.imu_port}, 波特率={self.imu_baud}")
        except Exception as e:
            print(f"IMU初始化失败：{e}")
            # 创建一个模拟串口对象，避免程序崩溃
            self.imu_conn = None
            print("使用模拟IMU数据")
    
    def setup_height_sensor(self):
        """初始化STP23高度传感器"""
        try:
            self.height_conn = serial.Serial(
                port=self.height_port,
                baudrate=self.height_baud,
                timeout=1
            )
            print(f"高度传感器初始化成功：端口={self.height_port}, 波特率={self.height_baud}")
            # 等待设备准备
            time.sleep(1)
            # 清空缓冲区
            self.height_conn.reset_input_buffer()
        except Exception as e:
            print(f"高度传感器初始化失败：{e}")
            self.height_conn = None
            print("使用模拟高度数据")
    
    def parse_height_data(self, data):
        """解析STP23传感器数据
        
        Args:
            data: 原始字节数据列表
            
        Returns:
            list: 测量点列表，每个点为(距离, 强度)元组
        """
        # 检查数据有效性
        if not data or len(data) < 11:  # 最小有效数据长度
            return []
        
        # 检查起始符
        if data[0] != 0x54:
            return []
        
        measurements = []
        
        try:
            # 从第7个字节开始解析每个测量数据点
            for i in range(6, len(data)-5, 3):
                # 确保有足够的数据解析一个完整的测量点(3字节)
                if i + 2 >= len(data):
                    break
                    
                # 解析距离值（更简洁的位运算方式）
                distance = (data[i+1] << 8) | data[i]
                intensity = data[i+2]
                
                # 添加有效测量点
                if 0 < distance < 2000:  # 合理范围检查
                    measurements.append((distance, intensity))
                    
        except Exception as e:
            print(f"高度数据解析错误: {e}")
            return []
        
        return measurements

    # def parse_height_data(self, data):
    #     """解析STP23传感器数据
        
    #     Args:
    #         data: 原始字节数据列表
            
    #     Returns:
    #         list: 测量点列表，每个点为(距离, 强度)元组
    #     """
    #     # 检查起始符 - 先检查数据长度再访问索引
    #     if not data or len(data) == 0 or data[0] != 0x54:
    #         return []
        
    #     measurements = []
        
    #     # 确保数据长度足够进行解析（至少需要11个字节：起始符+头部+至少一个测量点）
    #     if len(data) < 11:
    #         return []
            
    #     # 从第7个字节开始解析每个测量数据点
    #     for i in range(6, len(data)-5, 3):
    #         # 更严格的边界检查
    #         if i + 2 < len(data) and i + 1 < len(data) and i < len(data):
    #             try:
    #                 distance1 = data[i + 1]
    #                 distance1 = hex(distance1)  # 将十进制整数转换为十六进制字符串表示
    #                 interval1 = distance1[2:].upper()  # (去掉十六进制前面的0x)
    #                 str1 = str(interval1)  # 将 interval1 转为字符串

    #                 distance2 = data[i]
    #                 distance2 = hex(distance2)
    #                 interval2 = distance2[2:].upper()
    #                 str2 = str(interval2)

    #                 string1 = str1 + str2  # 拼接高字节和低字节（用字符串来表示）

    #                 distance = int(string1, 16)  # 将一个字符串按照16进制的方式解析并转换为整数

    #                 intensity = data[i + 2]  # 1字节信号强度

    #                 measurements.append((distance, intensity))
    #             except (IndexError, ValueError) as e:
    #                 # 如果数据解析出错，跳过这个测量点
    #                 continue
        
    #     return measurements
    
    def read_imu(self):
        """读取IMU数据，返回四元数和角速度"""
        if self.imu_conn is None or not self.imu_conn.is_open:
            return self.quaternion, self.angular_velocity
            
        try:
            # 更保守的数据读取策略
            if self.imu_conn.in_waiting > 0:
                data = self.imu_conn.read_all()
                if data is None:
                    return self.quaternion, self.angular_velocity
                    
                # 只有当数据长度合理时才尝试解析
                if len(data) >= 7:  # 最小协议长度
                    try:
                        if decode_data(data, len(data), self.yis_out, False):
                            self.quaternion = [
                                self.yis_out['q0'], 
                                self.yis_out['q1'], 
                                self.yis_out['q2'], 
                                self.yis_out['q3']
                            ]
                            self.angular_velocity = [
                                self.yis_out['gyro_x'], 
                                self.yis_out['gyro_y'], 
                                self.yis_out['gyro_z']
                            ]
                            self.linear_acceleration = [
                                self.yis_out['acc_x'], 
                                self.yis_out['acc_y'], 
                                self.yis_out['acc_z']
                            ]
                    except (IndexError, ValueError, TypeError) as parse_error:
                        # 解析错误时静默跳过，不重连
                        pass
                        
        except Exception as e:
            # 只有在严重错误时才重连
            if "device" in str(e).lower() or "closed" in str(e).lower():
                print(f"IMU连接错误: {e}")
                self.reconnect_imu()
        
        return self.quaternion, self.angular_velocity
    # def read_imu(self):
    #     """读取IMU数据，返回四元数和角速度
        
    #     Returns:
    #         quaternion: [q0, q1, q2, q3]四元数
    #         angular_velocity: [gyro_x, gyro_y, gyro_z]角速度(rad/s)
    #     """
    #     if self.imu_conn is not None and self.imu_conn.is_open:
    #         try:
    #             # 读取IMU串口数据 - 模拟原例程的方式
    #             data = self.imu_conn.read_all()
    #             if data is None:
    #                 data = b''
    #             num = len(data)
                
    #             if num > 0:
    #                 # 解析IMU数据
    #                 ret = decode_data(data, num, self.yis_out, False)
    #                 if ret:
    #                     # 更新四元数和角速度
    #                     self.quaternion = [
    #                         self.yis_out['q0'], 
    #                         self.yis_out['q1'], 
    #                         self.yis_out['q2'], 
    #                         self.yis_out['q3']
    #                     ]
    #                     self.angular_velocity = [
    #                         self.yis_out['gyro_x'], 
    #                         self.yis_out['gyro_y'], 
    #                         self.yis_out['gyro_z']
    #                     ]
    #                     self.linear_acceleration = [
    #                         self.yis_out['acc_x'], 
    #                         self.yis_out['acc_y'], 
    #                         self.yis_out['acc_z']
    #                     ]
    #         except Exception as e:
    #             print(f"IMU数据读取错误：{e}")
        
    #     return self.quaternion, self.angular_velocity
    
    def read_height_sensor(self):
        """读取高度传感器数据
        
        Returns:
            height: 当前高度(m)
            prev_height: 上一帧高度(m)
        """
        # 保存上一帧高度
        self.prev_height = self.height
        
        # 读取当前高度
        self.height = self._read_raw_height()
        
        return self.height, self.prev_height
    
    def _read_raw_height(self):
        """读取STP23高度传感器数据
        
        Returns:
            float: 当前高度(m) - 与仿真训练保持一致的米制单位，已减去安装偏置
        """
        # 检查连接状态
        if not hasattr(self, 'height_conn') or self.height_conn is None or not self.height_conn.is_open:
            # 默认高度也要减去偏置
            return getattr(self, 'height', 0.1 - self.height_offset)
        
        try:
            if self.height_conn.in_waiting > 0:
                raw_data = self.height_conn.read(self.height_conn.in_waiting)
                if not raw_data:
                    return getattr(self, 'height', 0.1 - self.height_offset)
                    
                measurements = self.parse_height_data(list(bytearray(raw_data)))
                
                if not measurements:
                    return getattr(self, 'height', 0.1 - self.height_offset)
                    
                # 计算有效距离平均值
                valid_distances = [
                    d for d, intensity in measurements 
                    if intensity > 10 and 0 < d < 2000  # 毫米范围筛选
                ]
                
                if valid_distances:
                    # 单位转换：毫米 -> 米
                    avg_distance_mm = sum(valid_distances) / len(valid_distances)
                    avg_distance_m = avg_distance_mm / 1000.0  # mm -> m
                    
                    # 关键：减去安装偏置，使高度与训练时保持一致
                    corrected_height = avg_distance_m - self.height_offset
                    
                    # 确保高度不为负数
                    return max(0.001, corrected_height)  # 最小高度1mm
                    
        except serial.SerialException as e:
            print(f"高度传感器串口错误: {e}")
            self.reconnect_height_sensor()
        except Exception as e:
            print(f"高度传感器读取错误: {e}")
        
        # 默认高度也要减去偏置
        return max(0.001, getattr(self, 'height', 0.1) - self.height_offset)

    # def _read_raw_height(self):
    #     """读取STP23高度传感器数据
        
    #     Returns:
    #         float: 当前高度(m)
    #     """
    #     if hasattr(self, 'height_conn') and self.height_conn is not None and self.height_conn.is_open:
    #         try:
    #             if self.height_conn.in_waiting > 0:
    #                 # 读取所有可用数据
    #                 raw_data = self.height_conn.read(self.height_conn.in_waiting)
    #                 # 转换为列表以便处理
    #                 data = list(bytearray(raw_data))
    #                 # 解析测量数据
    #                 measurements = self.parse_height_data(data)
                    
    #                 if measurements:
    #                     # 存储测量数据
    #                     self.height_measurements = measurements
                        
    #                     # 计算平均距离，忽略异常值
    #                     valid_distances = []
    #                     for distance, intensity in measurements:
    #                         # 过滤掉强度太低或距离异常的读数
    #                         if intensity > 10 and 0 < distance < 2000:
    #                             valid_distances.append(distance)
                        
    #                     if valid_distances:
    #                         # 将毫米转换为米
    #                         avg_distance = sum(valid_distances) / len(valid_distances) / 1000.0
    #                         return avg_distance
    #         except Exception as e:
    #             print(f"高度传感器读取错误：{e}")
        
    #     # 如果读取失败或没有传感器，返回上一个有效高度或默认值
    #     if hasattr(self, 'height') and self.height > 0:
    #         return self.height
    #     return 0.1  # 默认高度值
    
    def get_linear_acceleration(self):
        """获取线性加速度数据
        
        Returns:
            acceleration: [acc_x, acc_y, acc_z]加速度(m/s^2)
        """
        return self.linear_acceleration
    
    def close(self):
        """关闭所有传感器连接"""
        if hasattr(self, 'imu_conn') and self.imu_conn is not None:
            self.imu_conn.close()
            print("IMU连接已关闭")
        
        if hasattr(self, 'height_conn') and self.height_conn is not None:
            self.height_conn.close()
            print("高度传感器连接已关闭")
            
    def reconnect_imu(self):
        """尝试重新连接IMU
        
        Returns:
            bool: 重连是否成功
        """
        # 添加重连间隔控制
        current_time = time.time()
        if not hasattr(self, '_last_imu_reconnect'):
            self._last_imu_reconnect = 0
            
        # 限制重连频率：至少间隔5秒
        if current_time - self._last_imu_reconnect < 5.0:
            return False
            
        self._last_imu_reconnect = current_time
        
        try:
            if self.imu_conn is not None:
                self.imu_conn.close()
            time.sleep(0.1)  # 短暂等待
            self.setup_imu()
            return self.imu_conn is not None and self.imu_conn.is_open
        except Exception as e:
            print(f"IMU重连失败: {e}")
            return False
    
    def reconnect_height_sensor(self):
        """尝试重新连接高度传感器
        
        Returns:
            bool: 重连是否成功
        """
        if hasattr(self, 'height_conn') and self.height_conn is not None:
            self.height_conn.close()
        try:
            self.setup_height_sensor()
            # 测试能否读取数据
            return self.height_conn is not None and self.height_conn.is_open
        except Exception as e:
            print(f"高度传感器重连失败: {e}")
            return False
        
    def check_sensor_health(self):
        """检查传感器连接状态"""
        imu_ok = self.imu_conn is not None and self.imu_conn.is_open
        height_ok = hasattr(self, 'height_conn') and self.height_conn is not None and self.height_conn.is_open
        
        if not imu_ok:
            print("警告: IMU连接异常")
        if not height_ok:
            print("警告: 高度传感器连接异常")
            
        return imu_ok and height_ok