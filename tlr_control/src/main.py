# 
import sys
import time
import numpy as np
import os
from model import PolicyModel
from motors import MotorInterface
from sensor import SensorInterface

# 全局变量存储线速度估计值
global_linear_vel = np.zeros(3)
# 上一次更新的时间戳
last_update_time = time.time()
# 机器人模式 - 默认为爬行模式
robot_mode = "walk"

# 全局对象变量 - 在initialize_system中初始化
motors = None
sensors = None
policy = None

# 数据显示相关变量
display_counter = 0
display_interval = 10  # 每10个循环显示一次数据 (可通过命令行参数调节)

def set_display_frequency(freq_hz):
    """设置数据显示频率
    
    Args:
        freq_hz: 显示频率 (Hz)，如果为0则禁用显示
    """
    global display_interval
    if freq_hz <= 0:
        display_interval = 0  # 禁用显示
    else:
        # 控制频率约100Hz，所以显示间隔 = 100 / 显示频率
        display_interval = max(1, int(100 / freq_hz))
    print(f"数据显示频率设置为: {freq_hz}Hz (每{display_interval}个循环显示一次)")

def display_real_time_data(obs_dict, action, motor_debug_info=None):
    """实时显示控制和传感器数据"""
    global display_counter, display_interval
    
    # 如果禁用显示
    if display_interval <= 0:
        return
    
    display_counter += 1
    if display_counter % display_interval != 0:
        return
    
    print("\n" + "="*80)
    print(f"实时数据显示 (循环 #{display_counter}) - 时间: {time.strftime('%H:%M:%S')}")
    print("="*80)
    
    # 1. 传感器数据显示 (处理后的数据)
    print("📊 传感器数据 (处理后):")
    print(f"  IMU四元数    : [{obs_dict['quaternion'][0]:.3f}, {obs_dict['quaternion'][1]:.3f}, "
          f"{obs_dict['quaternion'][2]:.3f}, {obs_dict['quaternion'][3]:.3f}]")
    print(f"  角速度 (rad/s): [{obs_dict['angular_vel'][0]:.3f}, {obs_dict['angular_vel'][1]:.3f}, {obs_dict['angular_vel'][2]:.3f}]")
    print(f"  估计线速度(m/s): [{obs_dict['linear_vel'][0]:.3f}, {obs_dict['linear_vel'][1]:.3f}, {obs_dict['linear_vel'][2]:.3f}]")
    print(f"  高度 (m)     : {obs_dict['height']:.4f} (已减去35mm偏置)")
    print(f"  前一高度 (m) : {obs_dict['prev_height']:.4f}")
    
    # 2. 关节状态数据 (弧度制，与仿真训练一致)
    print("\n🦾 关节状态 (仿真训练单位 - 弧度制):")
    joint_names = ["L1_thigh", "L2_thigh", "L3_thigh", "L1_shin", "L2_shin", "L3_shin"]
    print("  关节位置 (rad):")
    for i, (name, pos) in enumerate(zip(joint_names, obs_dict['joint_pos'])):
        print(f"    {name:8}: {pos:7.3f} rad ({pos*180/np.pi:6.1f}°)")
    
    print("  关节速度 (rad/s):")
    for i, (name, vel) in enumerate(zip(joint_names, obs_dict['joint_vel'])):
        print(f"    {name:8}: {vel:7.3f} rad/s")
    
    print("  关节力矩 (N·m):")
    for i, (name, torque) in enumerate(zip(joint_names, obs_dict['joint_torque'])):
        print(f"    {name:8}: {torque:7.3f} N·m")
    
    # 3. 策略输出和电机控制
    print("\n🎯 策略输出 (动作命令，弧度制):")
    for i, (name, act) in enumerate(zip(joint_names, action)):
        print(f"    {name:8}: {act:7.3f}")
    
    # 4. 电机实际控制数据 (如果有调试信息)
    if motor_debug_info:
        print("\n⚙️  电机控制详情:")
        print("  经过scale后的目标位置 (弧度制):")
        for i, (name, pos) in enumerate(zip(joint_names, motor_debug_info['target_positions_rad'])):
            print(f"    {name:8}: {pos:7.3f} rad ({pos*180/np.pi:6.1f}°)")
        
        print("  发送给电机的角度 (硬件要求的角度制):")
        for i, (name, deg) in enumerate(zip(joint_names, motor_debug_info['target_degrees'])):
            print(f"    {name:8}: {deg:6.1f}°")
        
        print("  缩放系数:")
        for i, (name, scale) in enumerate(zip(joint_names, motor_debug_info['action_scales'])):
            print(f"    {name:8}: {scale:7.3f}")
        
        # CAN通信状态
        can_success = motor_debug_info.get('can_success_count', 0)
        can_errors = motor_debug_info.get('can_error_count', 0)
        total_motors = len(joint_names)
        success_rate = (can_success / total_motors * 100) if total_motors > 0 else 0
        
        print(f"  CAN通信状态: 成功{can_success}/{total_motors} ({success_rate:.1f}%), 累计错误{can_errors}")
        
        if can_errors > 5:
            print("    ⚠️  CAN通信错误较多，请检查连接")
        elif success_rate < 80:
            print("    ⚠️  CAN通信成功率较低")
        else:
            print("    ✅ CAN通信正常")
    
    # 5. 命令和模式信息
    print(f"\n🎮 运动命令: 前进={obs_dict['commands'][0]:.2f}, 侧移={obs_dict['commands'][1]:.2f}, 转向={obs_dict['commands'][2]:.2f}")
    print(f"🤖 机器人模式: {robot_mode}")
    
    print("="*80)

def get_observation_dict():
    """获取观测数据字典"""
    global sensors, motors
    
    # 确保sensors和motors已初始化
    if sensors is None or motors is None:
        raise RuntimeError("传感器或电机未初始化")
    
    # 读取传感器数据
    quaternion, angular_vel = sensors.read_imu()
    height, prev_height = sensors.read_height_sensor()
    positions, velocities, torques = motors.read_joint_states()
    
    # 默认前进命令
    commands = [0.3, 0.0, 0.0]  # 前进速度命令
    
    # 返回观测字典
    return {
        "quaternion": quaternion,
        "linear_vel": estimate_linear_vel(),  
        "angular_vel": angular_vel,
        "joint_pos": positions,
        "joint_vel": velocities,
        "height": height,
        "prev_height": prev_height,
        "prev_joint_vel": motors.prev_velocities,
        "joint_torque": torques,
        "prev_action": motors.prev_actions,
        "commands": commands
    }

def estimate_linear_vel():
    """估计机器人线速度"""
    global global_linear_vel, last_update_time, sensors
    
    # 确保sensors已初始化
    if sensors is None:
        return [0.0, 0.0, 0.0]
    
    current_time = time.time()
    dt = current_time - last_update_time
    
    if dt < 0.001:
        return [global_linear_vel[0], 0.0, 0.0]
    
    try:
        linear_acc = np.array(sensors.get_linear_acceleration())
        quaternion, _ = sensors.read_imu()
        q0, q1, q2, q3 = quaternion
        
        # 四元数旋转计算
        gravity_world = np.array([0, 0, 9.81])
        gravity_body = np.array([
            (1 - 2*q2*q2 - 2*q3*q3)*gravity_world[0] + 2*(q1*q2 - q0*q3)*gravity_world[1] + 2*(q1*q3 + q0*q2)*gravity_world[2],
            2*(q1*q2 + q0*q3)*gravity_world[0] + (1 - 2*q1*q1 - 2*q3*q3)*gravity_world[1] + 2*(q2*q3 - q0*q1)*gravity_world[2],
            2*(q1*q3 - q0*q2)*gravity_world[0] + 2*(q2*q3 + q0*q1)*gravity_world[1] + (1 - 2*q1*q1 - 2*q2*q2)*gravity_world[2]
        ])
        
        linear_acc_no_gravity = linear_acc - gravity_body
        delta_v = linear_acc_no_gravity * dt
        global_linear_vel += delta_v
        
        # 低通滤波
        alpha = 0.8
        global_linear_vel = alpha * global_linear_vel + (1 - alpha) * global_linear_vel
        
        # 漂移修正
        accel_magnitude = np.linalg.norm(linear_acc_no_gravity)
        speed = np.linalg.norm(global_linear_vel)
        if accel_magnitude < 0.1 and speed < 0.05:
            decay_factor = np.exp(-dt / 0.5)
            global_linear_vel *= decay_factor
        
        # 速度限幅
        max_velocity = 2.0
        velocity_magnitude = np.linalg.norm(global_linear_vel)
        if velocity_magnitude > max_velocity:
            global_linear_vel = global_linear_vel * (max_velocity / velocity_magnitude)
            
    except Exception as e:
        print(f"线速度估计错误: {e}")
    
    last_update_time = current_time
    return [global_linear_vel[0], 0.0, 0.0]

def initialize_system():
    """初始化整个系统"""
    global motors, sensors, policy
    
    # 设置默认模型路径
    model_path = find_model_path()
    print(f"使用模型: {model_path}")
    
    # 根据模型文件名确定当前模态
    if "jump" in model_path.lower():
        print("当前运行跳跃模态")
        # 修改观测函数以返回零速度命令
        global get_observation_dict
        original_get_obs = get_observation_dict
        def get_observation_dict_jump():
            obs = original_get_obs()
            obs["commands"] = [0.0, 0.0, 0.0]
            return obs
        get_observation_dict = get_observation_dict_jump
    else:
        print("当前运行爬行模态")
    
    # 初始化硬件和模型
    try:
        # 重要：默认位置使用弧度制，与仿真训练保持一致
        # 这里的值应该根据您的机器人实际情况调整
        default_positions_rad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 弧度制
        
        print("单位转换说明:")
        print("  - 高度传感器: 毫米 -> 米 (与仿真训练一致)")
        print("  - 电机控制: 弧度 -> 角度 (硬件要求)")
        print("  - 电机反馈: 角度 -> 弧度 (与仿真训练一致)")
        print(f"  - 默认关节位置: {default_positions_rad} (弧度制)")
        
        motors = MotorInterface(default_positions=default_positions_rad)
        sensors = SensorInterface(
            imu_port='/dev/ttyACM1',
            imu_baud=460800,
            height_port='/dev/ttyACM0',
            height_baud=921600,
            height_offset=0.035  # 30mm安装偏置补偿
        )
        
        # 检查传感器连接
        if not sensors.check_sensor_health():
            print("警告: 传感器连接异常，但继续运行...")
            
        # 初始化策略模型
        device = "cuda"
        print(f"使用设备: {device} (TensorRT)")
        policy = PolicyModel(model_path, device=device, fp16=True)
        
        print("系统初始化完成")
        return True
        
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return False

def find_model_path():
    """查找模型文件路径，处理命令行参数"""
    model_path = None
    display_freq = 1.0  # 默认1Hz显示频率
    
    # 处理命令行参数
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--display-freq' and i + 1 < len(args):
            try:
                display_freq = float(args[i + 1])
                i += 2
            except ValueError:
                print(f"无效的显示频率: {args[i + 1]}")
                i += 1
        elif args[i] == '--no-display':
            display_freq = 0
            i += 1
        elif not args[i].startswith('--'):
            # 这是模型路径
            model_path = args[i]
            i += 1
        else:
            i += 1
    
    # 设置显示频率
    set_display_frequency(display_freq)
    
    # 查找模型文件
    possible_paths = [
        "tlr_control/models/model_walk.onnx",
        "models/model_walk.onnx",
        "../models/model_walk.onnx",
        "model_walk.onnx"
    ]
    
    if model_path:
        possible_paths.insert(0, model_path)
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError("找不到模型文件")

def main_control_loop():
    """主控制循环"""
    global sensors, motors, policy
    
    loop_count = 0
    start_time = time.time()
    last_health_check = time.time()
    sensor_error_count = 0
    max_sensor_errors = 5
    
    print("初始化完成，开始控制循环...")
    print(f"目标控制频率: 50Hz (严格时间控制)")
    
    # 精确频率控制
    target_period = 0.02  # 50Hz = 20ms
    next_time = time.time()
    
    try:
        while True:
            try:
                current_time = time.time()
                
                # 定期健康检查 (每秒一次)
                if current_time - last_health_check > 1.0:
                    if not sensors.check_sensor_health():
                        sensor_error_count += 1
                        print(f"传感器健康检查失败 ({sensor_error_count}/{max_sensor_errors})")
                        
                        if sensor_error_count >= max_sensor_errors:
                            print("传感器错误过多，尝试重新初始化...")
                            sensors.close()
                            time.sleep(1)
                            sensors = SensorInterface(
                                imu_port='/dev/ttyACM1',
                                imu_baud=460800,
                                height_port='/dev/ttyACM0',
                                height_baud=921600,
                                height_offset=0.035  # 30mm安装偏置补偿
                            )
                            sensor_error_count = 0
                            
                    last_health_check = current_time
                
                # 1. 获取观测数据
                obs_dict = get_observation_dict()
                
                # 2. 策略推理
                action = policy.predict(obs_dict)
                
                # 3. 执行动作，获取电机调试信息
                motor_debug_info = motors.set_positions(action)
                
                # 4. 实时显示数据
                display_real_time_data(obs_dict, action, motor_debug_info)
                
                # 5. 控制循环频率
                loop_count += 1
                if loop_count % 50 == 0:  # 每50个循环报告一次频率
                    elapsed = time.time() - start_time
                    actual_freq = loop_count / elapsed
                    print(f"实际控制频率: {actual_freq:.1f}Hz")
                
                # 6. 精确的50Hz控制
                next_time += target_period
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # 如果已经超时，重新同步
                    next_time = time.time()
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"控制循环错误: {e}")
                time.sleep(0.5)  # 错误后短暂等待
                
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 清理资源
        motors.emergency_stop()
        sensors.close()
        print("已安全退出")

if __name__ == "__main__":
    print("TLR控制系统")
    print("命令行参数:")
    print("  <model_path>           : 指定模型文件路径")
    print("  --display-freq <freq>  : 设置数据显示频率 (Hz)，默认1Hz")
    print("  --no-display           : 禁用数据显示")
    print()
    
    if not initialize_system():
        sys.exit(1)
    
    main_control_loop()