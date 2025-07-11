![Isaac Lab](docs/source/_static/isaaclab.jpg)

---

# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows, such as reinforcement learning, imitation learning, and motion planning. Built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html), it combines fast and accurate physics and sensor simulation, making it an ideal choice for sim-to-real transfer in robotics.

Isaac Lab provides developers with a range of essential features for accurate sensor simulation, such as RTX-based cameras, LIDAR, or contact sensors. The framework's GPU acceleration enables users to run complex simulations and computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks. Moreover, Isaac Lab can run locally or be distributed across the cloud, offering flexibility for large-scale deployments.

## Key Features

Isaac Lab offers a comprehensive set of tools and environments designed to facilitate robot learning:
- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.


## Getting Started

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Show & Tell: Share Your Inspiration

We encourage you to utilize our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) area in the
`Discussions` section of this repository. This space is designed for you to:

* Share the tutorials you've created
* Showcase your learning content
* Present exciting projects you've developed

By sharing your work, you'll inspire others and contribute to the collective knowledge
of our community. Your contributions can spark new ideas and collaborations, fostering
innovation in robotics and simulation.

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## Connect with the NVIDIA Omniverse Community

Have a project or resource you'd like to share more widely? We'd love to hear from you! Reach out to the
NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to discuss potential opportunities
for broader dissemination of your work.

Join us in building a vibrant, collaborative ecosystem where creativity and technology intersect. Your
contributions can make a significant impact on the Isaac Lab community and beyond!

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```

# IMU传感器使用指南

本文档提供关于Yesense IMU传感器的使用方法、测试脚本说明和常见问题解决方案。

## 1. 硬件介绍

Yesense IMU（惯性测量单元）是一种精密的姿态传感器，可提供：
- 四元数姿态数据
- 角速度数据
- 加速度数据

该传感器通过USB接口连接到计算机，在Linux系统中表现为串口设备。

## 2. 环境准备

### 2.1 Linux环境

#### 查找IMU设备
连接IMU后，设备通常会被识别为`/dev/ttyUSB0`或类似端口，可以通过以下命令查找：
```bash
# 查看所有USB串口设备
ls /dev/ttyUSB*

# 或者查看系统日志中的串口信息
dmesg | grep tty
```

#### 设置权限
确保当前用户有权限访问串口设备：
```bash
# 临时设置权限
sudo chmod 666 /dev/ttyUSB0

# 永久解决（添加用户到dialout组）
sudo usermod -a -G dialout $USER
# 添加后需要重新登录系统生效
```

### 2.2 依赖安装
确保安装了必要的Python库：
```bash
pip install numpy pyserial
```

## 3. 测试IMU

### 3.1 使用测试脚本

我们提供了`try_IMU.py`脚本用于测试IMU连接和数据读取：

```bash
# 使用默认端口(/dev/ttyUSB0)运行
python try_IMU.py

# 指定端口运行
python try_IMU.py /dev/ttyUSB1

# 指定端口和波特率
python try_IMU.py /dev/ttyUSB0 460800
```

### 3.2 测试输出说明

测试脚本会显示以下数据：
- **四元数**：表示当前IMU的姿态（q0, q1, q2, q3）
- **角速度**：IMU的旋转速率，单位为弧度/秒（rad/s）
- **加速度**：IMU感知到的加速度，单位为米/秒²（m/s²）

正常情况下，静止放置的IMU应该显示：
- 四元数接近[1,0,0,0]（或随姿态变化）
- 角速度接近[0,0,0]
- 加速度接近[0,0,1]（z轴受重力影响）

## 4. 在机器人项目中使用

### 4.1 导入和初始化

```python
from tlr_control.src.sensor import SensorInterface

# 初始化传感器
sensors = SensorInterface(imu_port='/dev/ttyUSB0')

# 读取IMU数据
quaternion, angular_vel = sensors.read_imu()
linear_acc = sensors.get_linear_acceleration()
```

### 4.2 主循环中使用

```python
while True:
    # 读取IMU数据
    quaternion, angular_vel = sensors.read_imu()
    
    # 使用数据进行姿态控制...
    
    # 保持100Hz的控制频率
    time.sleep(0.01)
```

## 5. 常见问题解决

### 5.1 找不到设备

症状：`无法打开端口`或`端口不存在`

解决方法：
1. 检查IMU是否正确连接
2. 检查端口名称：`ls /dev/ttyUSB*`
3. 尝试重新插拔USB

### 5.2 权限问题

症状：`权限被拒绝`

解决方法：
```bash
sudo chmod 666 /dev/ttyUSB0
```

### 5.3 数据异常或不稳定

症状：数据跳变或不合理值

解决方法：
1. 检查线缆连接是否稳定
2. 确认波特率设置正确（默认460800）
3. 远离强磁场或电磁干扰源

### 5.4 多设备冲突

症状：多个设备使用同一COM口

解决方法：
1. 使用`lsof | grep ttyUSB`查看占用情况
2. 确保同一时间只有一个程序访问IMU

## 6. 改进建议

### 6.1 代码改进

修复传感器读取中的潜在问题，如`read_all()`可能返回`None`：
```python
# 在sensor.py中的read_imu()方法中修改
data = self.imu_conn.read_all() or b''  # 确保data不是None
```

### 6.2 健壮性增强

添加断线重连机制：
```python
def reconnect_imu(self):
    if self.imu_conn:
        self.imu_conn.close()
    try:
        self.setup_imu()
        return True
    except:
        return False
```

## 7. 技术支持

如有问题，请联系项目维护人员或参考Yesense官方文档。
