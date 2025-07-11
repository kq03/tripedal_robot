# IsaacLab 训练扩展工具

本目录包含IsaacLab项目的训练扩展工具，用于增强强化学习训练过程。

## JointMonitorRunner

JointMonitorRunner 是一个继承自 OnPolicyRunner 的扩展类，它可以在训练过程中定期保存机器人关节电机输出数据（角度、速度、力矩）的图表。

### 特性

- 在训练过程中收集关节电机数据
- 定期保存图表，与模型保存间隔相同（默认每100个迭代）
- 提供直观的关节角度、速度和力矩变化趋势图
- 自动适配不同类型的环境和机器人

### 使用方法

```python
from isaaclab_tasks.runners.joint_monitor_runner import JointMonitorRunner
from isaaclab_tasks.direct.triiple_legs_robot.tripple_legs_robot_env import TLR6Env
from isaaclab_tasks.direct.triiple_legs_robot.tripple_legs_robot_env_cfg import TLR6FlatEnvCfg

# 创建环境
env_cfg = TLR6FlatEnvCfg()
env = TLR6Env(env_cfg)

# 创建JointMonitorRunner
runner = JointMonitorRunner(
    env=env,
    train_cfg=train_cfg,  # 训练配置字典
    log_dir="logs/training",  # 日志保存目录
    device="cuda"  # 设备
)

# 开始训练
runner.learn(num_learning_iterations=10000)
```

### 输出图表

JointMonitorRunner 会在指定的 log_dir 目录下创建一个 joint_data 子目录，保存以下图表：

- `joint_positions_{iteration}.png`: 关节位置变化图
- `joint_velocities_{iteration}.png`: 关节速度变化图
- `joint_torques_{iteration}.png`: 关节力矩变化图

每个图表显示训练过程中最近一段时间内各关节数据的变化趋势。

### 注意事项

1. 确保环境提供关节数据（位置、速度和力矩）
2. 如果环境实现了 `_get_joint_data_from_obs` 方法，JointMonitorRunner 会优先使用该方法获取关节数据
3. 图表保存间隔与模型保存间隔相同，可通过训练配置中的 `save_interval` 参数调整
4. 默认情况下只保存最近1000个数据点，可以在创建JointDataLogger时调整 