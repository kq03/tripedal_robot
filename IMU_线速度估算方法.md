# IMU线速度估算技术文档

## 1. 基本原理

线速度估算是机器人导航和控制的关键环节。在没有直接速度传感器的情况下，基于IMU数据的速度估算提供了一种可行方案。本文档详细介绍通过IMU加速度数据估算线速度的方法、误差控制策略和数学原理。

### 1.1 问题定义

在强化学习环境中，机器人观测空间包含线速度信息。然而，在实际硬件上，我们通常只有IMU提供的加速度和姿态数据，需要通过计算推导出线速度。

### 1.2 核心思路

线速度估算基于物理学基本原理：速度是加速度的积分。因此，我们可以将IMU测量的加速度进行积分，得到速度变化。然而，这一过程存在多种误差来源，需要应用各种技术来减少累积误差。

## 2. 数学基础

### 2.1 四元数表示与旋转

四元数是表示3D旋转的高效方法，比欧拉角更稳定，比旋转矩阵更紧凑。

#### 2.1.1 四元数定义

四元数 $q = [q_0, q_1, q_2, q_3] = [w, x, y, z]$ 其中:
- $w$: 实部
- $x, y, z$: 虚部

单位四元数表示旋转：$w^2 + x^2 + y^2 + z^2 = 1$

#### 2.1.2 四元数旋转公式

使用四元数对向量 $v$ 进行旋转的公式：

$v' = q \otimes v \otimes q^{-1}$

其中 $\otimes$ 表示四元数乘法，$q^{-1}$ 是 $q$ 的逆四元数。

在代码中，这一旋转的具体实现为：

```python
def quat_rotate(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    
    result = np.zeros(3)
    result[0] = (1 - 2*y*y - 2*z*z)*vx + 2*(x*y - w*z)*vy + 2*(x*z + w*y)*vz
    result[1] = 2*(x*y + w*z)*vx + (1 - 2*x*x - 2*z*z)*vy + 2*(y*z - w*x)*vz
    result[2] = 2*(x*z - w*y)*vx + 2*(y*z + w*x)*vy + (1 - 2*x*x - 2*y*y)*vz
    
    return result
```

### 2.2 积分方法

#### 2.2.1 基本积分公式

速度 $v$ 与加速度 $a$ 的关系：

$v(t) = v(t_0) + \int_{t_0}^{t} a(\tau) d\tau$

离散形式：

$v_k = v_{k-1} + a_k \cdot \Delta t$

#### 2.2.2 梯形积分法

为提高精度，可使用梯形法：

$v_k = v_{k-1} + \frac{1}{2}(a_k + a_{k-1}) \cdot \Delta t$

## 3. 重力补偿

### 3.1 重力影响

IMU测量的加速度包含惯性加速度和重力加速度的综合影响。我们需要移除重力分量，只保留运动产生的加速度。

### 3.2 补偿方法

我们采用"直接在世界坐标系中补偿重力"的方法，步骤如下：

1. 获取IMU加速度数据和姿态四元数
2. 使用四元数将IMU加速度转换到世界坐标系
3. 在世界坐标系中直接减去重力向量
4. 在世界坐标系中进行后续计算

```python
# 获取IMU数据
linear_acc = np.array(sensors.get_linear_acceleration())  # IMU坐标系下的加速度
quaternion, _ = sensors.read_imu()
q0, q1, q2, q3 = quaternion  # 四元数[w, x, y, z]

# 使用四元数共轭进行反向旋转（从IMU坐标系到世界坐标系）
linear_acc_world = quat_rotate([q0, -q1, -q2, -q3], linear_acc)

# 在世界坐标系中减去重力
gravity_world = np.array([0, 0, 9.81])  # 世界坐标系下的重力向量
linear_acc_no_gravity = linear_acc_world - gravity_world
```

与传统的"在IMU坐标系中补偿重力"方法相比，这种方法更加简洁，避免了最后还需要将速度从IMU坐标系转换到世界坐标系的步骤。

## 4. 积分漂移问题与解决方案

### 4.1 漂移原因

1. **传感器噪声**：IMU数据存在随机噪声
2. **采样误差**：离散采样导致积分误差
3. **偏置误差**：IMU传感器存在零点漂移
4. **积分累积**：小误差通过积分不断累加

### 4.2 解决方案

#### 4.2.1 低通滤波

使用低通滤波减少高频噪声：

$v_{\text{filtered}} = (1 - \alpha) \cdot v_{\text{previous}} + \alpha \cdot v_{\text{current}}$

其中 $\alpha$ 是滤波系数（0-1之间）。

#### 4.2.2 零速度检测与校正

当检测到机器人可能静止时（加速度接近零），强制速度逐渐衰减至零：

```python
accel_magnitude = np.linalg.norm(linear_acc_no_gravity)
if accel_magnitude < threshold:  # 可能静止
    decay_factor = np.exp(-dt / time_constant)
    velocity *= decay_factor
```

#### 4.2.3 速度限幅

防止速度估计发散，设定最大速度阈值：

```python
max_velocity = 2.0  # 最大速度限制
velocity_magnitude = np.linalg.norm(velocity)
if velocity_magnitude > max_velocity:
    velocity = velocity * (max_velocity / velocity_magnitude)
```

#### 4.2.4 互补滤波

结合其他信息源（如视觉里程计、车轮编码器等）进行互补滤波。

## 5. 实现算法流程

### 5.1 完整算法步骤

1. **初始化**：设置初始速度和时间戳
2. **数据获取**：读取IMU加速度和姿态数据
3. **坐标转换**：将IMU加速度转换到世界坐标系
4. **重力补偿**：在世界坐标系中减去重力
5. **速度积分**：积分加速度得到速度增量
6. **滤波处理**：应用低通滤波减少噪声
7. **漂移校正**：检测静止状态并校正速度
8. **限幅保护**：防止速度估计发散

### 5.2 核心代码实现

```python
def estimate_linear_vel():
    global global_linear_vel, last_update_time
    
    # 计算时间增量
    current_time = time.time()
    dt = current_time - last_update_time
    
    # 防止dt过小或首次运行
    if dt < 0.001:
        return global_linear_vel.tolist()
    
    # 获取IMU数据
    linear_acc = np.array(sensors.get_linear_acceleration())  # IMU坐标系下的加速度
    quaternion, _ = sensors.read_imu()
    q0, q1, q2, q3 = quaternion  # 四元数[w, x, y, z]
    
    # 使用四元数共轭进行反向旋转（从IMU坐标系到世界坐标系）
    linear_acc_world = quat_rotate([q0, -q1, -q2, -q3], linear_acc)
    
    # 在世界坐标系中减去重力
    gravity_world = np.array([0, 0, 9.81])  # 重力加速度
    linear_acc_no_gravity = linear_acc_world - gravity_world
    
    # 积分计算速度变化
    delta_v = linear_acc_no_gravity * dt
    
    # 低通滤波
    alpha = 0.8
    global_linear_vel = (1 - alpha) * global_linear_vel + alpha * (global_linear_vel + delta_v)
    
    # 零速度校正
    accel_magnitude = np.linalg.norm(linear_acc_no_gravity)
    if accel_magnitude < 0.1:
        decay_factor = np.exp(-dt / 0.5)
        global_linear_vel *= decay_factor
    
    # 速度限幅
    max_velocity = 2.0
    velocity_magnitude = np.linalg.norm(global_linear_vel)
    if velocity_magnitude > max_velocity:
        global_linear_vel = global_linear_vel * (max_velocity / velocity_magnitude)
    
    last_update_time = current_time
    return global_linear_vel.tolist()  # 直接返回世界坐标系下的速度
```

## 6. 性能优化与参数调整

### 6.1 关键参数解释

| 参数 | 描述 | 典型值 | 影响 |
|------|------|--------|------|
| alpha | 低通滤波系数 | 0.8 | 越大越信任当前测量，越小越平滑 |
| accel_threshold | 静止检测阈值 | 0.1 m/s² | 越小越容易被判定为静止 |
| decay_factor | 速度衰减系数 | exp(-dt/0.5) | 影响静止时速度归零速率 |
| max_velocity | 速度上限 | 2.0 m/s | 限制最大估计速度 |

### 6.2 调优建议

1. **增加预热阶段**：
   在开始控制前，让机器人保持静止并校准初始状态

2. **环境适应性调整**：
   - 平坦地面：可使用较大的alpha值
   - 崎岖地形：降低alpha值，增加平滑性
   
3. **针对不同机器人调整**：
   - 大型机器人：增大max_velocity和降低accel_threshold
   - 小型机器人：减小max_velocity和提高accel_threshold

## 7. 高级扩展：卡尔曼滤波器

当前实现使用的是简化的滤波方法。更高级的实现可以考虑卡尔曼滤波器。

### 7.1 卡尔曼滤波原理

卡尔曼滤波器通过预测-更新循环，结合系统模型和测量值，得到最优状态估计。

### 7.2 应用到速度估计

1. **状态向量**：包含位置和速度
2. **系统模型**：描述位置和速度的演化
3. **测量模型**：加速度测量与状态的关系
4. **噪声建模**：对系统噪声和测量噪声进行建模

### 7.3 Python实现示例

```python
def kalman_filter_velocity_estimation():
    # 简化的卡尔曼滤波器实现
    # 需要更详细的状态模型和噪声参数
    pass
```

## 8. 常见问题与解决方案

### 8.1 速度漂移

**症状**：静止时速度不为零，或速度持续增长
**解决**：增强零速度检测，调整decay_factor

### 8.2 过度平滑

**症状**：速度响应滞后，无法反映快速变化
**解决**：增大alpha值，减小滤波强度

### 8.3 噪声干扰

**症状**：速度估计抖动明显
**解决**：减小alpha值，增强滤波效果

## 9. 结论与建议

线速度估算是机器人控制系统的重要组成部分。通过合理的数学模型和滤波技术，可以从IMU数据中获得较为可靠的速度估计。由于积分漂移的客观存在，建议：

1. 定期重置或校准速度估计
2. 结合多种信息源（如有）进行融合
3. 针对具体应用场景调整参数
4. 考虑更高级的滤波方法（如卡尔曼滤波）
5. 实时监测估计质量，发现异常及时干预

通过以上技术和方法，可以有效提高线速度估算的准确性和稳定性，为机器人控制提供可靠的速度反馈。 