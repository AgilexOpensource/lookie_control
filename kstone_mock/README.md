# kstone_mock

kstone_ros 运控层完成之前的替身。它用 MuJoCo 加载 `lookie_description` 的 URDF，用 [mink](https://github.com/kevinzakka/mink) 做全身 QP 逆解（躯干 + 双臂），并按 kstone 的话题约定对外提供接口，供 [lookie_teleop](../lookie_teleop/README.md) 联调。

## 环境与构建

在工作区根目录执行：

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv   # 首次
source .venv/bin/activate
python3 -m pip install 'mujoco==3.3.7' 'mink==1.1.0' 'qpsolvers==4.12.0' 'daqp==0.10.3' 'numpy<2'
python3 -m colcon build --symlink-install --packages-up-to kstone_mock
source install/setup.bash
```

`python3 -m colcon` 让节点绑定 venv 中的 Python；直接用 `colcon` 会绑定系统 Python，无法导入 MuJoCo。

## 运行

```bash
ros2 launch kstone_mock kstone_mock.launch.py              # 带 MuJoCo 可视化窗口
ros2 launch kstone_mock kstone_mock.launch.py viewer:=false
```

`forward_to_driver:=true` 会把关节状态转发到 `/trio/control/joint_states`，用于驱动真实机器人。默认关闭，只有确认机器人安全时才打开。

## 接口约定

| 接口 | 类型 | 说明 |
|---|---|---|
| `/kstone/tasks/<id>/target` | PoseStamped（订阅） | 末端目标。frame 为 `chassis_base_link` 或 `odom` |
| `/kstone/tasks/<id>/pose`, `home_pose` | PoseStamped | 当前末端位姿和零位位姿 |
| `/kstone/extra_joint_commands` | JointState（订阅） | 不参与逆解的关节，例如 Revo2 和头部。mimic 关节自动跟随 |
| `/kstone/cmd_vel` | Twist（订阅） | 底盘速度，0.5 s 无输入自动停止 |
| `/kstone/joint_states`, `/kstone/base_pose` | 发布 | 关节状态和底盘位姿 |
| `/kstone/tasks/<id>/reset_target`, `/kstone/reset_home` | Trigger | 回零 |
| `/kstone/tasks/torso/{target,pose,home_pose}` | PoseStamped | 躯干 `torso_link4` 的目标、当前位姿和零位位姿，frame 只接受 `chassis_base_link`。`reset_target` 和 `reset_home` 都会把躯干目标恢复到零位 |

task id 默认为 `left_arm` 和 `right_arm`，对应 `left_base_link` 和 `right_base_link`。逆解权重、关节列表等参数见 [config](config/)。

## 逆解

每个控制周期分 `ik_iterations` 步调用 `mink.solve_ik`（daqp 求解器）：

- 代价：两个手的 `FrameTask`，`PostureTask` 拉向零位，`DampingTask` 抑制关节运动。`slow_joints`（躯干）阻尼更大，所以手臂优先动。
- 硬约束：关节限位（`limit_margin`、`limit_gain`）、关节速度上限，非逆解关节（底盘、头、手指）冻结不动。手臂速度上限为 `max_joint_velocity`（默认 3.0 rad/s），躯干单独用 `slow_joint_max_velocity`（默认 1.0 rad/s）。
- 躯干加速度上限 `slow_joint_max_acceleration`（默认 3.0 rad/s²，0 为关闭）：手部突然甩动时手臂先饱和，躯干每个周期只能多加一点速度，等它加速起来手已经回来了，所以躯干几乎不动。转身这类持续需求会让躯干在约 v/a = 0.33 s 内加到满速，然后跟上。代价是手部高频抖动（例如 4 Hz、±4 cm）只能靠手臂吸收，跟踪误差约 1.4 cm。
- 躯干位姿：`torso_body`（默认 `torso_link4`）相对 `base_frame` 趋向 `torso/target`（未收到时为零位位姿），按该 link 自身轴分别加权：`torso_position_weight` 为 x/y/z，默认 `[1.0, 0, 3.0]`；`torso_orientation_weight` 为滚转/俯仰/偏航，默认 `[3.0, 3.0, 0.3]`。x 和偏航权重要足够大，遥操下发的前后和转身目标才能被跟上；代价是目标停在零位时手臂借腰够取变少（手前伸 25 cm 时残差约 1.6 cm）。只想锁零位不跟目标时可用 `[0, 0, 3]` / `[3, 3, 0.01]`。全 0 关闭。
- `torso_gain`（默认 0.3）是躯干任务的增益，每周期只要求消除误差的一部分。增益为 1 时 QP 要求速度 = 误差/dt，被躯干加速度上限截断后会超调和来回振荡；0.3 配合遥操端限速后的目标可以稳定收敛。
- 自碰撞：`collision_pairs` 中每项为 `"A 组 link | B 组 link"`，保持两组之间距离不小于 `collision_min_distance`。默认配置了双臂之间和手臂与躯干之间。设为 `[""]` 可关闭。碰撞距离按网格凸包计算。

## 测试

```bash
cd src/lookie_control/kstone_mock && python3 -m pytest test -q
```
