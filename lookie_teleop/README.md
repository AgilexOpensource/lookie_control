# lookie_teleop

Lookie 的 VR 遥操逻辑包。输入来自 `xr_media_ros`，输出是发给 kstone 运控层的任务目标。launch 只启动遥操节点，VR 节点和运控节点（或 [kstone_mock](../kstone_mock/README.md)）需要另外启动。

当前只支持模型 `lookie__pf-lookie_trio_r01__ee-revo2std_r01`。

## 构建

在工作区根目录执行，venv 的创建见 [kstone_mock](../kstone_mock/README.md)：

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python3 -m colcon build --symlink-install --packages-up-to kstone_mock lookie_teleop
source install/setup.bash
```

## 运行

每个终端都要先 source 上面三个环境（ROS、`.venv`、`install`）。

```bash
# 终端 1：仿真运控（替代 kstone_ros）
ros2 launch kstone_mock kstone_mock.launch.py
# 终端 2：VR 节点
ros2 launch xr_media_ros ...
# 终端 3：遥操
ros2 launch lookie_teleop lookie_teleop.launch.py            # 默认未使能
ros2 launch lookie_teleop lookie_teleop.launch.py start_enabled:=true
```

launch 参数：`config`（yaml 路径）、`start_enabled`、`view_mode`（`third_person` 默认 / `first_person`）。

## 接口

输入（来自 xr_media）：

| 话题 | 类型 |
|---|---|
| `/xr_media/headset/transform` | TransformStamped |
| `/xr_media/{left,right}_controller/transform` | TransformStamped |
| `/xr_media/{left,right}_controller/buttons` | Joy，axes 为 `[stick_x, stick_y, trigger, grip]`，buttons 为 `[trigger, grip, stick, X/A, Y/B]` |
| `/xr_media/{left,right}_controller/hand` | PoseArray，25 个手部关节 |

输出（发往 kstone，下表为 mock 定义的约定）：

| 接口 | 类型 | 说明 |
|---|---|---|
| `/kstone/tasks/{left_arm,right_arm}/target` | PoseStamped | 末端目标，frame 为 `chassis_base_link` |
| `/kstone/tasks/<task>/pose` | PoseStamped（订阅） | 当前末端位姿，接合时用作锚点 |
| `/kstone/tasks/torso/target` | PoseStamped | 躯干目标。未命令过时不发布，之后每周期发布 |
| `/kstone/tasks/torso/pose`, `home_pose` | PoseStamped（订阅） | 躯干当前和零位位姿，`home_pose` 为 latched |
| `/kstone/extra_joint_commands` | JointState | Revo2 手指关节 |
| `/kstone/cmd_vel` | Twist | 底盘速度。只在有输入时发送，松开摇杆后补发一次零速度 |
| `/kstone/tasks/<task>/reset_target`, `/kstone/reset_home` | Trigger（客户端） | 回零 |
| `~/enable` | SetBool（服务） | 使能或失能 |
| `~/state` | String (JSON) | 状态，变化时发布 |

## 操作

- 右手 B 长按：校准。停止遥操，腰和手臂都回 home（调用 `reset_home`），并把此刻的手柄（或手腕）位姿、头显位姿和朝向记为参考，对应各自的 home 位姿。所以长按 B 时要摆好与 home 对应的姿态。
- 右手 A 短按：开始遥操。必须先做过一次 B 校准，否则只发出 `calibration_required` 事件；遥操中按 A 无作用，停止只能用 B。`~/enable` 服务不受校准限制。
- 握 grip：接合该侧手臂，按 B 校准时的参考跟随。grip 不重新标定：松开后手臂保持，再握住时目标从当前实测位姿出发、经限速回到校准映射，不能靠松开-移动-再握来“挪手”。也可以改为 `clutch_mode: double_press`。
- 扳机：控制 Revo2 握合。拇指先对掌，再与四指一起合拢（`hand.trigger_ranges`），避免拇指压在四指上。无手柄时如果有手势追踪，手指按 25 点手部关键点重定向（见下节）。无手柄的一侧用手势控制，见“手势遥操”一节。
- 左摇杆：底盘平移。右摇杆 x：底盘旋转。

手臂映射以校准时的头显朝向（yaw）为参考系，$C_0$、$E_0$ 为校准时的手柄位姿和 EE home（若该侧当时没有追踪或 home 位姿，则退化为首次接合时锁存）：

$$p = p_{E0} + s\,R_{yaw}^\top (p_C - p_{C0}),\quad R = (R_{yaw}^\top R_C R_{C0}^\top R_{yaw})\,R_{E0}$$

此外还有最大偏移、工作空间裁剪和低通平滑（`arm.*`）。目标速度上限 `arm.max_linear_speed`（1.5 m/s）和 `arm.max_angular_speed`（6.0 rad/s）限制甩手时的目标速度。`arm.jump_reject_m`（0.15 m）把单次更新中手柄位置的跳变当作追踪丢失，重新锚定而不跟随。

## 手势遥操

没有手柄、只有手部追踪的一侧按以下规则工作（`gesture.*`）：

- 手势：双手张开、掌心朝向头显并保持静止。手腕速度超过 `still_speed`（0.15 m/s），或离开手势起点超过 `still_drift_m`（0.06 m），或任一手指弯曲相对起点变化超过 `still_finger_deg`（20°），计时立即清零，所以手或手指在运动时掌心朝头显都不算。第三人称时头显挂在胸前，视野偏低，手势要在胸前低处做。
- 短持（默认 1–3 s，松开时触发）：未校准时校准；已校准且停止时开始遥操；遥操中短持无效，不会暂停。
  - 终端提示：`gesture: holding` 表示开始计时，`gesture: release now to trigger` 表示可以松开，`gesture: released` 表示已松开。短于 `release_s`（0.3 s）的追踪闪断不算松开。
- 长持（≥ `long_s`，按住时触发）：重新校准。遥操中手势只有长持有效，终端在保持满 `hold_s` 后才打印 `gesture: holding`。
- 校准时机器人立即回 home。参考位姿不取手势那一刻，而是在松开手势、双手腕静止 `settle_s` 后记录，所以回 home 后要把双手摆到与 home 对应的姿态并停住。
- 接合：遥操中，某侧映射后的手腕目标与实测 EE 相差在 `align_position_m`、`align_angle_deg` 内才接合，避免一接合就跳。遥操中做手势不释放手臂，手臂照常跟随；停止时双臂释放。
- 丢失：手腕丢失不超过 `lost_hold_s` 时保持目标；超过后释放该臂，重新对齐后接合，不重新标定。
- 手指只在该侧手臂接合时发送。

阈值需要在 PICO 上实测调整。

## 手势重定向

每只手 6 个主动关节 `[拇指侧摆, 拇指弯曲, 食, 中, 无名, 小]`。远端关节是 mimic，由运控补。

- 度量与尺度、姿态无关：
  - 四指：MCP+PIP+DIP 弯角和。
  - 拇指弯曲：两节弯角和。
  - 拇指侧摆：拇指尖到小指掌骨的距离除以掌长。
- 每个度量按 `*_range` 线性映射到关节范围。默认范围有意取窄，大多数人都能张满、握满。`adaptive_range` 会把范围扩展到操作者实际达到的极值，但不超过 `*_limits`。右手 B 校准时重置。
- 捏合：对 `pinch_fingers`（默认食指、中指）中的每根手指，拇指尖到该指尖距离/掌长落入 `pinch_range`（0.45 → 0.15）时，拇指两关节和该手指平滑过渡到 `pinch_positions` 中对应的一组 `[拇指侧摆, 拇指弯曲, 手指]`。这些位姿用 Revo2 URDF 正运动学搜出，两指尖距离小于 1 mm。Revo2 拇指够不到无名指（差 9 mm）和小指（差 24 mm），默认不启用。多指同时捏合时，拇指取各目标按权重的平均。
  - 迟滞：捏合完成（距离 ≤ 0.15）后，直到距离超过 `pinch_release`（0.6）才松开，避免在阈值附近抖动。过渡权重在进入和松开时都连续。
  - 手指弯曲超过 `pinch_max_curl` 时该指的捏合淡出，握拳不受影响。
- 度量经 One-Euro 滤波（`filter_*`）：慢动作去抖，快动作低延迟。掌长单帧变化超过 `palm_jump_ratio` 视为追踪跳变，保持上一帧输出；连续 5 帧以上则接受新值。

## 视角模式

`view_mode` 选择头显佩戴方式，两种模式的映射分别实现在 [third_person.py](lookie_teleop/third_person.py) 和 [first_person.py](lookie_teleop/first_person.py)，默认 `third_person`。

**第三人称（头显挂胸前，默认）**：操作者直接看机器人，头显当作躯干追踪器。

- 安装姿态：B 校准时测量 $R_m = R_z(\text{yaw})^\top R_H$，之后躯干姿态 $B = (p_H,\ R_H R_m^\top)$。校准时请站直、面向机器人前方。
- 躯干：髋部支点 $p_H + R_B\,\text{pivot\_offset\_m}$ 的位移给出高度和前后量；$R_{B_0}^\top R_B$ 给出俯仰和偏航。转身即转腰，弯腰即俯仰。
- 手臂：在身体坐标系下映射。校准时记录 $Q_0 = B_0^{-1} C_0$（手柄相对胸口）和 $A_0 = F_0^{-1} E_0$（末端相对机器人躯干），运行时 $E = F\,A$，其中 $A$ 的位置为 $A_0 + s\,(Q - Q_0)$，姿态为 $R_Q R_{Q_0}^\top R_{A_0}$。转身时手臂目标随机器人躯干一起转。头显丢失时手臂保持。
- 注意：手柄依赖头显摄像头追踪，挂胸前时手举过头或背到身后会丢失；俯仰来自躯干本身，不受低头影响。

**第一人称（头显戴在头上）**：手臂在世界坐标（校准时的操作者朝向）下映射；躯干跟随头部，偏航相对校准时的头部朝向。

## 躯干跟随头显

任意一侧手臂接合时，躯干跟随头显相对 B 校准时的变化（在校准时的操作者朝向下表示）。状态为相对躯干零位的 4 个偏移 `[前后 x, 高度 z, 俯仰, 偏航]`：

- 高度：`height_scale × 头显 Δz`，蹲下即降腰。
- 俯仰：第三人称取胸口姿态的俯仰。第一人称默认 `first_person.pitch_source: translation`，$\text{pitch} = \operatorname{atan2}(\Delta x, \text{lean\_radius\_m})$，即用头显前移估计弯腰，不受点头影响；`head` 则用头显前向轴的俯仰变化，点头也会弯腰。
- 偏航：`yaw_scale × 头显 Δyaw`。
- 前后：`forward_scale × 头显 Δx`，默认 0，因为前倾已由俯仰表达。

每轴先过软死区，再乘比例、限幅（`*_limits`），然后低通平滑和限速（`max_linear_speed` 0.3 m/s，`max_angular_speed` 1.0 rad/s）。头显单次跳变超过 `jump_reject_m` 时只平移参考以吸收跳变。松开 grip 后躯干保持在最后的目标；重新接合时从当前目标出发、经限速回到校准映射，不会突跳。B 长按校准时躯干回零并重设参考。

右摇杆 y（`stick_side`）随时叠加高度偏移，前推升高，速度 `stick_rate_mps`，偏移会被限制在高度范围内，不会积累。

可以尝试的预设（参数在 `torso.*` 下）：

| 预设 | 设置 |
|---|---|
| 全 1:1（默认） | 各 scale 为 1，`forward_scale: 0` |
| 减半 | 各 scale 为 0.5 |
| 只跟高度 + 摇杆 | `pitch_scale: 0`，`yaw_scale: 0` |
| 点头弯腰（第一人称） | `first_person.pitch_source: head` |
| 关闭跟随，只用摇杆 | 各 scale 为 0 |

注意：躯干误差会在末端被放大（腰部俯仰 0.1 rad 在 0.6 m 外约 6 cm），第一人称下手臂锚点基于 base 坐标，手臂目标不随躯干移动，由逆解吸收；第三人称下手臂目标挂在躯干上（滞后一个控制周期）。尚未实现按眼点（`head_base_link`）映射。

## 待确认的假设

- 真实 kstone_ros 的接口还没有定，这里先按 kstone_mock 的约定实现。等运控层完成后，只需要改话题前缀、task id 或节点里的消息转换。
- 头显前向轴默认是 `+X`（`headset_forward_axis`），摇杆符号（`chassis.*_sign`）也需要在实机上核对。
- 手势重定向的初始范围（`hand.*_range`）和滤波参数是估计值，需要用 PICO 实测后调整。Revo2 的握合方向假定为正。
- 节点不开放任何网络服务，只使用 ROS 话题。

## 测试

```bash
cd src/lookie_control/lookie_teleop && python3 -m pytest test -q
```
