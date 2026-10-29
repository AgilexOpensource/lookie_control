"""ROS 2 node: xr_media VR input -> kstone task-space commands."""

import json
import math

import numpy as np
import rclpy
import rclpy.executors
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from .first_person import FirstPersonConfig
from .gesture import GestureConfig
from .hand_retarget import PINCH_FINGERS, HandRetargetConfig
from .policy import (
    SIDES, ArmMappingConfig, ChassisConfig, InputFrame, Joy as JoyInput, TeleopConfig, TeleopPolicy,
    TorsoFollowConfig)
from .third_person import ThirdPersonConfig
from .transforms import make_transform, matrix_to_quat

HAND_JOINTS = ("thumb_metacarpal", "thumb_proximal", "index_proximal",
               "middle_proximal", "ring_proximal", "pinky_proximal")


def _pose_matrix(msg: PoseStamped) -> np.ndarray:
    q, o = msg.pose.position, msg.pose.orientation
    return make_transform([q.x, q.y, q.z], [o.w, o.x, o.y, o.z])


class _Stamped:
    """Latest value plus its receive time (node clock)."""

    def __init__(self) -> None:
        self.value = None
        self.stamp = -math.inf

    def set(self, value, now: float) -> None:
        self.value, self.stamp = value, now

    def fresh(self, now: float, timeout: float):
        return self.value if now - self.stamp <= timeout else None


class LookieTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("lookie_teleop")
        p = self.declare_parameter
        p("xr_prefix", "/xr_media")
        p("kstone_prefix", "/kstone")
        p("base_frame", "chassis_base_link")
        p("rate_hz", 50.0)
        p("pose_timeout_s", 0.3)
        p("joy_timeout_s", 0.5)
        p("ee_pose_timeout_s", 1.0)
        p("start_enabled", False)
        p("enabled_sides", ["left", "right"])
        p("task_ids", ["left_arm", "right_arm"])
        p("hand_joint_prefixes", ["left_", "right_"])
        p("enable_hands", True)
        p("clutch_mode", "hold")
        p("clutch_engage_threshold", 0.6)
        p("clutch_release_threshold", 0.3)
        for name, default in (
                ("enabled", True), ("hold_s", 1.0), ("long_s", 3.0), ("palm_facing_deg", 45.0),
                ("open_curl_max", 1.0), ("still_speed", 0.15), ("still_drift_m", 0.06), ("still_finger_deg", 20.0), ("release_s", 0.3), ("rearm_s", 0.5), ("settle_s", 0.8),
                ("settle_speed", 0.05), ("align_position_m", 0.05), ("align_angle_deg", 15.0),
                ("lost_hold_s", 0.2)):
            p(f"gesture.{name}", default)
        p("long_press_s", 1.0)
        p("use_headset_yaw", True)
        p("headset_forward_axis", [1.0, 0.0, 0.0])
        p("view_mode", "third_person")
        p("first_person.pitch_source", "translation")
        p("first_person.lean_radius_m", 0.6)
        p("third_person.pivot_offset_m", [-0.1, 0.0, -0.45])
        p("arm.position_scale", 1.0)
        p("arm.rotation_enabled", True)
        p("arm.max_offset_m", 1.0)
        p("arm.workspace_min", [-0.2, -1.0, 0.0])
        p("arm.workspace_max", [1.2, 1.0, 1.8])
        p("arm.smoothing", 0.5)
        p("arm.max_linear_speed", 1.5)
        p("arm.max_angular_speed", 6.0)
        p("arm.jump_reject_m", 0.15)
        p("hand.open_positions", [0.0] * 6)
        p("hand.closed_positions", [1.57, 1.03, 1.41, 1.41, 1.41, 1.41])
        hd = HandRetargetConfig()
        for name in ("finger_curl_range", "thumb_curl_range", "thumb_opposition_range",
                     "finger_curl_limits", "thumb_curl_limits", "thumb_opposition_limits",
                     "pinch_range"):
            p("hand." + name, [float(x) for x in getattr(hd, name)])
        p("hand.pinch_fingers", list(hd.pinch_fingers))
        p("hand.pinch_positions", [float(x) for n in hd.pinch_fingers for x in hd.pinch_positions[n]])
        for name in ("adaptive_range", "pinch_enabled", "pinch_release", "pinch_max_curl",
                     "filter_min_cutoff",
                     "filter_beta", "filter_d_cutoff", "palm_jump_ratio", "trigger_deadzone"):
            p("hand." + name, getattr(hd, name))
        p("hand.trigger_ranges", [float(x) for r in hd.trigger_ranges for x in r])
        p("chassis.enabled", True)
        p("chassis.deadzone", 0.15)
        p("chassis.max_linear", 0.4)
        p("chassis.max_angular", 0.8)
        p("chassis.forward_sign", -1.0)
        p("chassis.lateral_sign", -1.0)
        p("chassis.yaw_sign", -1.0)
        p("torso_task_id", "torso")
        d = TorsoFollowConfig()
        for name in ("enabled", "height_scale", "forward_scale", "pitch_scale", "yaw_scale",
                     "height_deadband_m", "forward_deadband_m",
                     "pitch_deadband_rad", "yaw_deadband_rad", "height_limits_m", "forward_limits_m",
                     "pitch_limits_rad", "yaw_limits_rad", "smoothing", "max_linear_speed",
                     "max_angular_speed", "jump_reject_m", "stick_side", "stick_rate_mps",
                     "stick_sign", "stick_deadzone"):
            default = getattr(d, name)
            p(f"torso.{name}", list(default) if isinstance(default, tuple) else default)
        v = lambda name: self.get_parameter(name).value  # noqa: E731
        self._v = v

        sides = tuple(s for s in v("enabled_sides") if s in SIDES)
        task_ids = list(v("task_ids"))
        prefixes = list(v("hand_joint_prefixes"))
        if len(task_ids) != 2 or len(prefixes) != 2:
            raise ValueError("task_ids and hand_joint_prefixes need one entry per side (left, right)")
        self._task_of = dict(zip(SIDES, task_ids))
        self._hand_names = {
            s: [f"{prefix}{j}_joint" for j in HAND_JOINTS] for s, prefix in zip(SIDES, prefixes)}

        config = TeleopConfig(
            arm=ArmMappingConfig(
                position_scale=float(v("arm.position_scale")),
                rotation_enabled=bool(v("arm.rotation_enabled")),
                max_offset_m=float(v("arm.max_offset_m")),
                workspace_min=tuple(float(x) for x in v("arm.workspace_min")),
                workspace_max=tuple(float(x) for x in v("arm.workspace_max")),
                smoothing=float(v("arm.smoothing")),
                max_linear_speed=float(v("arm.max_linear_speed")),
                max_angular_speed=float(v("arm.max_angular_speed")),
                jump_reject_m=float(v("arm.jump_reject_m")),
            ),
            hand=self._hand_config(v),
            chassis=ChassisConfig(
                enabled=bool(v("chassis.enabled")),
                deadzone=float(v("chassis.deadzone")),
                max_linear=float(v("chassis.max_linear")),
                max_angular=float(v("chassis.max_angular")),
                forward_sign=float(v("chassis.forward_sign")),
                lateral_sign=float(v("chassis.lateral_sign")),
                yaw_sign=float(v("chassis.yaw_sign")),
            ),
            torso=TorsoFollowConfig(**{
                name: (tuple(float(x) for x in v(f"torso.{name}")) if isinstance(default, tuple)
                       else type(default)(v(f"torso.{name}")))
                for name, default in vars(d).items()}),
            view_mode=str(v("view_mode")),
            first_person=FirstPersonConfig(
                pitch_source=str(v("first_person.pitch_source")),
                lean_radius_m=float(v("first_person.lean_radius_m")),
            ),
            third_person=ThirdPersonConfig(
                pivot_offset_m=tuple(float(x) for x in v("third_person.pivot_offset_m")),
            ),
            enabled_sides=sides,
            clutch_mode=str(v("clutch_mode")),
            clutch_engage_threshold=float(v("clutch_engage_threshold")),
            clutch_release_threshold=float(v("clutch_release_threshold")),
            gesture=GestureConfig(
                enabled=bool(v("gesture.enabled")),
                hold_s=float(v("gesture.hold_s")),
                long_s=float(v("gesture.long_s")),
                palm_facing_cos=math.cos(math.radians(float(v("gesture.palm_facing_deg")))),
                open_curl_max=float(v("gesture.open_curl_max")),
                still_speed=float(v("gesture.still_speed")),
                still_drift_m=float(v("gesture.still_drift_m")),
                still_finger_rad=math.radians(float(v("gesture.still_finger_deg"))),
                release_s=float(v("gesture.release_s")),
                rearm_s=float(v("gesture.rearm_s")),
                settle_s=float(v("gesture.settle_s")),
                settle_speed=float(v("gesture.settle_speed")),
                align_position_m=float(v("gesture.align_position_m")),
                align_angle_rad=math.radians(float(v("gesture.align_angle_deg"))),
                lost_hold_s=float(v("gesture.lost_hold_s")),
            ),
            long_press_s=float(v("long_press_s")),
            start_enabled=bool(v("start_enabled")),
            headset_forward_axis=tuple(float(x) for x in v("headset_forward_axis")),
            use_headset_yaw=bool(v("use_headset_yaw")),
            enable_hands=bool(v("enable_hands")),
        )
        if len(config.hand.open_positions) != 6 or len(config.hand.closed_positions) != 6:
            raise ValueError("hand.open_positions / hand.closed_positions need 6 values")
        self._policy = TeleopPolicy(config)

        xr = v("xr_prefix").rstrip("/")
        ks = v("kstone_prefix").rstrip("/")
        self._headset = _Stamped()
        self._controllers = {s: _Stamped() for s in SIDES}
        self._joys = {s: _Stamped() for s in SIDES}
        self._hands = {s: _Stamped() for s in SIDES}
        self._ee = {s: _Stamped() for s in SIDES}
        self._torso = _Stamped()
        self._torso_home = None
        self._ee_home = {s: None for s in SIDES}
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        torso_task = f"{ks}/tasks/{v('torso_task_id')}"
        self.create_subscription(
            PoseStamped, torso_task + "/pose", lambda m: self._torso.set(_pose_matrix(m), self._now()), 10)
        self.create_subscription(
            PoseStamped, torso_task + "/home_pose", self._on_torso_home, latched)
        self._torso_pub = self.create_publisher(PoseStamped, torso_task + "/target", 10)

        self.create_subscription(
            TransformStamped, f"{xr}/headset/transform", lambda m: self._on_tf(self._headset, m), 10)
        for side in SIDES:
            self.create_subscription(
                TransformStamped, f"{xr}/{side}_controller/transform",
                lambda m, s=side: self._on_tf(self._controllers[s], m), 10)
            self.create_subscription(
                Joy, f"{xr}/{side}_controller/buttons", lambda m, s=side: self._on_joy(s, m), 10)
            self.create_subscription(
                PoseArray, f"{xr}/{side}_controller/hand",
                lambda m, s=side: self._on_hand(s, m), qos_profile_sensor_data)
            self.create_subscription(
                PoseStamped, f"{ks}/tasks/{self._task_of[side]}/pose",
                lambda m, s=side: self._on_ee(s, m), 10)
            self.create_subscription(
                PoseStamped, f"{ks}/tasks/{self._task_of[side]}/home_pose",
                lambda m, s=side: self._ee_home.__setitem__(s, _pose_matrix(m)), latched)

        self._target_pubs = {
            s: self.create_publisher(PoseStamped, f"{ks}/tasks/{self._task_of[s]}/target", 10)
            for s in sides}
        self._hand_pub = self.create_publisher(JointState, f"{ks}/extra_joint_commands", 10)
        self._cmd_vel_pub = self.create_publisher(Twist, f"{ks}/cmd_vel", 10)
        self._state_pub = self.create_publisher(
            String, "~/state", QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self._home_client = self.create_client(Trigger, f"{ks}/reset_home")
        self.create_service(SetBool, "~/enable", self._on_enable)

        self._moving = False
        self._last_state = None
        self._gesture_stage = None
        self.create_timer(1.0 / float(v("rate_hz")), self._tick)
        self.get_logger().info(
            f"lookie_teleop ready: xr={xr}, kstone={ks}, sides={list(sides)}, "
            f"view_mode={config.view_mode}, enabled={self._policy.enabled}")

    @staticmethod
    def _hand_config(v) -> HandRetargetConfig:
        def pair(name):
            x = [float(e) for e in v("hand." + name)]
            if len(x) != 2:
                raise ValueError(f"hand.{name} needs 2 values")
            return tuple(x)
        ranges = [float(x) for x in v("hand.trigger_ranges")]
        if len(ranges) != 12:
            raise ValueError("hand.trigger_ranges needs 12 values")
        fingers = tuple(str(f) for f in v("hand.pinch_fingers"))
        unknown = [f for f in fingers if f not in PINCH_FINGERS]
        if unknown or len(set(fingers)) != len(fingers):
            raise ValueError(f"hand.pinch_fingers must be distinct names from {PINCH_FINGERS}")
        pinch = [float(x) for x in v("hand.pinch_positions")]
        if len(pinch) != 3 * len(fingers):
            raise ValueError("hand.pinch_positions needs 3 values per hand.pinch_fingers entry")
        return HandRetargetConfig(
            open_positions=[float(x) for x in v("hand.open_positions")],
            closed_positions=[float(x) for x in v("hand.closed_positions")],
            finger_curl_range=pair("finger_curl_range"),
            thumb_curl_range=pair("thumb_curl_range"),
            thumb_opposition_range=pair("thumb_opposition_range"),
            adaptive_range=bool(v("hand.adaptive_range")),
            finger_curl_limits=pair("finger_curl_limits"),
            thumb_curl_limits=pair("thumb_curl_limits"),
            thumb_opposition_limits=pair("thumb_opposition_limits"),
            pinch_enabled=bool(v("hand.pinch_enabled")),
            pinch_range=pair("pinch_range"),
            pinch_fingers=fingers,
            pinch_positions={f: tuple(pinch[3 * i:3 * i + 3]) for i, f in enumerate(fingers)},
            pinch_release=float(v("hand.pinch_release")),
            pinch_max_curl=float(v("hand.pinch_max_curl")),
            filter_min_cutoff=float(v("hand.filter_min_cutoff")),
            filter_beta=float(v("hand.filter_beta")),
            filter_d_cutoff=float(v("hand.filter_d_cutoff")),
            palm_jump_ratio=float(v("hand.palm_jump_ratio")),
            trigger_deadzone=float(v("hand.trigger_deadzone")),
            trigger_ranges=[tuple(ranges[i:i + 2]) for i in range(0, 12, 2)],
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_tf(self, store: _Stamped, msg: TransformStamped) -> None:
        t, r = msg.transform.translation, msg.transform.rotation
        store.set(make_transform([t.x, t.y, t.z], [r.w, r.x, r.y, r.z]), self._now())

    def _on_joy(self, side: str, msg: Joy) -> None:
        self._joys[side].set(JoyInput(list(msg.axes), list(msg.buttons)), self._now())

    def _on_hand(self, side: str, msg: PoseArray) -> None:
        if len(msg.poses) != 25:
            return
        points = np.array([[q.position.x, q.position.y, q.position.z] for q in msg.poses])
        w = msg.poses[0]
        wrist = make_transform(
            points[0], [w.orientation.w, w.orientation.x, w.orientation.y, w.orientation.z])
        self._hands[side].set((points, wrist), self._now())

    def _on_ee(self, side: str, msg: PoseStamped) -> None:
        self._ee[side].set(_pose_matrix(msg), self._now())

    def _on_torso_home(self, msg: PoseStamped) -> None:
        self._torso_home = _pose_matrix(msg)

    def _on_enable(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self._policy.set_enabled(request.data)
        response.success = True
        response.message = "enabled" if request.data else "disabled"
        self.get_logger().info(f"teleop {response.message} via service")
        return response

    def _frame(self, now: float) -> InputFrame:
        pose_to = float(self._v("pose_timeout_s"))
        joy_to = float(self._v("joy_timeout_s"))
        ee_to = float(self._v("ee_pose_timeout_s"))
        hands = {s: self._hands[s].fresh(now, pose_to) for s in SIDES}
        return InputFrame(
            now=now,
            headset=self._headset.fresh(now, pose_to),
            # xr_media also publishes a grip transform for tracked hands; only a side
            # without hand joints is a real controller.
            controllers={s: (self._controllers[s].fresh(now, pose_to) if hands[s] is None else None)
                         for s in SIDES},
            joys={s: self._joys[s].fresh(now, joy_to) for s in SIDES},
            hands={s: (h[0] if h is not None else None) for s, h in hands.items()},
            hand_wrists={s: (h[1] if h is not None else None) for s, h in hands.items()},
            ee_poses={s: self._ee[s].fresh(now, ee_to) for s in SIDES},
            ee_homes=dict(self._ee_home),
            torso_pose=self._torso.fresh(now, ee_to),
            torso_home=self._torso_home,
        )

    def _tick(self) -> None:
        now = self._now()
        out = self._policy.step(self._frame(now))
        stamp = self.get_clock().now().to_msg()

        stage = out.state.get("gesture_stage")
        if stage != self._gesture_stage:
            hint = {"hold": "gesture: holding", "ready": "gesture: release now to trigger",
                    None: "gesture: released"}[stage]
            self.get_logger().info(hint)
            self._gesture_stage = stage

        for event in out.events:
            self.get_logger().info(f"event: {event}")
            if event == "calibrate":
                self._call(self._home_client, event)

        for side, target in out.arm_targets.items():
            self._target_pubs[side].publish(self._pose_msg(target, stamp))
        if out.torso_target is not None:
            self._torso_pub.publish(self._pose_msg(out.torso_target, stamp))

        if out.hand_positions:
            msg = JointState()
            msg.header.stamp = stamp
            for side, positions in out.hand_positions.items():
                msg.name.extend(self._hand_names[side])
                msg.position.extend(positions)
            self._hand_pub.publish(msg)

        moving = any(abs(x) > 1e-6 for x in out.cmd_vel)
        if moving or self._moving:
            # Send while moving and one zero command on stop.
            twist = Twist()
            twist.linear.x, twist.linear.y, twist.angular.z = out.cmd_vel
            self._cmd_vel_pub.publish(twist)
        self._moving = moving

        state = json.dumps(out.state, sort_keys=True)
        if state != self._last_state or out.events:
            self._state_pub.publish(String(data=state))
            self._last_state = state

    def _pose_msg(self, target: np.ndarray, stamp) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp, msg.header.frame_id = stamp, self._v("base_frame")
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(x) for x in target[:3, 3])
        o = msg.pose.orientation
        o.w, o.x, o.y, o.z = (float(x) for x in matrix_to_quat(target[:3, :3]))
        return msg

    def _call(self, client, label: str) -> None:
        if not client.service_is_ready():
            self.get_logger().warning(f"{label}: service {client.srv_name} not available")
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda f: self._on_done(label, f))

    def _on_done(self, label: str, future) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"{label} failed: {exc}")
            return
        log = self.get_logger().info if result.success else self.get_logger().warning
        log(f"{label}: {result.message}")

    def stop_base(self) -> None:
        self._cmd_vel_pub.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LookieTeleopNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop_base()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
