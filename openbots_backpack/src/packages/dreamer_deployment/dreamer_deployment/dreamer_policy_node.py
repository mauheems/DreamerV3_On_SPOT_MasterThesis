#!/usr/bin/env python3
"""
Dreamer Policy Inference Node for SPOT Robot

Subscribes to deployment topics:
  /camera/frontmiddle_virtual/image  → image observation
  /odometry                          → position, orientation, velocity
  /spot/local_grid/terrain           → terrain height grid

Publishes:
  /cmd_vel  (geometry_msgs/Twist)    → velocity commands decoded from policy action

Calls service:
  /locomotion_mode                   → gait selection decoded from policy action[3]

Subscribes to:
  /spot/policy/goal  (geometry_msgs/PointStamped) → set goal waypoint (x, y in world frame)

The policy is a trained DreamerV3 JAX agent. Its RSSM maintains internal recurrent
state between steps — no manual history buffer is needed.

Action space (4D in [-1, 1]):
  action[0] * max_vel  → cmd_vel.linear.x
  action[1] * max_vel  → cmd_vel.linear.y
  action[2] * 1.0      → cmd_vel.angular.z
  action[3] > 0        → locomotion HINT_TROT (max_vel=2.0)
  action[3] <= 0       → locomotion HINT_AUTO (max_vel=1.0)

To switch policies, only change `checkpoint_path` in params.yaml.
"""

import os
import sys
import importlib
from pathlib import Path

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

from spot_msgs.srv import SetLocomotion  # type: ignore

# Locomotion hint values (Boston Dynamics)
_HINT_CRAWL = 4   # action[3] <= 0 → slower crawl gait
_HINT_TROT  = 2   # action[3]  > 0 → faster trot gait


class DreamerPolicyNode(Node):
    """
    DreamerV3 policy inference node for SPOT navigation.

    Observation / action spaces mirror the training environment (spot.py)
    exactly so that the loaded checkpoint runs without modification.
    """

    def __init__(self):
        super().__init__('dreamer_policy_node')

        # ============ PARAMETERS ============
        # Only change checkpoint_path to swap between training sessions.
        self.declare_parameter(
            'checkpoint_path',
            '/home/maurits-heemskerk/Documents/Uni/Master_Thesis/'
            'dreamer_results_local/dreamer-results-20260310-160927-informed-fulldata/checkpoint.ckpt'
        )
        self.declare_parameter('control_rate_hz', 10)   # Hz; keep ≤ 10 for JAX inference
        self.declare_parameter('robot_name', 'spot')
        self.declare_parameter('dreamerv3_root', '')

        # Image dimensions must match training data exactly.
        self.declare_parameter('image_height', 64)
        self.declare_parameter('image_width',  128)

        # Terrain grid dimensions must match training data exactly.
        self.declare_parameter('terrain_height', 40)
        self.declare_parameter('terrain_width',  40)
        
        # TEMPORARY DEBUG: set to true to flip goal_rel sign to test if inverse model
        self.declare_parameter('invert_goal_sign', False)
        
        # TEMPORARY DEBUG: set to true to flip action[0] (forward/backward) sign
        self.declare_parameter('invert_action_x_sign', False)

        checkpoint_path  = self.get_parameter('checkpoint_path').value
        control_rate_hz  = self.get_parameter('control_rate_hz').value
        robot_name       = self.get_parameter('robot_name').value
        self._dreamerv3_root = self.get_parameter('dreamerv3_root').value
        self._img_h      = self.get_parameter('image_height').value
        self._img_w      = self.get_parameter('image_width').value
        self._ter_h      = self.get_parameter('terrain_height').value
        self._ter_w      = self.get_parameter('terrain_width').value
        self._invert_goal_sign = self.get_parameter('invert_goal_sign').value
        self._invert_action_x_sign = self.get_parameter('invert_action_x_sign').value

        self.get_logger().info('=== Dreamer Policy Node ===')
        self.get_logger().info(f'  Checkpoint : {checkpoint_path}')
        self.get_logger().info(f'  Control Hz : {control_rate_hz}')
        self.get_logger().info(f'  Image size : {self._img_h}x{self._img_w}')
        self.get_logger().info(f'  Terrain sz : {self._ter_h}x{self._ter_w}')
        if self._dreamerv3_root:
            self.get_logger().info(f'  dreamerv3 root override : {self._dreamerv3_root}')

        # ============ SENSOR STATE ============
        self._bridge             = CvBridge()
        self._latest_image       = None   # np.uint8 (H, W, 3)
        self._latest_odometry    = None   # nav_msgs/Odometry
        self._latest_terrain     = np.zeros((self._ter_h, self._ter_w), dtype=np.float32)
        self._robot_position     = np.zeros(3, dtype=np.float32)  # x, y, z

        # Goal in world frame; set via /spot/policy/set_goal service
        self._goal_world         = None   # np.float32 (2,) — [x, y]
        self._is_first_step      = True   # tells RSSM this is the start of an episode
        self._step_counter       = 0      # for throttled logging

        # ============ DREAMER AGENT ============
        self._agent    = None
        self._rssm_state = None           # recurrent state maintained between steps
        self._load_agent(checkpoint_path)

        # ============ SUBSCRIBERS ============
        # Mirror the exact topics from episode_bag_recorder.py
        self.create_subscription(
            Image,
            '/camera/frontmiddle_virtual/image',
            self._image_callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            '/odometry',
            self._odometry_callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self.create_subscription(
            OccupancyGrid,
            '/spot/local_grid/terrain',
            self._terrain_callback,
            rclpy.qos.qos_profile_sensor_data,
        )

        # ============ PUBLISHERS ============
        # Primary output: velocity commands consumed by spot_driver
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        # Debug: raw 4D action vector for monitoring / logging
        self._action_debug_pub = self.create_publisher(
            Float32MultiArray, '/spot/policy_action_debug', 1
        )

        # ============ SERVICE CLIENTS ============
        self._locomotion_client = self.create_client(SetLocomotion, '/locomotion_mode')

        # ============ GOAL TOPIC ============
        # Publish a goal with: ros2 topic pub --once /spot/policy/goal \
        #   geometry_msgs/msg/PointStamped "{header: {frame_id: 'odom'}, point: {x: 2.0, y: 0.0, z: 0.0}}"
        self.create_subscription(
            PointStamped,
            '/spot/policy/goal',
            self._goal_callback,
            10,
        )
        self.get_logger().info('Goal topic ready at /spot/policy/goal')

        # ============ CONTROL LOOP ============
        self._control_timer = self.create_timer(
            1.0 / control_rate_hz,
            self._control_loop,
        )
        self.get_logger().info('Policy node initialised — waiting for sensor data …')

    # ──────────────────────────────────────────────────────────────────────────
    # AGENT LOADING
    # ──────────────────────────────────────────────────────────────────────────

    def _load_agent(self, checkpoint_path: str):
        """
        Instantiate the DreamerV3 JAX agent and restore weights from checkpoint.

        The obs_space / act_space must match the training environment (spot.py).
        """
        try:
            self._ensure_dreamerv3_on_path(checkpoint_path)
            dreamerv3 = importlib.import_module('dreamerv3')
            embodied = importlib.import_module('dreamerv3.embodied')
            import ruamel.yaml as yaml

            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                self.get_logger().error(f'Checkpoint not found: {checkpoint_path}')
                return

            # ── Load config from the same directory as the checkpoint ──────────
            config_file = checkpoint_path.parent / 'config.yaml'
            if not config_file.exists():
                self.get_logger().error(f'config.yaml not found next to checkpoint: {config_file}')
                return

            raw_cfg = yaml.YAML(typ='safe').load(config_file.read_text())
            config = embodied.Config(dreamerv3.Agent.configs['defaults'])
            config = config.update(raw_cfg)
            # Run in eval mode on CPU (no GPU required for deployment)
            config = config.update({
                'jax.platform': 'cpu',
                'jax.prealloc': False,
                'jax.jit': True,
                'batch_size': 1,
                'batch_length': 1,
            })

            # ── Rebuild observation and action spaces matching spot.py ─────────
            obs_space = {
                'image':        embodied.Space(np.uint8,   (self._img_h, self._img_w, 3)),
                'velocity':     embodied.Space(np.float32, (6,)),
                'position':     embodied.Space(np.float32, (3,)),
                'orientation':  embodied.Space(np.float32, (4,)),
                'goal':         embodied.Space(np.float32, (2,)),
                'info_terrain': embodied.Space(np.float32, (self._ter_h, self._ter_w)),
                'reward':       embodied.Space(np.float32),
                'is_first':     embodied.Space(bool),
                'is_last':      embodied.Space(bool),
                'is_terminal':  embodied.Space(bool),
            }
            act_space = {
                'action': embodied.Space(np.float32, (4,), -1.0, 1.0),
                'reset':  embodied.Space(bool),
            }

            step = embodied.Counter()
            self._agent = dreamerv3.Agent(obs_space, act_space, step, config)

            # ── Restore weights ───────────────────────────────────────────────
            # Inject optax.transforms shim: checkpoints pickled with older optax
            # reference 'optax.transforms' which no longer exists as a submodule.
            self._patch_optax_transforms()
            checkpoint = embodied.Checkpoint()
            checkpoint.agent = self._agent
            checkpoint.load(str(checkpoint_path), keys=['agent'])

            self.get_logger().info(f'✓ Agent loaded from {checkpoint_path}')

        except Exception as e:
            self.get_logger().error(f'Failed to load agent: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _patch_optax_transforms(self):
        """
        Install an import hook that satisfies any 'optax.transforms.*' import.

        Checkpoints pickled with older optax contain references to private
        sub-modules like 'optax.transforms._conditionality' that no longer exist.
        pickle.loads() calls Python's import machinery directly, so a simple
        module-attribute shim is not enough — we need a meta-path finder that
        intercepts those imports and returns a proxy pointing back to optax.
        """
        import types
        import importlib.abc
        import importlib.machinery

        try:
            import optax
        except ImportError:
            self.get_logger().warn('optax not importable — skipping shim.')
            return

        if 'optax.transforms' in sys.modules:
            return  # already patched

        class _OptaxTransformsFinder(importlib.abc.MetaPathFinder):
            """Intercept any import whose name starts with 'optax.transforms'."""

            def find_spec(self, fullname, path, target=None):
                if not fullname.startswith('optax.transforms'):
                    return None
                loader = _OptaxTransformsLoader()
                return importlib.machinery.ModuleSpec(fullname, loader)

        class _OptaxTransformsLoader(importlib.abc.Loader):
            def create_module(self, spec):
                if spec.name in sys.modules:
                    return sys.modules[spec.name]
                mod = types.ModuleType(spec.name)
                # Forward attribute lookups to top-level optax.
                mod.__getattr__ = lambda name: getattr(optax, name, None)
                sys.modules[spec.name] = mod
                return mod

            def exec_module(self, module):
                pass  # nothing to execute; attributes come via __getattr__

        sys.meta_path.insert(0, _OptaxTransformsFinder())
        self.get_logger().info('Installed optax.transforms import hook.')

    def _ensure_dreamerv3_on_path(self, checkpoint_path: str):
        """Add plausible dreamerv3 source roots to sys.path for container and host layouts."""
        # If no explicit root is set, check if dreamerv3 is already importable.
        # Skip the early-return when an explicit root is provided — we must ensure
        # the correct version wins even if an incompatible installed copy exists.
        if not self._dreamerv3_root:
            try:
                importlib.import_module('dreamerv3')
                return
            except Exception:
                pass

        candidates = []

        # 1) Explicit override parameter has highest priority.
        if self._dreamerv3_root:
            candidates.append(Path(self._dreamerv3_root))

        # 2) Environment variable override (useful in docker launch scripts).
        env_root = os.getenv('DREAMERV3_ROOT', '')
        if env_root:
            candidates.append(Path(env_root))

        # 3) Common container workspace layouts.
        candidates.extend([
            Path('/home/ob/openbots_ws/src/dreamer_SPOT_implementation/informed-dreamer'),
            Path('/home/ob/openbots_ws/src/packages/dreamer_SPOT_implementation/informed-dreamer'),
            Path('/workspaces/openbots_ws/src/dreamer_SPOT_implementation/informed-dreamer'),
            Path('/workspaces/openbots_ws/src/packages/dreamer_SPOT_implementation/informed-dreamer'),
        ])

        # 4) Current repository-relative guesses (for local runs).
        repo_root_guess = Path(__file__).resolve().parents[4]
        candidates.extend([
            repo_root_guess / 'dreamer_SPOT_implementation' / 'informed-dreamer',
            repo_root_guess / '..' / 'dreamer_SPOT_implementation' / 'informed-dreamer',
        ])

        # 5) Sibling to checkpoint directory (if user keeps code near logs).
        ckpt = Path(checkpoint_path)
        candidates.append(ckpt.parent.parent / 'dreamer_SPOT_implementation' / 'informed-dreamer')

        # Keep first match only; preserve import priority by inserting at index 0.
        tried = []
        for cand in candidates:
            cand = cand.expanduser().resolve()
            tried.append(str(cand))
            if (cand / 'dreamerv3').exists():
                if str(cand) not in sys.path:
                    sys.path.insert(0, str(cand))
                # Purge any cached dreamerv3 modules so everything reimports
                # cleanly from the correct source tree (avoids version mixing
                # when a colcon-installed dreamerv3 is also on the path).
                stale = [k for k in sys.modules if k == 'dreamerv3' or k.startswith('dreamerv3.')]
                for k in stale:
                    del sys.modules[k]
                self.get_logger().info(f'Using dreamerv3 from: {cand}')
                return

        self.get_logger().error('Could not locate dreamerv3 source tree. Tried:')
        for path in tried:
            self.get_logger().error(f'  - {path}')

    # ──────────────────────────────────────────────────────────────────────────
    # SENSOR CALLBACKS
    # ──────────────────────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image):
        """Decode front camera image and resize to training resolution."""
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Apply same preprocessing as convert_bag_to_hdf5.py:
            # Use PIL.rotate() to match converter exactly (PIL.rotate(-90) = counterclockwise)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img)
            pil_img = pil_img.rotate(-90, expand=True)
            img = np.array(pil_img)
            
            img = cv2.resize(img, (self._img_w, self._img_h), interpolation=cv2.INTER_LINEAR)
            self._latest_image = img.astype(np.uint8)
        except Exception as e:
            self.get_logger().warn(f'Image decode error: {e}')

    def _odometry_callback(self, msg: Odometry):
        """Store latest odometry; update robot position for goal computation."""
        self._latest_odometry = msg
        p = msg.pose.pose.position
        self._robot_position = np.array([p.x, p.y, p.z], dtype=np.float32)

    def _terrain_callback(self, msg: OccupancyGrid):
        """Decode OccupancyGrid terrain into a float32 array matching training shape."""
        try:
            data = np.array(msg.data, dtype=np.float32)
            h = msg.info.height
            w = msg.info.width
            grid = data.reshape(h, w)
            # Resize to the training resolution if dimensions differ.
            if (h, w) != (self._ter_h, self._ter_w):
                grid = cv2.resize(grid, (self._ter_w, self._ter_h), interpolation=cv2.INTER_LINEAR)
            self._latest_terrain = grid
        except Exception as e:
            self.get_logger().warn(f'Terrain decode error: {e}')

    # ──────────────────────────────────────────────────────────────────────────
    # GOAL TOPIC
    # ──────────────────────────────────────────────────────────────────────────

    def _goal_callback(self, msg: PointStamped):
        """
        Set the navigation goal relative to the robot's current position.

        The x/y in the message are treated as a displacement from where the
        robot is standing right now, NOT absolute world-frame coordinates.
        This means you don't need to know the accumulated odometry offset.

        Example: publish {x: 2.0, y: 0.0} to drive 2 m straight ahead in
        the odom frame from the current position.

        Publish with:
          ros2 topic pub --once /spot/policy/goal \\
            geometry_msgs/msg/PointStamped \\
            "{header: {frame_id: 'odom'}, point: {x: 2.0, y: 0.0, z: 0.0}}"
        """
        if self._robot_position is None:
            self.get_logger().warn('No odometry yet — cannot set relative goal.')
            return
        dx, dy = msg.point.x, msg.point.y
        abs_x = float(self._robot_position[0]) + dx
        abs_y = float(self._robot_position[1]) + dy
        self._goal_world    = np.array([abs_x, abs_y], dtype=np.float32)
        self._is_first_step = True
        self._rssm_state    = None
        self.get_logger().info(
            f'Goal set: robot=({self._robot_position[0]:.2f}, {self._robot_position[1]:.2f})'
            f'  +  offset=({dx:.2f}, {dy:.2f})'
            f'  →  world=({abs_x:.2f}, {abs_y:.2f})'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # OBSERVATION BUILDING
    # ──────────────────────────────────────────────────────────────────────────

    def _build_observation(self) -> dict:
        """
        Assemble one observation dict matching spot.py exactly:
          image        (H, W, 3)  uint8
          velocity     (6,)       float32  [vx, vy, vz, wx, wy, wz]
          position     (3,)       float32  [x, y, z]
          orientation  (4,)       float32  [qx, qy, qz, qw]
          goal         (2,)       float32  [goal_x - robot_x, goal_y - robot_y]
          info_terrain (H, W)     float32
          reward       scalar     float32  (always 0.0 during deployment)
          is_first     bool
          is_last      bool
          is_terminal  bool
        """
        odom = self._latest_odometry
        if odom is None or self._latest_image is None:
            return None

        # Extract orientation for velocity rotation
        q = odom.pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        orientation = np.array([qx, qy, qz, qw], dtype=np.float32)
        
        # Velocity: odometry publishes in world frame, rotate to body frame
        # using the same formula as convert_bag_to_hdf5.py
        lv = odom.twist.twist.linear
        av = odom.twist.twist.angular
        
        # Extract yaw from quaternion
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        # Rotate linear velocity from world frame to body frame
        vwx = float(lv.x)
        vwy = float(lv.y)
        vbx = np.cos(yaw) * vwx + np.sin(yaw) * vwy
        vby = -np.sin(yaw) * vwx + np.cos(yaw) * vwy
        
        velocity = np.array([vbx, vby, lv.z, av.x, av.y, av.z], dtype=np.float32)

        # Position [x, y, z]
        p = odom.pose.pose.position
        position = np.array([p.x, p.y, p.z], dtype=np.float32)

        # Relative goal in ego-centric (body) frame — must match training!
        # Training (spot.py) computes goal as rotated by yaw using:
        #   goal_ego = R(yaw) @ (goal_world - position)
        # where R(yaw) is 2D rotation matrix.
        # This must match for consistent inference.
        if self._goal_world is not None:
            dx = self._goal_world[0] - position[0]
            dy = self._goal_world[1] - position[1]
            # Rotate from world frame to body frame (same as velocity rotation)
            goal_x = np.cos(yaw) * dx + np.sin(yaw) * dy
            goal_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
            goal = np.array([goal_x, goal_y], dtype=np.float32)
            if self._invert_goal_sign:
                goal = -goal
        else:
            goal = np.zeros(2, dtype=np.float32)

        terrain = self._latest_terrain

        # Batch dimension expected by JAX agent: add leading axis → (1, ...)
        obs = {
            'image':        self._latest_image[None],        # (1, H, W, 3)
            'velocity':     velocity[None],                   # (1, 6)
            'position':     position[None],                   # (1, 3)
            'orientation':  orientation[None],                # (1, 4)
            'goal':         goal[None],                       # (1, 2)
            'info_terrain': terrain[None],                    # (1, H, W)
            'reward':       np.zeros((1,), dtype=np.float32),
            'is_first':     np.array([self._is_first_step]),
            'is_last':      np.array([False]),
            'is_terminal':  np.array([False]),
        }
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # CONTROL LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def _control_loop(self):
        """
        Run one policy step:
          obs → agent.policy() → action (4D) → cmd_vel + locomotion mode
        """
        if self._agent is None:
            return

        if self._goal_world is None:
            self.get_logger().warn('No goal set — publish zero velocity. Use /spot/policy/set_goal.', throttle_duration_sec=5.0)
            self._cmd_vel_pub.publish(Twist())
            return

        # ── Goal-reached check ────────────────────────────────────────────────
        dist = float(np.linalg.norm(self._goal_world - self._robot_position[:2]))
        if dist < 0.5:
            self.get_logger().info(
                f'Goal reached (dist={dist:.2f} m < 0.5 m) — stopping.',
                throttle_duration_sec=2.0,
            )
            self._cmd_vel_pub.publish(Twist())   # zero velocity
            return

        obs = self._build_observation()
        if obs is None:
            self.get_logger().warn('Waiting for sensor data …', throttle_duration_sec=5.0)
            return

        try:
            # Debug: log observations every 5 seconds to diagnose coordinate frame issues
            if self._step_counter % 25 == 0:  # 25 steps * 0.1s = 2.5s
                self.get_logger().info(
                    f'[OBS] pos={self._robot_position[:2]}, goal_world={self._goal_world}, '
                    f'goal_rel={obs["goal"][0]}, vel_body={obs["velocity"][0, :3]}',
                    throttle_duration_sec=1.0,
                )
            self._step_counter += 1

            outs, self._rssm_state = self._agent.policy(
                obs, self._rssm_state, mode='eval'
            )
            self._is_first_step = False

            # action shape: (1, 4) → extract first (and only) batch element
            action = np.array(outs['action'][0], dtype=np.float32)

            self._decode_and_publish(action)

        except Exception as e:
            self.get_logger().error(f'Policy inference error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    # ──────────────────────────────────────────────────────────────────────────
    # ACTION DECODING
    # ──────────────────────────────────────────────────────────────────────────

    def _decode_and_publish(self, action: np.ndarray):
        """
        Decode 4D action vector to ROS commands exactly as spot.py._action() does.

          action[3] > 0  → HINT_TROT,   max_vel = 2.0 m/s
          action[3] <= 0 → HINT_CRAWL,  max_vel = 1.0 m/s

          vel_x   = action[0] * max_vel
          vel_y   = action[1] * max_vel
          vel_yaw = action[2] * 1.0
        """
        gait    = _HINT_TROT if action[3] > 0 else _HINT_CRAWL
        max_vel = 2.0        if action[3] > 0 else 1.0

        # Clip to SPOT's hard API limits (tanh can slightly exceed ±1.0).
        # Use 1.5 instead of 1.6 for angular: SPOT's check is strictly > ±1.6
        # so clipping to the boundary still triggers the rejection.
        _MAX_LIN = 1.9   # m/s  (SPOT hard limit is 2.0, stay clear)
        _MAX_YAW = 1.5   # rad/s (SPOT hard limit is 1.6, stay clear)

        vel_x   = action[0]
        if self._invert_action_x_sign:
            vel_x = -vel_x
        vel_x   = float(np.clip(vel_x * max_vel, -_MAX_LIN, _MAX_LIN))
        vel_y   = float(np.clip(action[1] * max_vel, -_MAX_LIN, _MAX_LIN))
        vel_yaw = float(np.clip(action[2] * 1.0,     -_MAX_YAW, _MAX_YAW))

        # ── Publish cmd_vel ───────────────────────────────────────────────────
        twist = Twist()
        twist.linear.x  = vel_x
        twist.linear.y  = vel_y
        twist.angular.z = vel_yaw
        self._cmd_vel_pub.publish(twist)

        # ── Request gait change (non-blocking) ────────────────────────────────
        if self._locomotion_client.service_is_ready():
            req = SetLocomotion.Request()
            req.locomotion_mode = gait
            self._locomotion_client.call_async(req)

        # ── Debug publisher ───────────────────────────────────────────────────
        dbg = Float32MultiArray()
        dbg.data = action.tolist()
        self._action_debug_pub.publish(dbg)

        dist = float(np.linalg.norm(self._goal_world - self._robot_position[:2])) if self._goal_world is not None else float('nan')
        self.get_logger().info(
            f'[policy] raw=({action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f},{action[3]:+.3f})'
            f'  →  vx={vel_x:+.2f} vy={vel_y:+.2f} vyaw={vel_yaw:+.2f}  '
            f'gait={"TROT" if gait == _HINT_TROT else "CRAWL"}  dist_to_goal={dist:.2f}m',
            throttle_duration_sec=2.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = DreamerPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down …')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()   # no-op if already shut down by signal handler


if __name__ == '__main__':
    main()

