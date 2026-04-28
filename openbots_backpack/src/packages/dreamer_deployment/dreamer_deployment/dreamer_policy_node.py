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
  action[0] * max_vel        → cmd_vel.linear.x   (max_vel: trot=2.0, crawl=1.0)
  action[1] * (max_vel/2.0)  → cmd_vel.linear.y   (SPOT vy limit = vx/2)
  action[2] * 1.0            → cmd_vel.angular.z
  action[3] > 0              → locomotion HINT_TROT (max_vel=2.0)
  action[3] <= 0             → locomotion HINT_AUTO (max_vel=1.0)

To switch policies, only change `checkpoint_path` in params.yaml.
"""

import os
import sys
import time
import importlib
from pathlib import Path

import threading
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
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

        # DEPLOYMENT CONTROL: disable gait selection to use fixed locomotion mode
        # When disabled, always uses fixed_gait_mode (trot or crawl) regardless of action[3]
        # Useful to test if gait selection is causing deployment issues.
        self.declare_parameter('disable_gait_selection', False)
        self.declare_parameter('fixed_gait_mode', 'trot')  # 'trot' or 'crawl'

        # JAX platform: 'cpu' or 'gpu'. Use 'gpu' if NVIDIA GPU is available in the container.
        self.declare_parameter('jax_platform', 'cpu')
        
        # JAX precision: 'float32' (default) or 'float16' (half precision for memory savings).
        # Both xlargerssm and dyn1.0 have deter=4096 which OOMs on GTX 1050 at float32.
        self.declare_parameter('jax_precision', 'float32')

        # Policy inference rate — should match the dataset recording frequency (~3.5 Hz).
        # cmd_vel is still republished at control_rate_hz between policy steps so SPOT
        # doesn't hit its command timeout.
        self.declare_parameter('policy_rate_hz', 3.5)

        checkpoint_path  = self.get_parameter('checkpoint_path').value
        control_rate_hz  = self.get_parameter('control_rate_hz').value
        policy_rate_hz   = float(self.get_parameter('policy_rate_hz').value)
        robot_name       = self.get_parameter('robot_name').value
        self._dreamerv3_root = self.get_parameter('dreamerv3_root').value
        self._img_h      = self.get_parameter('image_height').value
        self._img_w      = self.get_parameter('image_width').value
        self._ter_h      = self.get_parameter('terrain_height').value
        self._ter_w      = self.get_parameter('terrain_width').value
        self._invert_goal_sign = self.get_parameter('invert_goal_sign').value
        self._invert_action_x_sign = self.get_parameter('invert_action_x_sign').value
        self._disable_gait_selection = self.get_parameter('disable_gait_selection').value
        self._fixed_gait_mode = self.get_parameter('fixed_gait_mode').value
        self._jax_platform = self.get_parameter('jax_platform').value
        self._jax_precision = self.get_parameter('jax_precision').value

        # How many control ticks between RSSM policy calls.
        # e.g. control=10Hz, policy=3.5Hz → stride=3 (call every 3rd tick ≈ 3.33Hz)
        self._policy_stride = max(1, round(control_rate_hz / policy_rate_hz))
        self._tick_counter  = 0

        self.get_logger().info('=== Dreamer Policy Node ===')
        self.get_logger().info(f'  Checkpoint : {checkpoint_path}')
        self.get_logger().info(f'  Control Hz : {control_rate_hz}  (policy every {self._policy_stride} ticks ≈ {control_rate_hz/self._policy_stride:.1f} Hz)')
        self.get_logger().info(f'  Image size : {self._img_h}x{self._img_w}')
        self.get_logger().info(f'  Terrain sz : {self._ter_h}x{self._ter_w}')
        if self._dreamerv3_root:
            self.get_logger().info(f'  dreamerv3 root override : {self._dreamerv3_root}')
        
        # Log deployment control settings
        if self._disable_gait_selection:
            self.get_logger().warn(f'  ⚠️  Gait selection DISABLED — using fixed mode: {self._fixed_gait_mode}')
        else:
            self.get_logger().info(f'  Gait selection: enabled (action[3]-driven)')

        # ============ SENSOR STATE ============
        self._bridge             = CvBridge()
        self._latest_image       = None   # np.uint8 (H, W, 3)
        self._latest_odometry    = None   # nav_msgs/Odometry
        self._latest_terrain     = np.zeros((self._ter_h, self._ter_w), dtype=np.float32)
        self._robot_position     = np.zeros(3, dtype=np.float32)  # x, y, z
        self._last_valid_observation = None  # cache last good observation for robustness to sensor lag

        # Goal in world frame; set via /spot/policy/set_goal service
        self._goal_world         = None   # np.float32 (2,) — [x, y]
        self._is_first_step      = True   # tells RSSM this is the start of an episode
        self._step_counter       = 0      # for throttled logging

        # Episode origin for zero-centering — matches convert_bag_to_hdf5 _zero_center_trajectory.
        # Training data always starts at (0,0,0) facing +X, so we must do the same at deployment.
        self._episode_pos0       = None   # np.float32 (3,) — world position at episode start
        self._episode_yaw0       = None   # float — yaw at episode start

        # ── Inference threading ───────────────────────────────────────────────
        # Inference runs in a background thread so the ROS executor is never
        # blocked and sensor callbacks always receive fresh data.
        self._inference_lock    = threading.Lock()   # guards _rssm_state
        self._inference_running = False              # prevents overlapping calls
        self._last_twist        = None               # held and republished between policy steps
        self._prev_vel          = np.zeros(3, dtype=np.float32)  # [vx, vy, vyaw] of previous cmd for rate limiting

        # ── Keyboard input thread ─────────────────────────────────────────────
        # Listen for 'r' key in terminal to trigger reset without reloading agent
        self._keyboard_thread = None
        self._keyboard_running = False

        # ── Sensor status tracking ────────────────────────────────────────────
        # Track which sensors have received data to handle restart scenarios
        self._sensor_ready = {
            'image': False,
            'odometry': False,
            'terrain': False,
        }

        # ============ DREAMER AGENT ============
        self._agent    = None
        self._rssm_state = None           # recurrent state maintained between steps
        self._latest_obs_for_record = None  # obs snapshot used by recorder
        self._recording_session_id = None

        # ── Command recording ─────────────────────────────────────────────────
        ckpt_dir = Path(checkpoint_path).parent
        self._recordings_dir = ckpt_dir / 'recordings'
        self._recordings_dir.mkdir(parents=True, exist_ok=True)
        self._recording_buffer: list = []
        self._recording_active: bool = False
        self._recording_lock = threading.Lock()
        self.get_logger().info(f'Recording directory: {self._recordings_dir}')

        self._load_agent(checkpoint_path)

        # ============ SUBSCRIBERS ============
        # Mirror the exact topics from episode_bag_recorder.py
        self.get_logger().info('Creating subscriptions with sensor QoS ...')
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
        self.get_logger().info('Subscriptions created. Waiting for topic publishers ...')

        # ============ PUBLISHERS ============
        # Primary output: velocity commands consumed by spot_driver
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        # Debug: raw 4D action vector for monitoring / logging
        self._action_debug_pub = self.create_publisher(
            Float32MultiArray, '/spot/policy_action_debug', 1
        )

        # ============ SERVICE CLIENTS ============
        self._locomotion_client = self.create_client(SetLocomotion, '/locomotion_mode')

        # ============ SERVICE SERVERS ============
        # Reset the policy state without reloading the agent
        self.create_service(Empty, '/dreamer_policy/reset', self._handle_reset_service)
        self.get_logger().info('Reset service available at /dreamer_policy/reset (use: ros2 service call /dreamer_policy/reset std_srvs/Empty)')

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
        
        # ============ SENSOR WARMUP ============
        # Wait for all sensors to connect and deliver at least one message.
        # This handles restart scenarios where subscriptions need time to reconnect.
        self.get_logger().info('Waiting for sensor data …')
        self._wait_for_sensors(timeout_sec=5.0)  # Reduced from 30s: checkpoint loads fast now

        # ============ TERMINAL INPUT ============
        # Single stdin thread handles goal prompts AND reset commands.
        # Previously two threads (readchar keyboard_listener + input() goal_prompt)
        # fought over stdin — readchar put the tty in raw mode and swallowed all
        # keystrokes before input() could see them.
        self._keyboard_running = True
        self._stdin_thread = threading.Thread(target=self._stdin_handler, daemon=True)
        self._stdin_thread.start()

        self.get_logger().info('✓ All sensors ready. Policy node initialised.')

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
            # Run in eval mode; fall back to CPU if requested platform is unavailable
            import jax
            platform = self._jax_platform
            try:
                jax.devices(platform)
            except Exception:
                self.get_logger().warn(
                    f'Platform "{platform}" unavailable — falling back to cpu.')
                platform = 'cpu'
            config = config.update({
                'jax.platform': platform,
                'jax.precision': self._jax_precision,
                'jax.prealloc': False,
                'jax.jit': True,
                'batch_size': 1,
                'batch_length': 1,
                # Reduce imagination horizon for init-only trace to cut peak GPU
                # memory during XLA compilation.  The full imag_horizon is only
                # needed for training; inference (policy()) never uses it.
                'imag_horizon': 1,
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

            # Inject optax.transforms shim: checkpoints pickled with older optax
            # reference 'optax.transforms' which no longer exists as a submodule.
            self.get_logger().info('Setting up optax shim...')
            self._patch_optax_transforms()
            
            self.get_logger().info(f'Loading checkpoint from: {checkpoint_path}')
            checkpoint = embodied.Checkpoint()
            checkpoint.agent = self._agent
            self.get_logger().info('Calling checkpoint.load()...')
            checkpoint.load(str(checkpoint_path), keys=['agent'])

            self.get_logger().info(f'✓ Agent loaded from {checkpoint_path}')

            # ── Pre-trace mode='eval' so first real inference has no JIT delay ──
            # _init_varibs() only traces mode='train'. The first call to
            # agent.policy(..., mode='eval') would trigger full XLA compilation,
            # which can take 30-60 s for large models. During that time
            # _inference_running=True and _last_twist=None → no cmd_vel published
            # → SPOT's command timeout fires → robot ignores subsequent commands.
            # Warm-up here (before rclpy.spin) so compilation happens at startup,
            # not mid-episode.
            self.get_logger().info('Pre-tracing policy (mode=eval) — this may take 30-60 s …')
            dummy_obs = {
                'image':        np.zeros((1, self._img_h, self._img_w, 3), dtype=np.uint8),
                'velocity':     np.zeros((1, 6), dtype=np.float32),
                'position':     np.zeros((1, 3), dtype=np.float32),
                'orientation':  np.array([[0, 0, 0, 1]], dtype=np.float32),
                'goal':         np.zeros((1, 2), dtype=np.float32),
                'info_terrain': np.zeros((1, self._ter_h, self._ter_w), dtype=np.float32),
                'reward':       np.zeros((1,), dtype=np.float32),
                'is_first':     np.array([True]),
                'is_last':      np.array([False]),
                'is_terminal':  np.array([False]),
            }
            _, _ = self._agent.policy(dummy_obs, None, mode='eval')
            self.get_logger().info('✓ Policy trace complete — inference will be fast.')

        except Exception as e:
            self.get_logger().error(f'Failed to load agent: {e}')
            import traceback
            self.get_logger().error('Full traceback:')
            for line in traceback.format_exc().split('\n'):
                self.get_logger().error(line)

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
    # SENSOR WARMUP
    # ──────────────────────────────────────────────────────────────────────────

    def _wait_for_sensors(self, timeout_sec: float = 5.0):
        """
        Block until all sensors have delivered at least one message.
        This handles restart scenarios where subscriptions need time to reconnect.
        
        Args:
            timeout_sec: Maximum time to wait in seconds
        """
        start_time = time.time()
        timeout = start_time + timeout_sec
        missing_sensors = set(self._sensor_ready.keys())
        
        while time.time() < timeout:
            # Check which sensors have delivered data
            missing_sensors = {s for s in self._sensor_ready.keys() if not self._sensor_ready[s]}
            
            if not missing_sensors:
                self.get_logger().info(
                    f'✓ All sensors connected: image, odometry, terrain'
                )
                return  # Success!
            
            elapsed = time.time() - start_time
            self.get_logger().info(
                f'  Waiting for sensors … ({elapsed:.1f}s) Missing: {", ".join(sorted(missing_sensors))}',
                throttle_duration_sec=2.0,  # Log every 2 seconds max
            )
            time.sleep(0.5)
        
        # Timeout reached
        missing = ', '.join(sorted(missing_sensors))
        self.get_logger().error(
            f'Timeout waiting for sensors! Missing after {timeout_sec}s: {missing}\n'
            f'  • Is the SPOT driver running? (Run: ros2 launch ... spot_driver.launch.py)\n'
            f'  • Check topic connectivity: ros2 topic list | grep -E "camera|odometry|terrain"\n'
            f'  • Check: ros2 topic echo /camera/frontmiddle_virtual/image  # (Ctrl+C to exit)'
        )
        # Continue anyway (inference will just warn about missing data)

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
            if not self._sensor_ready['image']:
                self.get_logger().info('✓ Image subscription connected')
            self._sensor_ready['image'] = True
        except Exception as e:
            self.get_logger().warn(f'Image decode error: {e}')

    def _odometry_callback(self, msg: Odometry):
        """Store latest odometry; update robot position for goal computation."""
        self._latest_odometry = msg
        p = msg.pose.pose.position
        self._robot_position = np.array([p.x, p.y, p.z], dtype=np.float32)
        if not self._sensor_ready['odometry']:
            self.get_logger().info('✓ Odometry subscription connected')
        self._sensor_ready['odometry'] = True

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
            if not self._sensor_ready['terrain']:
                self.get_logger().info('✓ Terrain subscription connected')
            self._sensor_ready['terrain'] = True
        except Exception as e:
            self.get_logger().warn(f'Terrain decode error: {e}')

    # ──────────────────────────────────────────────────────────────────────────
    # GOAL TOPIC
    # ──────────────────────────────────────────────────────────────────────────

    def _goal_callback(self, msg: PointStamped):
        """
        Set the navigation goal relative to the robot's current position.

        By default the x/y offset is interpreted in the ROBOT'S BODY FRAME:
          x = forward distance (positive = ahead)
          y = lateral distance (positive = left)

        The offset is rotated into the odom/world frame using the robot's
        current yaw so the model always sees a goal that is in-distribution
        (mostly forward, small lateral), matching the teleop training data.

        Publish with:
          ros2 topic pub --once /spot/policy/goal \\
            geometry_msgs/msg/PointStamped \\
            "{header: {frame_id: 'body'}, point: {x: 3.0, y: 0.0, z: 0.0}}"

        If frame_id is 'odom' the offset is treated as a world-frame
        displacement (old behaviour) instead.
        """
        if self._robot_position is None:
            self.get_logger().warn('No odometry yet — cannot set relative goal.')
            return

        dx_in, dy_in = msg.point.x, msg.point.y
        frame = msg.header.frame_id if msg.header.frame_id else 'body'

        odom = self._latest_odometry
        if odom is not None:
            q = odom.pose.pose.orientation
            yaw = float(np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)))
        else:
            yaw = 0.0

        if frame == 'odom':
            # Legacy: treat offset as world-frame displacement
            dx_world, dy_world = dx_in, dy_in
        else:
            # Default: rotate body-frame (forward, left) offset into world frame
            dx_world = np.cos(yaw) * dx_in - np.sin(yaw) * dy_in
            dy_world = np.sin(yaw) * dx_in + np.cos(yaw) * dy_in

        abs_x = float(self._robot_position[0]) + dx_world
        abs_y = float(self._robot_position[1]) + dy_world

        self._goal_world = np.array([abs_x, abs_y], dtype=np.float32)
        self._reset_episode_state(yaw if odom is not None else 0.0, odom)

        # Compute expected initial goal_rel for logging (should be ≈ [dx_in, dy_in])
        expected_rel_x = np.cos(yaw) * (abs_x - self._robot_position[0]) + np.sin(yaw) * (abs_y - self._robot_position[1])
        expected_rel_y = -np.sin(yaw) * (abs_x - self._robot_position[0]) + np.cos(yaw) * (abs_y - self._robot_position[1])

        offset_label = 'offset_body' if frame != 'odom' else 'offset_world'
        self.get_logger().info(
            f'🎯 Goal set [{frame}]: {offset_label}=({dx_in:.2f}, {dy_in:.2f})'
            f'  →  model sees: pos=(0.00, 0.00)  goal=({expected_rel_x:.2f}, {expected_rel_y:.2f})'
            f'  dist={np.hypot(expected_rel_x, expected_rel_y):.2f}m  yaw0={np.degrees(yaw):.1f}°'
        )
        self._start_recording()

    # ──────────────────────────────────────────────────────────────────────────
    # KEYBOARD INPUT
    # ──────────────────────────────────────────────────────────────────────────

    def _stdin_handler(self):
        """
        Single background thread for all terminal interaction.

        Prompts for a goal (body frame: x=forward, y=left), then waits for
        any Enter key press to reset and re-prompt.  Uses plain input() so
        there is no stdin contention — the old readchar keyboard_listener
        was putting the tty in raw mode and swallowing the goal prompt input.
        """
        while self._keyboard_running:
            try:
                # ── Goal prompt ───────────────────────────────────────────
                print('\n──────────────────────────────────────────────────────')
                print('  🎯  Set goal  (body frame: x = forward m, y = left m)')
                print('──────────────────────────────────────────────────────')

                while True:
                    try:
                        x_str = input('  X (forward, m): ').strip()
                        if not x_str:
                            continue
                        x = float(x_str)
                        break
                    except ValueError:
                        print(f'  ⚠  Not a number — try again.')

                while True:
                    try:
                        y_str = input('  Y (lateral, m): ').strip()
                        if not y_str:
                            y_str = '0'
                        y = float(y_str)
                        break
                    except ValueError:
                        print(f'  ⚠  Not a number — try again.')

                # Rotate body-frame offset into world frame (mirrors _goal_callback)
                odom = self._latest_odometry
                yaw = 0.0
                if odom is not None:
                    q = odom.pose.pose.orientation
                    yaw = float(np.arctan2(
                        2*(q.w*q.z + q.x*q.y),
                        1 - 2*(q.y*q.y + q.z*q.z)
                    ))
                dx_world = np.cos(yaw) * x - np.sin(yaw) * y
                dy_world = np.sin(yaw) * x + np.cos(yaw) * y
                abs_x = float(self._robot_position[0]) + dx_world
                abs_y = float(self._robot_position[1]) + dy_world
                self._goal_world = np.array([abs_x, abs_y], dtype=np.float32)
                self._reset_episode_state(yaw, odom)

                rel_x = np.cos(yaw)*(abs_x-self._robot_position[0]) + np.sin(yaw)*(abs_y-self._robot_position[1])
                rel_y = -np.sin(yaw)*(abs_x-self._robot_position[0]) + np.cos(yaw)*(abs_y-self._robot_position[1])
                self.get_logger().info(
                    f'🎯 Goal set [terminal]: body=({x:.2f}, {y:.2f})'
                    f'  →  model sees: goal=({rel_x:.2f}, {rel_y:.2f})'
                    f'  dist={np.hypot(rel_x, rel_y):.2f}m  yaw0={np.degrees(yaw):.1f}°'
                )
                self._start_recording()

                # ── Wait for reset ────────────────────────────────────────
                input('\n⌨️  Press Enter to set a new goal … ')
                self.get_logger().info('⟲ Reset — enter new goal.')
                yaw2 = 0.0
                odom2 = self._latest_odometry
                if odom2 is not None:
                    q2 = odom2.pose.pose.orientation
                    yaw2 = float(np.arctan2(
                        2*(q2.w*q2.z + q2.x*q2.y),
                        1 - 2*(q2.y*q2.y + q2.z*q2.z)
                    ))
                self._reset_episode_state(yaw2, odom2)

            except (EOFError, KeyboardInterrupt):
                self.get_logger().info('Terminal input closed.')
                break
            except Exception as e:
                self.get_logger().debug(f'stdin handler error: {e}')
                time.sleep(0.5)

    # ──────────────────────────────────────────────────────────────────────────
    # RESET SERVICE & EPISODE STATE
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_reset_service(self, request, response):
        """
        Service handler for /dreamer_policy/reset (std_srvs/Empty).
        Resets the policy state: clears goal, RSSM state, and episode tracking.
        Does NOT reload the agent — only resets internal state.
        
        Called via: ros2 service call /dreamer_policy/reset std_srvs/Empty
        """
        self.get_logger().info('⟲ Reset requested — clearing goal and RSSM state')
        yaw = 0.0
        if self._latest_odometry is not None:
            q = self._latest_odometry.pose.pose.orientation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        self._reset_episode_state(yaw, self._latest_odometry)
        return response

    def _reset_episode_state(self, yaw: float, odom):
        """
        Reset episode state: clear goal, RSSM state, and position/orientation tracking.
        Called when:
          1. A new goal is set via _goal_callback()
          2. Reset service is triggered via ros2 service call
        
        Args:
            yaw: Current robot yaw (rad) in world frame
            odom: Latest odometry message (may be None)
        """
        self._save_recording('reset')
        self._is_first_step = True
        self._rssm_state    = None
        self._last_twist    = None   # clear stale command
        self._prev_vel      = np.zeros(3, dtype=np.float32)  # reset rate limiter
        self._tick_counter  = 0      # fire inference on next stride tick
        self._last_valid_observation = None  # clear stale cached observation

        # Snapshot episode origin for zero-centering (matches training data convention)
        if odom is not None:
            p = odom.pose.pose.position
            self._episode_pos0 = np.array([p.x, p.y, p.z], dtype=np.float32)
            self._episode_yaw0 = yaw
        else:
            self._episode_pos0 = self._robot_position.copy()
            self._episode_yaw0 = 0.0
        
        self.get_logger().info(
            f'  Episode state reset: pos0=({self._episode_pos0[0]:.2f}, {self._episode_pos0[1]:.2f}), '
            f'yaw0={np.degrees(self._episode_yaw0):.1f}°'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # INTERACTIVE GOAL INPUT
    # ──────────────────────────────────────────────────────────────────────────

    def _set_goal_from_input(self, x: float, y: float):
        """
        Set goal from user input coordinates.
        Mimics the behavior of _goal_callback but with user-provided coordinates.
        """
        if self._robot_position is None:
            self.get_logger().warn('No odometry yet — cannot set goal.')
            return

        # Goal is in world frame (odom)
        self._goal_world = np.array([float(x), float(y)], dtype=np.float32)
        
        # Reset episode state
        yaw = 0.0
        odom = self._latest_odometry
        if odom is not None:
            q = odom.pose.pose.orientation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        self._reset_episode_state(yaw, odom)
        
        # Compute and log expected goal_rel
        expected_rel_x = np.cos(yaw) * (x - self._robot_position[0]) + np.sin(yaw) * (y - self._robot_position[1])
        expected_rel_y = -np.sin(yaw) * (x - self._robot_position[0]) + np.cos(yaw) * (y - self._robot_position[1])
        
        self.get_logger().info(
            f'🎯 Goal set [user input]: offset_world=({x:.2f}, {y:.2f})'
            f'  →  model sees: pos=(0.00, 0.00)  goal=({expected_rel_x:.2f}, {expected_rel_y:.2f})'
            f'  dist={np.hypot(expected_rel_x, expected_rel_y):.2f}m  yaw0={np.degrees(yaw):.1f}°'
        )
        self._start_recording()

    # ──────────────────────────────────────────────────────────────────────────
    # COMMAND RECORDING
    # ──────────────────────────────────────────────────────────────────────────

    def _start_recording(self):
        """Start a new recording episode (called when a goal is set)."""
        self._recording_session_id = time.strftime('%Y%m%d_%H%M%S')
        with self._recording_lock:
            self._recording_buffer = []
            self._recording_active = True
        self.get_logger().info(f'Recording started → {self._recordings_dir}')

    def _save_recording(self, reason: str):
        """
        Flush the recording buffer to a .npz file and stop recording.
        Skips silently if no recording is active or the buffer is empty.
        reason: 'goal_reached' | 'reset' | 'shutdown'
        """
        with self._recording_lock:
            if not self._recording_active:
                return
            self._recording_active = False
            buf = self._recording_buffer
            self._recording_buffer = []

        if not buf:
            return

        session_id = self._recording_session_id or time.strftime('%Y%m%d_%H%M%S')
        fname = self._recordings_dir / f'episode_{session_id}_{reason}.npz'
        try:
            np.savez(
                str(fname),
                session_id   = np.array(session_id),
                t            = np.array([s['t']           for s in buf], dtype=np.float64),
                action       = np.array([s['action']       for s in buf], dtype=np.float32),
                vel_x        = np.array([s['vel_x']        for s in buf], dtype=np.float32),
                vel_y        = np.array([s['vel_y']        for s in buf], dtype=np.float32),
                vel_yaw      = np.array([s['vel_yaw']      for s in buf], dtype=np.float32),
                gait         = np.array([s['gait']         for s in buf], dtype=np.int32),
                dist_to_goal = np.array([s['dist_to_goal'] for s in buf], dtype=np.float32),
                pos_zc       = np.array([s['pos_zc']       for s in buf], dtype=np.float32),
                goal_zc      = np.array([s['goal_zc']      for s in buf], dtype=np.float32),
                velocity_obs = np.array([s['velocity_obs'] for s in buf], dtype=np.float32),
            )
            self.get_logger().info(
                f'Recording saved ({len(buf)} steps, reason={reason}): {fname.name}'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to save recording: {e}')
        finally:
            self._recording_session_id = None

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
        
        If fresh sensor data isn't available, reuses the last valid observation
        to handle transient network lag or sensor publishing delays.
        """
        odom = self._latest_odometry
        if odom is None or self._latest_image is None:
            # Sensor lag — reuse last valid observation if available
            if self._last_valid_observation is not None:
                return self._last_valid_observation
            # No valid observation cached yet (startup)
            return None

        # Extract orientation for velocity rotation
        q = odom.pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        
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

        # Position [x, y, z] — zero-centered to match training data convention.
        # convert_bag_to_hdf5._zero_center_trajectory() subtracts pos0 and rotates
        # by -yaw0 so every episode starts at (0,0,0) facing +X.
        p = odom.pose.pose.position
        raw_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        if self._episode_pos0 is not None:
            dp = raw_pos - self._episode_pos0
            cy0, sy0 = np.cos(-self._episode_yaw0), np.sin(-self._episode_yaw0)
            centered_x = cy0 * dp[0] - sy0 * dp[1]
            centered_y = sy0 * dp[0] + cy0 * dp[1]
            position = np.array([centered_x, centered_y, dp[2]], dtype=np.float32)

            # Zero-center orientation quaternion to match training data.
            # convert_bag_to_hdf5 applies R(-yaw0) * q_raw so episode starts at yaw=0.
            # Quaternion for R(-yaw0) around z: [0, 0, sin(-yaw0/2), cos(-yaw0/2)]
            sz = -np.sin(self._episode_yaw0 / 2)
            cz =  np.cos(self._episode_yaw0 / 2)
            orientation = np.array([
                cz*qx - sz*qy,
                cz*qy + sz*qx,
                cz*qz + sz*qw,
                cz*qw - sz*qz,
            ], dtype=np.float32)
        else:
            position = raw_pos
            orientation = np.array([qx, qy, qz, qw], dtype=np.float32)

        # [NEW] Goal delta in zero-centered world frame — same frame as position.
        # delta_zc = R(-yaw0) @ (goal_world - raw_pos)
        # Use this with new checkpoints trained after the frame-alignment fix.
        if self._goal_world is not None:
            dg = self._goal_world - raw_pos[:2]
            if self._episode_pos0 is not None:
                cy0 = np.cos(self._episode_yaw0)
                sy0 = np.sin(self._episode_yaw0)
                goal_x =  cy0 * dg[0] + sy0 * dg[1]
                goal_y = -sy0 * dg[0] + cy0 * dg[1]
                goal = np.array([goal_x, goal_y], dtype=np.float32)
            else:
                goal = dg.astype(np.float32)
            if self._invert_goal_sign:
                goal = -goal
        else:
            goal = np.zeros(2, dtype=np.float32)
        # [OLD] Ego-centric goal (body frame) — uncomment for old checkpoints.
        # if self._goal_world is not None:
        #     dx = self._goal_world[0] - raw_pos[0]
        #     dy = self._goal_world[1] - raw_pos[1]
        #     goal_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        #     goal_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        #     goal = np.array([goal_x, goal_y], dtype=np.float32)
        #     if self._invert_goal_sign:
        #         goal = -goal
        # else:
        #     goal = np.zeros(2, dtype=np.float32)

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
        # Cache this valid observation for robustness to sensor lag
        self._last_valid_observation = obs
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # CONTROL LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def _control_loop(self):
        """
        Timer callback — dispatches inference to a background thread immediately
        and returns so the ROS executor is never blocked.
        """
        if self._agent is None:
            return

        if self._goal_world is None:
            self.get_logger().warn('No goal set — publish zero velocity. Use /spot/policy/set_goal.', throttle_duration_sec=5.0)
            self._cmd_vel_pub.publish(Twist())
            self._tick_counter = 0  # reset so next goal triggers inference immediately
            return

        self._tick_counter += 1

        # Between policy steps: republish last cmd_vel so SPOT doesn't time out.
        # Publish zeros if no command has been produced yet — this keeps the
        # SPOT command stream alive even before the first inference step.
        if self._tick_counter % self._policy_stride != 0:
            self._cmd_vel_pub.publish(
                self._last_twist if self._last_twist is not None else Twist()
            )
            return

        # Skip if a previous inference call is still running
        if self._inference_running:
            self._cmd_vel_pub.publish(
                self._last_twist if self._last_twist is not None else Twist()
            )
            return

        self._inference_running = True
        t = threading.Thread(target=self._inference_thread, daemon=True)
        t.start()

    def _inference_thread(self):
        """Runs in a background thread — never blocks the ROS executor."""
        t_step_start = time.perf_counter()
        try:
            # ── Goal-reached check ────────────────────────────────────────────
            dist = float(np.linalg.norm(self._goal_world - self._robot_position[:2]))
            if dist < 0.5:
                self.get_logger().info(
                    f'Goal reached (dist={dist:.2f} m < 0.5 m) — stopping.',
                    throttle_duration_sec=2.0,
                )
                self._save_recording('goal_reached')
                self._cmd_vel_pub.publish(Twist())
                return

            obs = self._build_observation()
            if obs is None:
                # Sensors not yet ready — log throttled warning so this is visible
                self.get_logger().warn(
                    'Inference skipped: observation not available '
                    '(image or odometry not yet received). '
                    'Check /camera/frontmiddle_virtual/image and /odometry topics.',
                    throttle_duration_sec=5.0,
                )
                return

            # Debug: log what the model sees every 5 policy steps
            if self._step_counter % 5 == 0:
                pos_zc = obs['position'][0, :2]
                goal_r = obs['goal'][0]
                dist_g = float(np.linalg.norm(goal_r))
                self.get_logger().info(
                    f'[model] pos=({pos_zc[0]:+.2f}, {pos_zc[1]:+.2f})  '
                    f'goal=({goal_r[0]:+.2f}, {goal_r[1]:+.2f})  dist={dist_g:.2f}m  '
                    f'vel=({obs["velocity"][0,0]:+.2f}, {obs["velocity"][0,1]:+.2f})'
                )
            self._step_counter += 1

            t_obs_done = time.perf_counter()

            with self._inference_lock:
                t_inf_start = time.perf_counter()
                outs, self._rssm_state = self._agent.policy(
                    obs, self._rssm_state, mode='eval'
                )
                self._is_first_step = False
                t_inf_done = time.perf_counter()

            t_total = t_inf_done - t_step_start
            t_inf   = t_inf_done - t_inf_start
            t_build = t_obs_done - t_step_start
            self.get_logger().info(
                f'[TIMING] total={t_total:.3f}s  build_obs={t_build:.3f}s  jax_policy={t_inf:.3f}s'
            )

            action = np.array(outs['action'][0], dtype=np.float32)
            self._latest_obs_for_record = obs
            self._decode_and_publish(action)

        except Exception as e:
            self.get_logger().error(f'Policy inference error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            self._inference_running = False

    # ──────────────────────────────────────────────────────────────────────────
    # ACTION DECODING
    # ──────────────────────────────────────────────────────────────────────────

    def _decode_and_publish(self, action: np.ndarray):
        """
        Decode 4D action vector to ROS commands exactly as spot.py._action() does,
        corrected for SPOT's physical velocity asymmetry.

          action[3] > 0  → HINT_TROT,   max_vel = 2.0 m/s (vx), 1.0 m/s (vy)
          action[3] <= 0 → HINT_CRAWL,  max_vel = 1.0 m/s (vx), 0.5 m/s (vy)

          vel_x   = action[0] * max_vel          (forward/backward)
          vel_y   = action[1] * (max_vel / 2.0)  (lateral — SPOT caps vy at half vx)
          vel_yaw = action[2] * 1.0              (rotation, ±1.0 rad/s max)
        """
        # Determine gait and max velocity
        if self._disable_gait_selection:
            # Use fixed gait mode regardless of action[3]
            fixed_use_trot = self._fixed_gait_mode.lower() == 'trot'
            gait    = _HINT_TROT if fixed_use_trot else _HINT_CRAWL
            max_vel = 2.0        if fixed_use_trot else 1.0
        else:
            # Standard behavior: use action[3] to select gait
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
        vel_x   = float(np.clip(vel_x * max_vel,           -_MAX_LIN, _MAX_LIN))
        # SPOT's physical vy limit is half of vx: 1.0 m/s (trot), 0.5 m/s (crawl).
        # Training's spot.py uses action[1]*max_vel for vy (same as vx), but the
        # real robot caps vy at max_vel/2 — so we must match that here to avoid the
        # policy commanding lateral velocities that are never actually achieved.
        vel_y   = float(np.clip(action[1] * (max_vel / 2.0), -_MAX_LIN, _MAX_LIN))
        # Yaw: action[2] in [-1,1] * 1.0 → ±1.0 rad/s, well within SPOT's ±1.6 limit.
        vel_yaw = float(np.clip(action[2] * 1.0,             -_MAX_YAW, _MAX_YAW))

        # ── Rate limiting: cap change per step to prevent abrupt velocity jumps ─
        # 0.5 m/s/step for linear, 0.5 rad/s/step for yaw (at 0.3 s/step this is
        # ~1.67 m/s² acceleration limit, enough to go 0→2 m/s in 4 steps / 1.2 s).
        _MAX_DVLIN = 0.5   # m/s per policy step
        _MAX_DVYAW = 1.0   # rad/s per policy step (increased from 0.5 for faster turns)
        vel_x   = float(np.clip(vel_x,   self._prev_vel[0] - _MAX_DVLIN, self._prev_vel[0] + _MAX_DVLIN))
        vel_y   = float(np.clip(vel_y,   self._prev_vel[1] - _MAX_DVLIN, self._prev_vel[1] + _MAX_DVLIN))
        vel_yaw = float(np.clip(vel_yaw, self._prev_vel[2] - _MAX_DVYAW, self._prev_vel[2] + _MAX_DVYAW))
        self._prev_vel[:] = [vel_x, vel_y, vel_yaw]

        # ── Publish cmd_vel ───────────────────────────────────────────────────
        twist = Twist()
        twist.linear.x  = vel_x
        twist.linear.y  = vel_y
        twist.angular.z = vel_yaw
        self._last_twist = twist
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
            throttle_duration_sec=0.5,
        )

        # ── Append to recording buffer ────────────────────────────────────────
        with self._recording_lock:
            if self._recording_active:
                obs = self._latest_obs_for_record
                self._recording_buffer.append({
                    't':           time.time(),
                    'action':      action.copy(),
                    'vel_x':       np.float32(vel_x),
                    'vel_y':       np.float32(vel_y),
                    'vel_yaw':     np.float32(vel_yaw),
                    'gait':        np.int32(gait),
                    'dist_to_goal': np.float32(dist),
                    'pos_zc':      obs['position'][0].copy() if obs is not None else np.zeros(3, np.float32),
                    'goal_zc':     obs['goal'][0].copy() if obs is not None else np.zeros(2, np.float32),
                    'velocity_obs': obs['velocity'][0].copy() if obs is not None else np.zeros(6, np.float32),
                })


def main(args=None):
    rclpy.init(args=args)
    node = DreamerPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down …')
    finally:
        # Signal background threads to stop
        node._inference_running = False
        node._keyboard_running = False
        import time; time.sleep(0.3)
        node._save_recording('shutdown')
        node.destroy_node()
        rclpy.try_shutdown()   # no-op if already shut down by signal handler


if __name__ == '__main__':
    main()
