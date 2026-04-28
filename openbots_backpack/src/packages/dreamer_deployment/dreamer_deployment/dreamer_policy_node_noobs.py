#!/usr/bin/env python3
"""
Dreamer Policy Inference Node for SPOT Robot — NoObs (state-only) variant

For models trained on the `noobs-dataset` branch which use NO camera image or
terrain grid.  Observation space matches `spot.py` on that branch exactly:

  velocity     (3,)  float32  [vx, vy, wz]          — body frame (vz, wx, wy dropped)
  orientation  (2,)  float32  [cos(yaw), sin(yaw)]   — absolute (not episode-relative)
  goal         (2,)  float32  [goal_x - pos_x, goal_y - pos_y]  — world-relative

Subscribes to:
  /odometry   (nav_msgs/Odometry)     → position, orientation, velocity

Publishes:
  /cmd_vel    (geometry_msgs/Twist)   → velocity commands decoded from policy action

Calls service:
  /locomotion_mode                    → fixed HINT_TROT (trot-only model)

Subscribes to:
  /spot/policy/goal  (geometry_msgs/PointStamped) → set goal waypoint

Action space (3D in [-1, 1]):
  action[0] * 2.0  → cmd_vel.linear.x   (trot vx_max = 2.0 m/s)
  action[1] * 1.0  → cmd_vel.linear.y   (trot vy_max = 1.0 m/s)
  action[2] * 1.0  → cmd_vel.angular.z  (yaw_max = 1.0 rad/s)

To switch policies, only change `checkpoint_path` in params.yaml.
"""

import os
import sys
import time
import importlib
import select
import signal
import subprocess
from pathlib import Path

import threading
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

from spot_msgs.srv import SetLocomotion  # type: ignore

# Locomotion hint values (Boston Dynamics)
_HINT_CRAWL = 4   # action[3] <= 0 → slower crawl gait
_HINT_TROT  = 2   # action[3]  > 0 → faster trot gait


class DreamerPolicyNodeNoObs(Node):
    """
    DreamerV3 policy inference node for SPOT navigation — state-only (no camera/terrain).

    Observation / action spaces mirror the `noobs-dataset` training environment (spot.py)
    exactly so that the loaded checkpoint runs without modification.
    """

    def __init__(self):
        super().__init__('dreamer_policy_node_noobs')

        # ============ PARAMETERS ============
        self.declare_parameter(
            'checkpoint_path',
            '/home/maurits-heemskerk/Documents/Uni/Master_Thesis/'
            'dreamer_results_local_noobs/rep0.2/checkpoint.ckpt'
        )
        self.declare_parameter('control_rate_hz', 10)
        self.declare_parameter('robot_name', 'spot')
        self.declare_parameter('dreamerv3_root', '')

        # TEMPORARY DEBUG parameters
        self.declare_parameter('invert_goal_sign', False)
        self.declare_parameter('invert_action_x_sign', False)

        # DEPLOYMENT CONTROL: disable gait selection to use fixed locomotion mode
        self.declare_parameter('disable_gait_selection', False)
        self.declare_parameter('fixed_gait_mode', 'trot')  # 'trot' or 'crawl'

        # JAX platform and precision
        self.declare_parameter('jax_platform', 'cpu')
        self.declare_parameter('jax_precision', 'float32')

        # Policy inference rate — should match dataset recording frequency (~3.5 Hz).
        self.declare_parameter('policy_rate_hz', 3.5)

        # Whether to stop at the goal (default: True)
        self.declare_parameter('stop_at_goal', True)

        # Optional rosbag recording for deployment analysis
        self.declare_parameter('record_rosbag', False)
        self.declare_parameter('rosbag_output_dir', '')

        checkpoint_path  = self.get_parameter('checkpoint_path').value
        control_rate_hz  = self.get_parameter('control_rate_hz').value
        policy_rate_hz   = float(self.get_parameter('policy_rate_hz').value)
        self._dreamerv3_root = self.get_parameter('dreamerv3_root').value
        self._invert_goal_sign = self.get_parameter('invert_goal_sign').value
        self._invert_action_x_sign = self.get_parameter('invert_action_x_sign').value
        self._disable_gait_selection = self.get_parameter('disable_gait_selection').value
        self._fixed_gait_mode = self.get_parameter('fixed_gait_mode').value
        self._jax_platform = self.get_parameter('jax_platform').value
        self._jax_precision = self.get_parameter('jax_precision').value

        self._stop_at_goal = self.get_parameter('stop_at_goal').value
        self._record_rosbag = self.get_parameter('record_rosbag').value
        self._rosbag_output_dir = self.get_parameter('rosbag_output_dir').value

        # How many control ticks between RSSM policy calls.
        self._policy_stride = max(1, round(control_rate_hz / policy_rate_hz))
        self._tick_counter  = 0

        self.get_logger().info('=== Dreamer Policy Node (NoObs / state-only) ===')
        self.get_logger().info(f'  Checkpoint : {checkpoint_path}')
        self.get_logger().info(f'  Control Hz : {control_rate_hz}  (policy every {self._policy_stride} ticks ≈ {control_rate_hz/self._policy_stride:.1f} Hz)')
        self.get_logger().info(f'  Obs space  : [detected from checkpoint config at load time]')
        if self._dreamerv3_root:
            self.get_logger().info(f'  dreamerv3 root override : {self._dreamerv3_root}')
        if self._disable_gait_selection:
            self.get_logger().warn(f'  ⚠️  Gait selection DISABLED — using fixed mode: {self._fixed_gait_mode}')

        # ============ SENSOR STATE ============
        self._latest_odometry    = None   # nav_msgs/Odometry
        self._robot_position     = np.zeros(3, dtype=np.float32)  # x, y, z
        self._last_valid_observation = None

        # Goal in world frame; set via /spot/policy/goal topic
        self._goal_world         = None   # np.float32 (2,) — [x, y]
        self._is_first_step      = True
        self._step_counter       = 0
        self._goal_reached       = False

        # Episode origin for zero-centering
        self._episode_pos0       = None   # np.float32 (3,)
        self._episode_yaw0       = None   # float

        # ── Inference threading ───────────────────────────────────────────────
        self._inference_lock    = threading.Lock()
        self._inference_running = False
        self._last_twist        = None
        self._prev_vel          = np.zeros(3, dtype=np.float32)

        # ── Sensor status tracking ────────────────────────────────────────────
        self._sensor_ready = {'odometry': False}

        # ============ DREAMER AGENT ============
        self._agent      = None
        self._rssm_state = None
        self._latest_obs_for_record = None
        self._obs_keys   = []  # Initialize as empty list to avoid NoneType error; will be populated from checkpoint config
        self._recording_session_id = None
        self._rosbag_process = None
        self._rosbag_dir = None

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
        self.get_logger().info('Creating odometry subscription ...')
        self.create_subscription(
            Odometry,
            '/odometry',
            self._odometry_callback,
            rclpy.qos.qos_profile_sensor_data,
        )

        # ============ PUBLISHERS ============
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self._action_debug_pub = self.create_publisher(
            Float32MultiArray, '/spot/policy_action_debug', 1
        )

        # ============ SERVICE CLIENTS ============
        self._locomotion_client = self.create_client(SetLocomotion, '/locomotion_mode')

        # ============ SERVICE SERVERS ============
        self.create_service(Empty, '/dreamer_policy/reset', self._handle_reset_service)
        self.get_logger().info('Reset service available at /dreamer_policy/reset')

        # ============ GOAL TOPIC ============
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
        self.get_logger().info('Waiting for odometry ...')
        self._wait_for_sensors(timeout_sec=5.0)

        # ============ TERMINAL INPUT ============
        self._keyboard_running = True
        self._shutdown_event = threading.Event()
        self._stdin_thread = threading.Thread(target=self._stdin_handler, daemon=False)
        self._stdin_thread.start()

        self.get_logger().info('✓ Odometry ready. Policy node (NoObs) initialised.')

    # ──────────────────────────────────────────────────────────────────────────
    # AGENT LOADING
    # ──────────────────────────────────────────────────────────────────────────

    def _load_agent(self, checkpoint_path: str):
        """
        Instantiate the DreamerV3 JAX agent and restore weights from checkpoint.

        obs_space is auto-detected from the checkpoint config's encoder.mlp_keys,
        so this node supports any modality combination:
          - Standard noobs: velocity(5) + position(2) + orientation(2) + goal(2)
          - No-position ablation: velocity(5) + orientation(2) + goal(2)
          - Any future variant as long as it's defined in the config
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

            config_file = checkpoint_path.parent / 'config.yaml'
            if not config_file.exists():
                self.get_logger().error(f'config.yaml not found next to checkpoint: {config_file}')
                return

            raw_cfg = yaml.YAML(typ='safe').load(config_file.read_text())
            config = embodied.Config(dreamerv3.Agent.configs['defaults'])
            config = config.update(raw_cfg)

            # Extract observation space modalities from encoder config
            mlp_keys_str = config.encoder.get('mlp_keys', 'velocity|position|orientation|goal')
            self._obs_keys = mlp_keys_str.split('|')
            self.get_logger().info(f'  Obs keys from config: {self._obs_keys}')

            import jax
            platform = self._jax_platform
            try:
                jax.devices(platform)
            except Exception:
                self.get_logger().warn(f'Platform "{platform}" unavailable — falling back to cpu.')
                platform = 'cpu'
            config = config.update({
                'jax.platform': platform,
                'jax.precision': self._jax_precision,
                'jax.prealloc': False,
                'jax.jit': True,
                'batch_size': 1,
                'batch_length': 1,
                'imag_horizon': 1,
            })

            # ── Build observation space dynamically from config ────────────────────
            obs_space = {}
            for key in self._obs_keys:
                if key == 'velocity':
                    obs_space['velocity'] = embodied.Space(np.float32, (3,))
                elif key == 'position':
                    obs_space['position'] = embodied.Space(np.float32, (2,))
                elif key == 'orientation':
                    obs_space['orientation'] = embodied.Space(np.float32, (2,))
                elif key == 'goal':
                    obs_space['goal'] = embodied.Space(np.float32, (2,))
            # Always include reward, is_first, is_last, is_terminal
            obs_space.update({
                'reward':       embodied.Space(np.float32),
                'is_first':     embodied.Space(bool),
                'is_last':      embodied.Space(bool),
                'is_terminal':  embodied.Space(bool),
            })
            act_space = {
                'action': embodied.Space(np.float32, (3,), -1.0, 1.0),
                'reset':  embodied.Space(bool),
            }

            step = embodied.Counter()
            self._agent = dreamerv3.Agent(obs_space, act_space, step, config)

            self.get_logger().info('Setting up optax shim...')
            self._patch_optax_transforms()

            self.get_logger().info(f'Loading checkpoint from: {checkpoint_path}')
            checkpoint = embodied.Checkpoint()
            checkpoint.agent = self._agent
            checkpoint.load(str(checkpoint_path), keys=['agent'])

            self.get_logger().info(f'✓ Agent (NoObs) loaded from {checkpoint_path}')

            # ── Pre-trace mode='eval' (see dreamer_policy_node.py for full rationale) ──
            # Without this, the first real inference call compiles XLA for mode='eval',
            # which takes 30-60 s. During that gap no cmd_vel is published and SPOT
            # times out, leaving the robot unresponsive even after compilation.
            self.get_logger().info('Pre-tracing policy (mode=eval) — this may take 30-60 s …')
            dummy_obs = {
                'reward':    np.zeros((1,), dtype=np.float32),
                'is_first':  np.array([True]),
                'is_last':   np.array([False]),
                'is_terminal': np.array([False]),
            }
            _KEY_SHAPES = {
                'velocity':    (1, 3),
                'position':    (1, 2),
                'orientation': (1, 2),
                'goal':        (1, 2),
            }
            for k in self._obs_keys:
                if k in _KEY_SHAPES:
                    dummy_obs[k] = np.zeros(_KEY_SHAPES[k], dtype=np.float32)
            _, _ = self._agent.policy(dummy_obs, None, mode='eval')
            self.get_logger().info('✓ Policy trace complete — inference will be fast.')

        except Exception as e:
            self.get_logger().error(f'Failed to load agent: {e}')
            import traceback
            for line in traceback.format_exc().split('\n'):
                self.get_logger().error(line)

    def _patch_optax_transforms(self):
        """Install import hook that satisfies any 'optax.transforms.*' import."""
        import types
        import importlib.abc
        import importlib.machinery

        try:
            import optax
        except ImportError:
            self.get_logger().warn('optax not importable — skipping shim.')
            return

        if 'optax.transforms' in sys.modules:
            return

        class _OptaxTransformsFinder(importlib.abc.MetaPathFinder):
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
                mod.__getattr__ = lambda name: getattr(optax, name, None)
                sys.modules[spec.name] = mod
                return mod

            def exec_module(self, module):
                pass

        sys.meta_path.insert(0, _OptaxTransformsFinder())
        self.get_logger().info('Installed optax.transforms import hook.')

    def _ensure_dreamerv3_on_path(self, checkpoint_path: str):
        """Add plausible dreamerv3 source roots to sys.path."""
        if not self._dreamerv3_root:
            try:
                importlib.import_module('dreamerv3')
                return
            except Exception:
                pass

        candidates = []

        if self._dreamerv3_root:
            candidates.append(Path(self._dreamerv3_root))

        env_root = os.getenv('DREAMERV3_ROOT', '')
        if env_root:
            candidates.append(Path(env_root))

        candidates.extend([
            Path('/home/ob/openbots_ws/src/dreamer_SPOT_implementation/informed-dreamer'),
            Path('/home/ob/openbots_ws/src/packages/dreamer_SPOT_implementation/informed-dreamer'),
            Path('/workspaces/openbots_ws/src/dreamer_SPOT_implementation/informed-dreamer'),
            Path('/workspaces/openbots_ws/src/packages/dreamer_SPOT_implementation/informed-dreamer'),
        ])

        repo_root_guess = Path(__file__).resolve().parents[4]
        candidates.extend([
            repo_root_guess / 'dreamer_SPOT_implementation' / 'informed-dreamer',
            repo_root_guess / '..' / 'dreamer_SPOT_implementation' / 'informed-dreamer',
        ])

        ckpt = Path(checkpoint_path)
        candidates.append(ckpt.parent.parent / 'dreamer_SPOT_implementation' / 'informed-dreamer')

        tried = []
        for cand in candidates:
            cand = cand.expanduser().resolve()
            tried.append(str(cand))
            if (cand / 'dreamerv3').exists():
                if str(cand) not in sys.path:
                    sys.path.insert(0, str(cand))
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
        start_time = time.time()
        timeout = start_time + timeout_sec

        while time.time() < timeout:
            missing = {s for s in self._sensor_ready if not self._sensor_ready[s]}
            if not missing:
                self.get_logger().info('✓ Odometry connected')
                return
            elapsed = time.time() - start_time
            self.get_logger().info(
                f'  Waiting for sensors … ({elapsed:.1f}s) Missing: {", ".join(sorted(missing))}',
                throttle_duration_sec=2.0,
            )
            time.sleep(0.5)

        missing = ', '.join(s for s in self._sensor_ready if not self._sensor_ready[s])
        self.get_logger().error(
            f'Timeout waiting for sensors after {timeout_sec}s: {missing}\n'
            f'  • Is the SPOT driver running?\n'
            f'  • Check: ros2 topic echo /odometry'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # SENSOR CALLBACK
    # ──────────────────────────────────────────────────────────────────────────

    def _odometry_callback(self, msg: Odometry):
        """Store latest odometry and update robot position."""
        self._latest_odometry = msg
        p = msg.pose.pose.position
        self._robot_position = np.array([p.x, p.y, p.z], dtype=np.float32)
        if not self._sensor_ready['odometry']:
            self.get_logger().info('✓ Odometry subscription connected')
        self._sensor_ready['odometry'] = True

    # ──────────────────────────────────────────────────────────────────────────
    # GOAL TOPIC
    # ──────────────────────────────────────────────────────────────────────────

    def _goal_callback(self, msg: PointStamped):
        """
        Set the navigation goal.

        frame_id == 'body' (default): x=forward, y=left in robot body frame —
          rotated into world frame using current yaw.
        frame_id == 'odom': x, y treated directly as world-frame displacement.
        """
        if self._robot_position is None:
            self.get_logger().warn('No odometry yet — cannot set goal.')
            return

        dx_in, dy_in = msg.point.x, msg.point.y
        frame = msg.header.frame_id if msg.header.frame_id else 'body'

        odom = self._latest_odometry
        yaw = 0.0
        if odom is not None:
            q = odom.pose.pose.orientation
            yaw = float(np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)))

        if frame == 'odom':
            dx_world, dy_world = dx_in, dy_in
        else:
            dx_world = np.cos(yaw) * dx_in - np.sin(yaw) * dy_in
            dy_world = np.sin(yaw) * dx_in + np.cos(yaw) * dy_in

        abs_x = float(self._robot_position[0]) + dx_world
        abs_y = float(self._robot_position[1]) + dy_world

        self._goal_world = np.array([abs_x, abs_y], dtype=np.float32)
        self._goal_reached = False
        self._reset_episode_state(yaw, odom)

        # World-relative goal the model will see at step 0 (from zero-centred origin)
        dg = self._goal_world - self._episode_pos0[:2]
        cy0, sy0 = np.cos(self._episode_yaw0), np.sin(self._episode_yaw0)
        rel_x =  cy0 * dg[0] + sy0 * dg[1]
        rel_y = -sy0 * dg[0] + cy0 * dg[1]

        offset_label = 'offset_body' if frame != 'odom' else 'offset_world'
        pos_str = f'  pos=(0.00, 0.00)' if (self._obs_keys and 'position' in self._obs_keys) else '  [no position obs]'
        self.get_logger().info(
            f'🎯 Goal set [{frame}]: {offset_label}=({dx_in:.2f}, {dy_in:.2f})'
            f'  →  model sees:{pos_str}  goal=({rel_x:.2f}, {rel_y:.2f})'
            f'  dist={np.hypot(rel_x, rel_y):.2f}m  yaw0={np.degrees(yaw):.1f}°'
        )
        self._start_recording()

    # ──────────────────────────────────────────────────────────────────────────
    # KEYBOARD INPUT
    # ──────────────────────────────────────────────────────────────────────────

    def _stdin_handler(self):
        """Background thread: prompt for goal in body frame, then wait for reset."""
        while self._keyboard_running and not self._shutdown_event.is_set():
            try:
                print('\n──────────────────────────────────────────────────────')
                print('  🎯  Set goal  (body frame: x = forward m, y = left m)')
                print('──────────────────────────────────────────────────────')

                while True:
                    x_str = self._read_stdin_line('  X (forward, m): ')
                    if x_str is None:
                        return
                    try:
                        if not x_str:
                            continue
                        x = float(x_str)
                        break
                    except ValueError:
                        print('  ⚠  Not a number — try again.')

                while True:
                    y_str = self._read_stdin_line('  Y (lateral, m): ')
                    if y_str is None:
                        return
                    try:
                        if not y_str:
                            y_str = '0'
                        y = float(y_str)
                        break
                    except ValueError:
                        print('  ⚠  Not a number — try again.')

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
                self._goal_reached = False
                self._reset_episode_state(yaw, odom)

                dg = self._goal_world - self._episode_pos0[:2]
                cy0, sy0 = np.cos(self._episode_yaw0), np.sin(self._episode_yaw0)
                rel_x =  cy0 * dg[0] + sy0 * dg[1]
                rel_y = -sy0 * dg[0] + cy0 * dg[1]
                self.get_logger().info(
                    f'🎯 Goal set [terminal]: body=({x:.2f}, {y:.2f})'
                    f'  →  model sees: goal=({rel_x:.2f}, {rel_y:.2f})'
                    f'  dist={np.hypot(rel_x, rel_y):.2f}m  yaw0={np.degrees(yaw):.1f}°'
                )
                self._start_recording()

                if self._wait_for_enter('\n⌨️  Press Enter to set a new goal … '):
                    return
                self.get_logger().info('⟲ Reset — enter new goal.')
                odom2 = self._latest_odometry
                yaw2 = 0.0
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

    def _read_stdin_line(self, prompt: str) -> str | None:
        """Read one stdin line without blocking shutdown."""
        print(prompt, end='', flush=True)
        while not self._shutdown_event.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            except (OSError, ValueError):
                return None
            if not ready:
                continue
            line = sys.stdin.readline()
            if line == '':
                return None
            return line.strip()
        return None

    def _wait_for_enter(self, prompt: str) -> bool:
        """Wait for Enter while still allowing shutdown to interrupt cleanly."""
        print(prompt, flush=True)
        while not self._shutdown_event.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            except (OSError, ValueError):
                return True
            if not ready:
                continue
            line = sys.stdin.readline()
            if line == '':
                return True
            return False
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # RESET SERVICE & EPISODE STATE
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_reset_service(self, request, response):
        """Reset policy state without reloading the agent."""
        self.get_logger().info('⟲ Reset requested — clearing goal and RSSM state')
        yaw = 0.0
        if self._latest_odometry is not None:
            q = self._latest_odometry.pose.pose.orientation
            yaw = float(np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)))
        self._reset_episode_state(yaw, self._latest_odometry)
        return response

    def _reset_episode_state(self, yaw: float, odom):
        """
        Reset RSSM state and episode tracking.  Called when a new goal is set
        or the reset service is triggered.
        """
        self._save_recording('reset')
        self._is_first_step = True
        self._goal_reached  = False
        self._rssm_state    = None
        self._last_twist    = None
        self._prev_vel      = np.zeros(3, dtype=np.float32)
        self._tick_counter  = 0
        self._last_valid_observation = None  # clear stale cached observation

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
    # OBSERVATION BUILDING
    # ──────────────────────────────────────────────────────────────────────────

    def _build_observation(self) -> dict:
        """
        Assemble one observation dict with only the modalities defined by obs_keys.
        Auto-detected from checkpoint config, supporting:
          - Standard noobs: velocity(5) + position(2) + orientation(2) + goal(2)
          - No-position ablation: velocity(5) + orientation(2) + goal(2)

        All values have a leading batch dimension of 1 as expected by policy().
        Returns None during startup before any odometry has arrived.
        """
        odom = self._latest_odometry
        if odom is None:
            if self._last_valid_observation is not None:
                return self._last_valid_observation
            return None

        q = odom.pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        yaw = float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
        p = odom.pose.pose.position
        raw_pos = np.array([p.x, p.y, p.z], dtype=np.float32)

        obs = {'reward': np.zeros((1,), dtype=np.float32),
               'is_first': np.array([self._is_first_step]),
               'is_last': np.array([False]),
               'is_terminal': np.array([False])}

        # ── Add velocity if present in obs_keys ────────────────────────────────
        if 'velocity' in self._obs_keys:
            lv = odom.twist.twist.linear
            av = odom.twist.twist.angular
            # Rotate velocity into episode frame (initial yaw = 0)
            cy0 = np.cos(-self._episode_yaw0)
            sy0 = np.sin(-self._episode_yaw0)
            vx = cy0 * lv.x - sy0 * lv.y
            vy = sy0 * lv.x + cy0 * lv.y
            velocity = np.array([vx, vy, av.z], dtype=np.float32)
            obs['velocity'] = velocity[None]

        # ── Add position if present in obs_keys ────────────────────────────────
        if 'position' in self._obs_keys:
            if self._episode_pos0 is not None:
                dp = raw_pos - self._episode_pos0
                cy0 = np.cos(-self._episode_yaw0)
                sy0 = np.sin(-self._episode_yaw0)
                cx = cy0 * dp[0] - sy0 * dp[1]
                cy = sy0 * dp[0] + cy0 * dp[1]
                position = np.array([cx, cy], dtype=np.float32)
            else:
                position = raw_pos[:2].astype(np.float32)
            obs['position'] = position[None]

        # ── Add orientation if present in obs_keys ────────────────────────────
        if 'orientation' in self._obs_keys:
            # Episode-relative yaw (yaw - yaw0), matches training
            rel_yaw = yaw - self._episode_yaw0
            orientation = np.array([np.cos(rel_yaw), np.sin(rel_yaw)], dtype=np.float32)
            obs['orientation'] = orientation[None]

        # ── Add goal if present in obs_keys ──────────────────────────────────
        if 'goal' in self._obs_keys:
            if self._goal_world is not None and self._episode_pos0 is not None:
                # Express goal in episode frame (zero-centered, yaw-aligned)
                dg = self._goal_world - self._episode_pos0[:2]
                cy0 = np.cos(-self._episode_yaw0)
                sy0 = np.sin(-self._episode_yaw0)
                gx = cy0 * dg[0] - sy0 * dg[1]
                gy = sy0 * dg[0] + cy0 * dg[1]
                goal = np.array([gx, gy], dtype=np.float32)
                # Now subtract current position in episode frame
                dp = raw_pos - self._episode_pos0
                px = cy0 * dp[0] - sy0 * dp[1]
                py = sy0 * dp[0] + cy0 * dp[1]
                goal = goal - np.array([px, py], dtype=np.float32)
                if self._invert_goal_sign:
                    goal = -goal
            else:
                goal = np.zeros(2, dtype=np.float32)
            obs['goal'] = goal[None]

        self._last_valid_observation = obs
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # CONTROL LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def _control_loop(self):
        """Timer callback — dispatches inference to background thread."""
        if self._agent is None:
            return

        if self._goal_world is None:
            self.get_logger().warn(
                'No goal set — publishing zero velocity.',
                throttle_duration_sec=5.0,
            )
            self._cmd_vel_pub.publish(Twist())
            self._tick_counter = 0
            return

        self._tick_counter += 1

        if self._tick_counter % self._policy_stride != 0:
            self._cmd_vel_pub.publish(
                self._last_twist if self._last_twist is not None else Twist()
            )
            return

        if self._inference_running:
            self._cmd_vel_pub.publish(
                self._last_twist if self._last_twist is not None else Twist()
            )
            return

        self._inference_running = True
        t = threading.Thread(target=self._inference_thread, daemon=True)
        t.start()

    def _inference_thread(self):
        """Runs in background thread — never blocks the ROS executor."""
        t_step_start = time.perf_counter()
        try:
            # ── Goal-reached check ────────────────────────────────────────────
            dist = float(np.linalg.norm(self._goal_world - self._robot_position[:2]))
            if self._stop_at_goal and dist < 0.5:
                if not self._goal_reached:
                    self.get_logger().info(
                        f'Goal reached (dist={dist:.2f} m < 0.5 m) — holding position and continuing to record.',
                        throttle_duration_sec=2.0,
                    )
                    self._goal_reached = True
                self._cmd_vel_pub.publish(Twist())
                return

            obs = self._build_observation()
            if obs is None:
                self.get_logger().warn(
                    'Inference skipped: observation not available '
                    '(odometry not yet received). Check /odometry topic.',
                    throttle_duration_sec=5.0,
                )
                return

            if self._step_counter % 5 == 0:
                goal_r = obs['goal'][0] if 'goal' in obs else np.zeros(2)
                dist_g = float(np.linalg.norm(goal_r))
                log_parts = []
                if 'position' in obs:
                    pos_zc = obs['position'][0]
                    log_parts.append(f'pos=({pos_zc[0]:+.2f}, {pos_zc[1]:+.2f})')
                log_parts.append(f'goal=({goal_r[0]:+.2f}, {goal_r[1]:+.2f})  dist={dist_g:.2f}m')
                if 'orientation' in obs:
                    ori = obs['orientation'][0]
                    log_parts.append(f'ori=({ori[0]:+.2f}, {ori[1]:+.2f})')
                if 'velocity' in obs:
                    log_parts.append(f'vel=({obs["velocity"][0,0]:+.2f}, {obs["velocity"][0,1]:+.2f})')
                self.get_logger().info('[model] ' + '  '.join(log_parts))
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
        Decode 3D action vector to ROS commands (trot-only).

          vel_x   = action[0] * 2.0   (trot vx_max = 2.0 m/s)
          vel_y   = action[1] * 1.0   (trot vy_max = 1.0 m/s)
          vel_yaw = action[2] * 1.0   (yaw_max     = 1.0 rad/s)
        """
        gait = _HINT_TROT

        _MAX_LIN = 1.9
        _MAX_YAW = 1.5

        vel_x = action[0]
        if self._invert_action_x_sign:
            vel_x = -vel_x
        vel_x   = float(np.clip(vel_x * 2.0, -_MAX_LIN, _MAX_LIN))
        vel_y   = float(np.clip(action[1] * 1.0, -_MAX_LIN, _MAX_LIN))
        vel_yaw = float(np.clip(action[2] * 1.0, -_MAX_YAW, _MAX_YAW))

        # Rate limiting
        _MAX_DVLIN = 0.5
        _MAX_DVYAW = 1.0
        vel_x   = float(np.clip(vel_x,   self._prev_vel[0] - _MAX_DVLIN, self._prev_vel[0] + _MAX_DVLIN))
        vel_y   = float(np.clip(vel_y,   self._prev_vel[1] - _MAX_DVLIN, self._prev_vel[1] + _MAX_DVLIN))
        vel_yaw = float(np.clip(vel_yaw, self._prev_vel[2] - _MAX_DVYAW, self._prev_vel[2] + _MAX_DVYAW))
        self._prev_vel[:] = [vel_x, vel_y, vel_yaw]

        twist = Twist()
        twist.linear.x  = vel_x
        twist.linear.y  = vel_y
        twist.angular.z = vel_yaw
        self._last_twist = twist
        self._cmd_vel_pub.publish(twist)

        if self._locomotion_client.service_is_ready():
            req = SetLocomotion.Request()
            req.locomotion_mode = gait
            self._locomotion_client.call_async(req)

        dbg = Float32MultiArray()
        dbg.data = action.tolist()
        self._action_debug_pub.publish(dbg)

        dist = float(np.linalg.norm(self._goal_world - self._robot_position[:2])) \
            if self._goal_world is not None else float('nan')
        self.get_logger().info(
            f'[policy] raw=({action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f})'
            f'  →  vx={vel_x:+.2f} vy={vel_y:+.2f} vyaw={vel_yaw:+.2f}  '
            f'gait=TROT  dist_to_goal={dist:.2f}m',
            throttle_duration_sec=0.5,
        )

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
                    'goal_world':  self._goal_world.copy() if self._goal_world is not None else np.zeros(2, np.float32),
                    'pos_zc':      obs['position'][0].copy() if (obs is not None and 'position' in obs) else np.zeros(2, np.float32),
                    'goal_zc':     obs['goal'][0].copy() if obs is not None else np.zeros(2, np.float32),
                    'velocity_obs': obs['velocity'][0].copy() if obs is not None else np.zeros(3, np.float32),
                })

    # ──────────────────────────────────────────────────────────────────────────
    # COMMAND RECORDING
    # ──────────────────────────────────────────────────────────────────────────

    def _start_recording(self):
        self._recording_session_id = time.strftime('%Y%m%d_%H%M%S')
        self._start_rosbag_recording(self._recording_session_id)
        with self._recording_lock:
            self._recording_buffer = []
            self._recording_active = True
        self.get_logger().info(f'Recording started → {self._recordings_dir}')

    def _start_rosbag_recording(self, session_id: str):
        if not self._record_rosbag:
            return

        if self._rosbag_process is not None and self._rosbag_process.poll() is None:
            return

        ckpt_dir = Path(self.get_parameter('checkpoint_path').value).parent
        if self._rosbag_output_dir:
            base_dir = Path(self._rosbag_output_dir).expanduser()
        else:
            base_dir = ckpt_dir / 'rosbag'
        bag_dir = base_dir / f'episode_{session_id}'
        base_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            'ros2', 'bag', 'record',
            '-o', str(bag_dir),
            '/cmd_vel',
            '/spot/policy/goal',
            '/odometry',
            '/spot/policy_action_debug',
        ]

        try:
            self._rosbag_process = subprocess.Popen(
                cmd,
                cwd=str(base_dir),
                start_new_session=True,
            )
            self._rosbag_dir = bag_dir
            self.get_logger().info(f'Rosbag started → {bag_dir}')
        except Exception as e:
            self._rosbag_process = None
            self.get_logger().error(f'Failed to start rosbag recorder: {e}')

    def _stop_rosbag_recording(self, reason: str):
        proc = self._rosbag_process
        if proc is None:
            return

        self._rosbag_process = None
        if proc.poll() is not None:
            return

        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=10.0)
            self.get_logger().info(f'Rosbag stopped ({reason}) → {self._rosbag_dir}')
        except Exception as e:
            self.get_logger().warn(f'Rosbag stop timed out or failed ({reason}): {e}')
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _save_recording(self, reason: str):
        with self._recording_lock:
            if not self._recording_active:
                return
            self._recording_active = False
            buf = self._recording_buffer
            self._recording_buffer = []

        if not buf:
            self._stop_rosbag_recording(reason)
            return

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        session_id = self._recording_session_id or timestamp
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
                goal_world   = np.array([s['goal_world']   for s in buf], dtype=np.float32),
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
            self._stop_rosbag_recording(reason)
            self._recording_session_id = None


def main(args=None):
    rclpy.init(args=args)
    node = DreamerPolicyNodeNoObs()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down …')
    finally:
        node._inference_running = False
        node._keyboard_running = False
        node._shutdown_event.set()
        if hasattr(node, '_stdin_thread') and node._stdin_thread.is_alive():
            node._stdin_thread.join(timeout=2.0)
        import time; time.sleep(0.3)
        node._save_recording('shutdown')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
