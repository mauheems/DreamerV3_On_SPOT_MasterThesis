"""SPOTLive: Live ROS2 environment for DreamerV3 online finetuning.

rclpy / CycloneDDS run in a completely separate *spawned* subprocess so that
DDS background C-threads never share a process with JAX.  The two sides
communicate via multiprocessing.Queue with plain numpy/dict payloads.

Observation / action spaces are identical to the offline SPOT env.
"""

import multiprocessing as mp
import threading
import time

import numpy as np

import embodied


# ── Subprocess worker (module-level so spawn pickling works) ─────────────────

def _ros2_subprocess_worker(obs_queue, action_queue, goal_queue, ready_event):
    """Runs in a freshly spawned Python process — no JAX ever loaded here."""
    import os

    # Inherit CYCLONEDDS_URI / RMW from parent environment (set before spawn).
    # Pin to WiFi interface to avoid VPN/docker-bridge multicast crashes.
    os.environ.setdefault(
        'CYCLONEDDS_URI',
        '<CycloneDDS><Domain><General>'
        '<NetworkInterfaceAddress>wlp0s20f3</NetworkInterfaceAddress>'
        '</General></Domain></CycloneDDS>'
    )

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist, PointStamped
    import numpy as np

    rclpy.init()
    node = rclpy.create_node('spot_live_env')

    _latest = {'position': np.zeros(3, np.float32), 'yaw': 0.0}

    def odometry_callback(msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular
        yaw = float(np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)))
        pos = np.array([p.x, p.y, p.z], np.float32)
        _latest['position'] = pos
        _latest['yaw'] = yaw
        data = {
            'position':    pos,
            'quaternion':  np.array([q.x, q.y, q.z, q.w], np.float32),
            'yaw':         yaw,
            'linear_vel':  np.array([lv.x, lv.y, lv.z], np.float32),
            'angular_vel': np.array([av.x, av.y, av.z], np.float32),
        }
        try:
            obs_queue.put_nowait(data)
        except Exception:
            pass  # queue full — stale obs dropped

    def goal_callback(msg):
        data = {
            'frame_id':       msg.header.frame_id or 'body',
            'dx':             float(msg.point.x),
            'dy':             float(msg.point.y),
            'robot_position': _latest['position'].copy(),
            'yaw':            _latest['yaw'],
        }
        goal_queue.put(data)

    cmd_pub = node.create_publisher(Twist, '/cmd_vel', 1)
    node.create_subscription(Odometry, '/odometry', odometry_callback,
                             rclpy.qos.qos_profile_sensor_data)
    node.create_subscription(PointStamped, '/spot/policy/goal', goal_callback, 10)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    ready_event.set()  # signal parent: node is up

    while rclpy.ok():
        executor.spin_once(timeout_sec=0.005)
        # Drain action queue and publish any pending command
        try:
            action = action_queue.get_nowait()
            if action is None:
                break  # shutdown signal
            twist = Twist()
            twist.linear.x  = float(action['vx'])
            twist.linear.y  = float(action['vy'])
            twist.angular.z = float(action['vyaw'])
            cmd_pub.publish(twist)
        except Exception:
            pass

    try:
        cmd_pub.publish(Twist())   # zero velocity on exit
        node.destroy_node()
    except Exception:
        pass
    rclpy.shutdown()


# ── Main-process environment class ───────────────────────────────────────────

class SPOTLive(embodied.Env):
    """Live SPOT environment for online DreamerV3 finetuning via ROS2."""

    def __init__(
        self,
        task,
        obs_keys='velocity|position|orientation|goal',
        policy_rate_hz=3.5,
        goal_schedule=None,
        goal_timeout=300.0,
        max_episode_steps=200,
        reach_threshold=0.5,
        **kwargs,
    ):
        self._task = task
        self._obs_keys = obs_keys.split('|') if isinstance(obs_keys, str) else list(obs_keys)
        self._policy_rate_hz = float(policy_rate_hz)
        self._policy_period = 1.0 / self._policy_rate_hz
        self._goal_timeout = float(goal_timeout)
        self._max_episode_steps = int(max_episode_steps)
        self._reach_threshold = float(reach_threshold)

        if goal_schedule is not None:
            self._goal_schedule = [np.array(g, dtype=np.float32) for g in goal_schedule]
        else:
            self._goal_schedule = []
        self._goal_schedule_idx = 0

        # Episode state
        self._episode_pos0 = np.zeros(3, np.float32)
        self._episode_yaw0 = 0.0
        self._goal_world = None
        self._prev_dist = None
        self._ep_step = 0
        self._done = True
        self._is_first = True

        # Sensor state (kept up-to-date by reader thread)
        self._latest_obs_data = None
        self._robot_position = np.zeros(3, np.float32)
        self._last_raw_yaw = 0.0
        self._new_obs_event = threading.Event()
        self._goal_event = threading.Event()
        self._received_goal_data = None
        self._ros_running = True

        # ── Spawn the ROS2 subprocess ─────────────────────────────────────
        ctx = mp.get_context('fork')
        self._obs_queue    = ctx.Queue(maxsize=100)
        self._action_queue = ctx.Queue(maxsize=2)
        self._goal_queue   = ctx.Queue(maxsize=10)
        self._ready_event  = ctx.Event()

        self._ros_proc = ctx.Process(
            target=_ros2_subprocess_worker,
            args=(self._obs_queue, self._action_queue,
                  self._goal_queue, self._ready_event),
            daemon=True,
        )
        self._ros_proc.start()

        # Lightweight reader threads: drain queues, update local state
        self._obs_thread = threading.Thread(target=self._obs_reader_loop, daemon=True)
        self._goal_thread = threading.Thread(target=self._goal_reader_loop, daemon=True)
        self._obs_thread.start()
        self._goal_thread.start()

        print('[SPOTLive] Starting ROS2 subprocess …', flush=True)
        if not self._ready_event.wait(timeout=30.0):
            raise RuntimeError('[SPOTLive] ROS2 subprocess did not start within 30 s')

        print('[SPOTLive] Waiting for /odometry …', flush=True)
        deadline = time.time() + 30.0
        while self._latest_obs_data is None and time.time() < deadline:
            time.sleep(0.1)
        if self._latest_obs_data is None:
            raise RuntimeError('[SPOTLive] No /odometry received within 30 s — '                                'is the SPOT driver running?')
        print('[SPOTLive] /odometry OK', flush=True)

    # ── Reader threads (main process, no DDS/JAX) ────────────────────────────

    def _obs_reader_loop(self):
        while self._ros_running:
            try:
                data = self._obs_queue.get(timeout=0.1)
                self._latest_obs_data = data
                self._robot_position  = data['position']
                self._last_raw_yaw    = data['yaw']
                self._new_obs_event.set()
            except Exception:
                pass

    def _goal_reader_loop(self):
        while self._ros_running:
            try:
                data = self._goal_queue.get(timeout=0.1)
                self._received_goal_data = data
                self._goal_event.set()
            except Exception:
                pass

    # ── embodied.Env interface ───────────────────────────────────────────────

    @property
    def obs_space(self):
        shapes = {'velocity': (3,), 'position': (2,), 'orientation': (2,), 'goal': (2,)}
        spaces = {k: embodied.Space(np.float32, shapes[k])
                  for k in self._obs_keys if k in shapes}
        spaces.update({
            'reward':      embodied.Space(np.float32),
            'is_first':    embodied.Space(bool),
            'is_last':     embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        })
        return spaces

    @property
    def act_space(self):
        return {
            'action': embodied.Space(np.float32, (4,), -1.0, 1.0),
            'reset':  embodied.Space(bool),
        }

    def __len__(self):
        return 1

    def step(self, action):
        reset = bool(np.asarray(action.get('reset', False)).flat[0])
        if reset or self._done:
            return self._episode_reset()

        act = np.asarray(action['action']).reshape(-1)
        self._publish_action(act)

        # Wait for next odometry (up to 3× policy period)
        self._new_obs_event.clear()
        got_obs = self._new_obs_event.wait(timeout=self._policy_period * 3.0)
        if not got_obs:
            print('[SPOTLive] WARNING: odometry timeout — using stale obs', flush=True)

        obs = self._build_obs()
        dist = float(np.linalg.norm(self._goal_world - self._robot_position[:2]))
        reward = float(self._prev_dist - dist) if self._prev_dist is not None else 0.0
        self._prev_dist = dist
        self._ep_step += 1

        reached = dist < self._reach_threshold
        timeout  = self._ep_step >= self._max_episode_steps
        is_last  = reached or timeout

        if reached:
            reward += 5.0
            print(f'[SPOTLive] Goal reached (dist={dist:.2f} m)', flush=True)
        if timeout:
            print(f'[SPOTLive] Episode timeout ({self._ep_step} steps)', flush=True)
        if is_last:
            self._done = True
            self._publish_stop()

        return self._to_batch({
            **obs,
            'reward':      np.float32(reward),
            'is_first':    np.bool_(False),
            'is_last':     np.bool_(is_last),
            'is_terminal': np.bool_(is_last),
        })

    def render(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        self._ros_running = False
        try:
            self._action_queue.put_nowait(None)  # shutdown signal
        except Exception:
            pass
        if self._ros_proc.is_alive():
            self._ros_proc.join(timeout=3.0)
            if self._ros_proc.is_alive():
                self._ros_proc.terminate()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _episode_reset(self):
        self._publish_stop()
        time.sleep(0.3)

        yaw = self._last_raw_yaw
        if self._latest_obs_data is not None:
            self._episode_pos0 = self._latest_obs_data['position'].copy()
        else:
            self._episode_pos0 = self._robot_position.copy()
        self._episode_yaw0 = yaw

        if self._goal_schedule:
            body_goal = self._goal_schedule[self._goal_schedule_idx % len(self._goal_schedule)]
            self._goal_schedule_idx += 1
            dx_w = np.cos(yaw)*body_goal[0] - np.sin(yaw)*body_goal[1]
            dy_w = np.sin(yaw)*body_goal[0] + np.cos(yaw)*body_goal[1]
            self._goal_world = np.array([
                self._episode_pos0[0] + dx_w,
                self._episode_pos0[1] + dy_w,
            ], np.float32)
            rel = self._goal_world - self._episode_pos0[:2]
            print(f'[SPOTLive] Goal from schedule: body=({body_goal[0]:.2f},{body_goal[1]:.2f}) '                   f'world=({self._goal_world[0]:.2f},{self._goal_world[1]:.2f}) '                   f'model=({rel[0]:.2f},{rel[1]:.2f})', flush=True)
        else:
            print(f'[SPOTLive] Waiting for goal on /spot/policy/goal '                   f'(timeout={self._goal_timeout:.0f}s) …', flush=True)
            self._goal_event.clear()
            self._received_goal_data = None
            got_goal = self._goal_event.wait(timeout=self._goal_timeout)
            if not got_goal or self._received_goal_data is None:
                raise RuntimeError(
                    f'[SPOTLive] No goal received within {self._goal_timeout}s.'
                )
            d = self._received_goal_data
            dx_in, dy_in = d['dx'], d['dy']
            goal_yaw = d['yaw']
            if d['frame_id'] == 'odom':
                dx_w, dy_w = dx_in, dy_in
            else:
                dx_w = np.cos(goal_yaw)*dx_in - np.sin(goal_yaw)*dy_in
                dy_w = np.sin(goal_yaw)*dx_in + np.cos(goal_yaw)*dy_in
            self._goal_world = np.array([
                d['robot_position'][0] + dx_w,
                d['robot_position'][1] + dy_w,
            ], np.float32)
            rel = self._goal_world - self._episode_pos0[:2]
            print(f'[SPOTLive] Goal set: world=({self._goal_world[0]:.2f},{self._goal_world[1]:.2f}) '                   f'model=({rel[0]:.2f},{rel[1]:.2f})', flush=True)

        self._prev_dist = float(np.linalg.norm(self._goal_world - self._robot_position[:2]))
        self._ep_step = 0
        self._done = False
        self._is_first = True

        return self._to_batch({
            **self._build_obs(),
            'reward':      np.float32(0.0),
            'is_first':    np.bool_(True),
            'is_last':     np.bool_(False),
            'is_terminal': np.bool_(False),
        })

    def _build_obs(self):
        data = self._latest_obs_data
        obs = {}
        shapes = {'velocity': (3,), 'position': (2,), 'orientation': (2,), 'goal': (2,)}
        if data is None:
            return {k: np.zeros(shapes[k], np.float32)
                    for k in self._obs_keys if k in shapes}

        yaw     = data['yaw']
        raw_pos = data['position']
        lv      = data['linear_vel']
        av      = data['angular_vel']

        if 'velocity' in self._obs_keys:
            vbx = np.cos(yaw)*lv[0] + np.sin(yaw)*lv[1]
            vby = -np.sin(yaw)*lv[0] + np.cos(yaw)*lv[1]
            obs['velocity'] = np.array([vbx, vby, av[2]], np.float32)

        if 'position' in self._obs_keys:
            dp = raw_pos - self._episode_pos0
            cy0, sy0 = np.cos(-self._episode_yaw0), np.sin(-self._episode_yaw0)
            obs['position'] = np.array([cy0*dp[0]-sy0*dp[1], sy0*dp[0]+cy0*dp[1]], np.float32)

        if 'orientation' in self._obs_keys:
            rel_yaw = yaw - self._episode_yaw0
            obs['orientation'] = np.array([np.cos(rel_yaw), np.sin(rel_yaw)], np.float32)

        if 'goal' in self._obs_keys:
            if self._goal_world is not None:
                obs['goal'] = (self._goal_world - raw_pos[:2]).astype(np.float32)
            else:
                obs['goal'] = np.zeros(2, np.float32)

        return obs

    def _publish_action(self, action: np.ndarray):
        gait_trot = float(action[3]) > 0
        max_vel   = 2.0 if gait_trot else 1.0
        _MAX_LIN, _MAX_YAW = 1.9, 1.5
        cmd = {
            'vx':   float(np.clip(action[0] * max_vel,       -_MAX_LIN, _MAX_LIN)),
            'vy':   float(np.clip(action[1] * (max_vel/2),   -_MAX_LIN, _MAX_LIN)),
            'vyaw': float(np.clip(action[2] * 1.0,           -_MAX_YAW, _MAX_YAW)),
        }
        try:
            self._action_queue.put_nowait(cmd)
        except Exception:
            pass  # queue full (robot not keeping up) — skip frame

    def _publish_stop(self):
        try:
            self._action_queue.put_nowait({'vx': 0.0, 'vy': 0.0, 'vyaw': 0.0})
        except Exception:
            pass

    def _to_batch(self, obs):
        return {k: np.asarray(v)[None] for k, v in obs.items()}
