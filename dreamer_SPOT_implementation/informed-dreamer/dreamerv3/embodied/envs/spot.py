import embodied
import h5py
import numpy as np
from pathlib import Path



class SPOT(embodied.Env):
  def __init__(self, task, data_dir, episode_idx=0, **kwargs):
    self._task = task
    self._data_dir = Path(data_dir)
    self._episode_idx = episode_idx
    self._episode_files = sorted(self._data_dir.glob('**/*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
    assert self._episode_files, f'No .h5 episodes found in {self._data_dir}'
    self._load_episode()
    self._step = 0
    self._done = False

  def _load_episode(self):
    ep_file = self._episode_files[self._episode_idx % len(self._episode_files)]
    with h5py.File(ep_file, 'r') as f:
      self._velocities = f['observations/velocities'][:]
      self._states = f['observations/state'][:]
      self._actions = f['actions'][:]
      self._precomputed_rewards = f['rewards'][:].astype(np.float32)
      # Use the exact goal that was used when computing stored rewards (HER-relabelled).
      # Falling back to the last position is only correct for end_position variants;
      # for never_reach / overshoot / along_trajectory variants the goal differs and
      # using the wrong value makes reward and goal-obs contradict each other.
      if 'goal_position_x' in f.attrs and 'goal_position_y' in f.attrs:
        self._goal_position = np.array(
            [f.attrs['goal_position_x'], f.attrs['goal_position_y']],
            dtype=np.float32)
      else:
        self._goal_position = self._states[-1, :2].astype(np.float32)
    self._length = len(self._velocities)
    self._positions = self._states[:self._length, :2].astype(np.float32)

  @property
  def obs_space(self):
    return {
        # velocity: [vx, vy, wz] — drop vz (idx 2) and pitch/roll rates wx,wy (idx 3,4)
        'velocity': embodied.Space(np.float32, (3,)),
        # position: [x, y] — drop z (idx 2), irrelevant on flat ground
        'position': embodied.Space(np.float32, (2,)),
        # orientation: [cos(yaw), sin(yaw)] — continuous, no ±π wrap
        'orientation': embodied.Space(np.float32, (2,)),
        'goal': embodied.Space(np.float32, (2,)),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.float32, (3,), -1.0, 1.0),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action.get('reset', False) or self._done:
      self._episode_idx = (self._episode_idx + 1) % len(self._episode_files)
      self._load_episode()
      self._step = 0
      self._done = False
      return self._obs(is_first=True)

    self._action(action)
    self._step += 1
    is_last = (self._step >= self._length - 1)
    obs = self._obs(is_last=is_last, is_terminal=is_last)
    if is_last:
      self._done = True
    return obs

  def _action(self, action):
    raw_action = action['action']
    
    # Direct action mapping: vx, vy, yaw_rate (no gait switching)
    vel_x = float(raw_action[0] * 2.0)
    vel_y = float(raw_action[1] * 1.0)  # SPOT caps vy at half of vx max
    vel_yaw = float(raw_action[2] * 1.0)
    
    return action

  def _obs(self, is_first=False, is_last=False, is_terminal=False):
    t = self._step
    # velocity: keep only [vx, vy, wz] — drop vz (idx 2), wx/wy pitch+roll rates (idx 3,4)
    velocity = self._velocities[t, [0, 1, 5]].astype(np.float32)
    # position: x, y only
    position = self._states[t, :2].astype(np.float32)
    # orientation: cos/sin of yaw — continuous representation, no ±π wrap
    qx, qy, qz, qw = self._states[t, 3:7]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    orientation = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)

    goal_rel = np.array([
        self._goal_position[0] - position[0],
        self._goal_position[1] - position[1],
    ], dtype=np.float32)

    reward = float(self._precomputed_rewards[t]) if hasattr(self, '_precomputed_rewards') and t < len(self._precomputed_rewards) else 0.0

    return {
      'velocity': velocity,
      'position': position,
      'orientation': orientation,
      'goal': goal_rel,
      'reward': reward,
      'is_first': is_first,
      'is_last': is_last,
      'is_terminal': is_terminal,
    }

  def render(self):
    return np.zeros((64, 64, 3), dtype=np.uint8)

  def close(self):
    pass

  @staticmethod
  def compute_physics_loss(pos_pred, vel_pred, goal_pred=None, dt=0.3):
    """
    Compute physics consistency loss for Spot robot.

    Args:
      pos_pred:  (B, T, 2) position decoder means in symlog space, world frame [x, y]
      vel_pred:  (B, T, 3) velocity decoder means in symlog space, body frame [vx,vy,wz]
      goal_pred: (B, T, 2) goal decoder means in symlog space, world frame (optional)
      dt: timestep duration in seconds

    Returns:
      physics_loss: scalar loss enforcing kinematic consistency
    """
    import jax.numpy as jnp

    inv_symlog = lambda x: jnp.sign(x) * (jnp.exp(jnp.clip(jnp.abs(x), 0.0, 10.0)) - 1)

    pos_real = inv_symlog(pos_pred)   # (B, T, 2) world frame, metres
    vel_real = inv_symlog(vel_pred)   # (B, T, 5) body frame, m/s

    # --- Position-speed constraint (magnitude, rotation-invariant) ---
    pos_speed = jnp.linalg.norm(pos_real[:, 1:, :2] - pos_real[:, :-1, :2], axis=-1) / dt  # (B, T-1)
    vel_speed = jnp.linalg.norm(vel_real[:, :-1, :2], axis=-1)                               # (B, T-1)
    speed_scale = jnp.maximum(jnp.maximum(pos_speed, vel_speed), 1.0)
    physics_pos_loss = jnp.mean(((pos_speed - vel_speed) / speed_scale) ** 2)

    if goal_pred is not None:
      goal_real = inv_symlog(goal_pred)  # (B, T, 2) ego-centric, body frame, metres

      # --- Goal-velocity constraint (vector, body frame) ---
      # goal_body(t+1) ≈ goal_body(t) - vel_body_xy(t) * dt
      goal_derived = goal_real[:, :-1] - vel_real[:, :-1, :2] * dt  # (B, T-1, 2)
      goal_actual  = goal_real[:, 1:]                                 # (B, T-1, 2)
      goal_err = goal_derived - goal_actual
      # Normalize by the larger of the two goal magnitudes (min 1 m) so loss ∈ [0, 1]
      goal_scale = jnp.maximum(
          jnp.linalg.norm(goal_derived, axis=-1, keepdims=True) +
          jnp.linalg.norm(goal_actual,  axis=-1, keepdims=True),
          1.0)
      physics_goal_loss = jnp.mean((goal_err / goal_scale) ** 2)

      physics_loss = physics_pos_loss + physics_goal_loss
    else:
      physics_loss = physics_pos_loss

    # Final safety net — any residual inf/nan (e.g. from degenerate batches) → 0
    return jnp.nan_to_num(physics_loss, nan=0.0, posinf=0.0, neginf=0.0)

