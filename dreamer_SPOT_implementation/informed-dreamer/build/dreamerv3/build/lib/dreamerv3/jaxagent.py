import os

import embodied
import jax
import jax.numpy as jnp
import numpy as np

from . import jaxutils
from . import ninjax as nj

tree_map = jax.tree_util.tree_map
tree_flatten = jax.tree_util.tree_flatten


def Wrapper(agent_cls):
  class Agent(JAXAgent):
    configs = agent_cls.configs
    inner = agent_cls
    def __init__(self, *args, **kwargs):
      super().__init__(agent_cls, *args, **kwargs)
  return Agent


class JAXAgent(embodied.Agent):

  def __init__(self, agent_cls, obs_space, act_space, step, config):
    self.config = config.jax
    self.batch_size = config.batch_size
    self.batch_length = config.batch_length
    self.data_loaders = config.data_loaders
    self._setup()
    self.agent = agent_cls(obs_space, act_space, step, config, name='agent')
    self.rng = np.random.default_rng(config.seed)

    available = jax.devices(self.config.platform)
    self.policy_devices = [available[i] for i in self.config.policy_devices]
    self.train_devices = [available[i] for i in self.config.train_devices]
    self.single_device = (self.policy_devices == self.train_devices) and (
        len(self.policy_devices) == 1)
    print(f'JAX devices ({jax.local_device_count()}):', available)
    print('Policy devices:', ', '.join([str(x) for x in self.policy_devices]))
    print('Train devices: ', ', '.join([str(x) for x in self.train_devices]))

    self._once = True
    self._updates = embodied.Counter()
    self._should_metrics = embodied.when.Every(self.config.metrics_every)
    self._transform()
    self.varibs = self._init_varibs(obs_space, act_space)
    self.sync()

  def policy(self, obs, state=None, mode='train'):
    obs = obs.copy()
    obs = self._convert_inps(obs, self.policy_devices)
    rng = self._next_rngs(self.policy_devices)
    varibs = self.varibs if self.single_device else self.policy_varibs
    if state is None:
      state, _ = self._init_policy(varibs, rng, obs['is_first'])
    else:
      state = tree_map(
          np.asarray, state, is_leaf=lambda x: isinstance(x, list))
      state = self._convert_inps(state, self.policy_devices)
    (outs, state), _ = self._policy(varibs, rng, obs, state, mode=mode)
    outs = self._convert_outs(outs, self.policy_devices)
    # TODO: Consider keeping policy states in accelerator memory.
    state = self._convert_outs(state, self.policy_devices)
    return outs, state

  def train(self, data, state=None):
    rng = self._next_rngs(self.train_devices)
    if state is None:
      state, self.varibs = self._init_train(self.varibs, rng, data['is_first'])
    (outs, state, mets), self.varibs = self._train(
        self.varibs, rng, data, state)
    outs = self._convert_outs(outs, self.train_devices)
    self._updates.increment()
    if self._should_metrics(self._updates):
      mets = self._convert_mets(mets, self.train_devices)
    else:
      mets = {}
    if self._once:
      self._once = False
      assert jaxutils.Optimizer.PARAM_COUNTS
      for name, count in jaxutils.Optimizer.PARAM_COUNTS.items():
        mets[f'params_{name}'] = float(count)
    return outs, state, mets

  def report(self, data):
    rng = self._next_rngs(self.train_devices)
    mets, _ = self._report(self.varibs, rng, data)
    mets = self._convert_mets(mets, self.train_devices)
    return mets

  def dataset(self, generator):
    batcher = embodied.Batcher(
        sources=[generator] * self.batch_size,
        workers=self.data_loaders,
        postprocess=lambda x: self._convert_inps(x, self.train_devices),
        prefetch_source=4, prefetch_batch=1)
    return batcher()

  def save(self):
    if len(self.train_devices) > 1:
      varibs = tree_map(lambda x: x[0], self.varibs)
    else:
      varibs = self.varibs
    varibs = jax.device_get(varibs)
    data = tree_map(np.asarray, varibs)
    return data

  def load(self, state):
    # Aggressive filtering to only keep inference-needed weights on constrained GPUs.
    # Skip any keys containing training/optimization patterns.
    skip_patterns = {'opt', 'critic', 'reward', 'value', 'slowreg', 'dynamics', 'disag', 'expl'}
    
    original_size_mb = sum(
        v.size * v.itemsize / (1024**2)
        for v in jax.tree_util.tree_leaves(state)
        if hasattr(v, 'size') and hasattr(v, 'itemsize')
    )
    print(f'[Info] Total checkpoint: {original_size_mb:.0f}MB')
    
    # Strategy 1: Use _policy_keys if available (from traced init)
    filtered_state = {}
    if hasattr(self, '_policy_keys') and self._policy_keys:
      filtered_state = {k: v for k, v in state.items() if k in self._policy_keys}
    
    # Strategy 2: If Strategy 1 gave us nothing or very little, use pattern-based filter
    if not filtered_state or sum(
        v.size * v.itemsize / (1024**2)
        for v in jax.tree_util.tree_leaves(filtered_state)
        if hasattr(v, 'size') and hasattr(v, 'itemsize')
    ) < 10:  # Less than 10MB is too small
      print('[Info] Policy keys filter insufficient. Using pattern-based filter.')
      filtered_state = {
          k: v for k, v in state.items()
          if not any(pattern in str(k).lower() for pattern in skip_patterns)
      }
    
    filtered_size_mb = sum(
        v.size * v.itemsize / (1024**2)
        for v in jax.tree_util.tree_leaves(filtered_state)
        if hasattr(v, 'size') and hasattr(v, 'itemsize')
    )
    print(f'[Info] After filtering: {len(filtered_state)}/{len(state)} keys, {filtered_size_mb:.0f}MB')
    
    state = filtered_state if filtered_state else state
    
    # Cast checkpoint arrays to target precision (float32 or float16) BEFORE GPU upload
    # This saves memory if checkpoint was trained in float32 but inference runs in float16
    def cast_to_precision(v):
      if hasattr(v, 'dtype') and jnp.issubdtype(v.dtype, jnp.floating):
        return v.astype(jaxutils.COMPUTE_DTYPE)
      return v
    
    state = jax.tree_util.tree_map(cast_to_precision, state)
    post_cast_size_mb = sum(
        v.size * v.itemsize / (1024**2)
        for v in jax.tree_util.tree_leaves(state)
        if hasattr(v, 'size') and hasattr(v, 'itemsize')
    )
    if post_cast_size_mb != filtered_size_mb:
      print(f'[Info] After precision cast: {post_cast_size_mb:.0f}MB (was {filtered_size_mb:.0f}MB)')
    
    if len(self.train_devices) == 1:
      device = self.train_devices[0]
      # Load piece-by-piece to minimize peak memory
      # Do NOT reset self.varibs to {} here — keep init varibs so keys missing
      # from a partial checkpoint (e.g. from_checkpoint with subset of keys)
      # retain their randomly-initialized values from _init_varibs().
      for i, (key, value) in enumerate(state.items()):
        try:
          self.varibs[key] = jax.device_put(value, device)
          if (i + 1) % max(1, len(state) // 5) == 0:
            print(f'  [{i+1}/{len(state)}] Checkpoint loading progress')
        except Exception as e:
          print(f'[Error] Failed to load {key}: {e}')
          raise
    else:
      self.varibs = jax.device_put_replicated(state, self.train_devices)
    self.sync()

  def sync(self):
    if self.single_device:
      return
    if len(self.train_devices) == 1:
      varibs = self.varibs
    else:
      varibs = tree_map(lambda x: x[0].device_buffer, self.varibs)
    if len(self.policy_devices) == 1:
      self.policy_varibs = jax.device_put(varibs, self.policy_devices[0])
    else:
      self.policy_varibs = jax.device_put_replicated(
          varibs, self.policy_devices)

  def _setup(self):
    try:
      import tensorflow as tf
      tf.config.set_visible_devices([], 'GPU')
      tf.config.set_visible_devices([], 'TPU')
    except Exception as e:
      print('Could not disable TensorFlow devices:', e)
    if not self.config.prealloc:
      os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    xla_flags = []
    if self.config.logical_cpus:
      count = self.config.logical_cpus
      xla_flags.append(f'--xla_force_host_platform_device_count={count}')
    if xla_flags:
      os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
    jax.config.update('jax_platform_name', self.config.platform)
    jax.config.update('jax_disable_jit', not self.config.jit)
    jax.config.update('jax_debug_nans', self.config.debug_nans)
    # jax_transfer_guard='disallow' causes std::terminate() in JAX 0.4.28
    # when numpy arrays enter JIT without explicit device_put (e.g. from ROS2 env).
    # jax.config.update('jax_transfer_guard', 'disallow')
    if self.config.platform == 'cpu':
      jax.config.update('jax_disable_most_optimizations', self.config.debug)
    jaxutils.COMPUTE_DTYPE = getattr(jnp, self.config.precision)

  def _transform(self):
    self._init_policy = nj.pure(lambda x: self.agent.policy_initial(len(x)))
    self._init_train = nj.pure(lambda x: self.agent.train_initial(len(x)))
    self._policy = nj.pure(self.agent.policy)
    self._train = nj.pure(self.agent.train)
    self._report = nj.pure(self.agent.report)
    if len(self.train_devices) == 1:
      kw = dict(device=self.train_devices[0])
      self._init_train = nj.jit(self._init_train, **kw)
      self._train = nj.jit(self._train, **kw)
      self._report = nj.jit(self._report, **kw)
    else:
      kw = dict(devices=self.train_devices)
      self._init_train = nj.pmap(self._init_train, 'i', **kw)
      self._train = nj.pmap(self._train, 'i', **kw)
      self._report = nj.pmap(self._report, 'i', **kw)
    if len(self.policy_devices) == 1:
      kw = dict(device=self.policy_devices[0])
      self._init_policy = nj.jit(self._init_policy, **kw)
      self._policy = nj.jit(self._policy, static=['mode'], **kw)
    else:
      kw = dict(devices=self.policy_devices)
      self._init_policy = nj.pmap(self._init_policy, 'i', **kw)
      self._policy = nj.pmap(self._policy, 'i', static=['mode'], **kw)

  def _convert_inps(self, value, devices):
    if len(devices) == 1:
      value = jax.device_put(value, devices[0])
    else:
      check = tree_map(lambda x: len(x) % len(devices) == 0, value)
      if not all(jax.tree_util.tree_leaves(check)):
        shapes = tree_map(lambda x: x.shape, value)
        raise ValueError(
            f'Batch must by divisible by {len(devices)} devices: {shapes}')
      # TODO: Avoid the reshape?
      value = tree_map(
          lambda x: x.reshape((len(devices), -1) + x.shape[1:]), value)
      shards = []
      for i in range(len(devices)):
        shards.append(tree_map(lambda x: x[i], value))
      value = jax.device_put_sharded(shards, devices)
    return value

  def _convert_outs(self, value, devices):
    value = jax.device_get(value)
    value = tree_map(np.asarray, value)
    if len(devices) > 1:
      value = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
    return value

  def _convert_mets(self, value, devices):
    value = jax.device_get(value)
    value = tree_map(np.asarray, value)
    if len(devices) > 1:
      value = tree_map(lambda x: x[0], value)
    return value

  def _next_rngs(self, devices, mirror=False, high=2 ** 63 - 1):
    if len(devices) == 1:
      return jax.device_put(self.rng.integers(high), devices[0])
    elif mirror:
      return jax.device_put_replicated(
          self.rng.integers(high), devices)
    else:
      return jax.device_put_sharded(
          list(self.rng.integers(high, size=len(devices))), devices)

  def _init_varibs(self, obs_space, act_space):
    # Use policy-only init to avoid tracing the full training graph (imagination
    # rollout, optimizer states, RSSM scan over batch_size*batch_length steps).
    # load() will replace varibs entirely with checkpoint weights, so only the
    # encoder→RSSM→actor graph needs to be traced here.
    varibs = {}
    rng = self._next_rngs(self.policy_devices, mirror=True)
    obs = self._dummy_batch(obs_space, (1,))
    obs = self._convert_inps(obs, self.policy_devices)
    state, varibs = self._init_policy(varibs, rng, obs['is_first'])
    varibs = self._policy(varibs, rng, obs, state, mode='train', init_only=True)
    # Save the keys needed for inference so load() can filter the checkpoint.
    self._policy_keys = set(varibs.keys())
    return varibs

  def _dummy_batch(self, spaces, batch_dims):
    spaces = list(spaces.items())
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
    for dim in reversed(batch_dims):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data
