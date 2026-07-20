"""Online finetuning entry point for DreamerV3 on SPOT.

Workflow
--------
1. Load config from offline checkpoint + CLI overrides.
2. Create SPOTLiveEnv (ROS2) as the live environment.
3. Pre-populate the replay buffer from offline H5 episodes (same data the
   offline checkpoint was trained on) so the world model doesn't forget.
4. Load the offline checkpoint weights.
5. Run the standard DreamerV3 ``train`` loop at a lower train_ratio so that
   gradient steps don't overwhelm the small number of live transitions.

Two finetuning modes are supported via config:
  - ``spot_live_frozen``  — WM frozen, actor+critic only (safe first step)
  - ``spot_live_full``    — WM + AC trained together (more powerful, riskier)

Typical usage
-------------
Frozen WM:
  python train_online.py \\
    --configs spot_simple spot_live_frozen \\
    --run.from_checkpoint /path/to/offline/run/checkpoint.ckpt \\
    --logdir /path/to/online/run_frozen \\
    --env.spotlive.goal_schedule "[[5,0],[3,2],[-4,1]]"

Full online:
  python train_online.py \\
    --configs spot_simple spot_live_full \\
    --run.from_checkpoint /path/to/offline/run/checkpoint.ckpt \\
    --logdir /path/to/online/run_full \\
    --env.spotlive.goal_schedule "[[5,0],[3,2],[-4,1]]"

The offline data directory is read from ``env.spot.data_dir`` in the loaded
config (same key the offline training used).  Override with
``--env.spot.data_dir /other/path`` if needed.
"""

import faulthandler
import signal
faulthandler.enable()  # print C backtrace on SIGABRT/SIGSEGV
# Register SIGABRT handler for extra diagnostics
_orig_abrt = signal.getsignal(signal.SIGABRT)
def _abrt_handler(sig, frame):
    import traceback
    print("[SIGABRT received] Python stack:", flush=True)
    traceback.print_stack(frame)
    faulthandler.dump_traceback()
    if callable(_orig_abrt):
        _orig_abrt(sig, frame)
signal.signal(signal.SIGABRT, _abrt_handler)
import importlib
import json
import os
import pathlib
import sys
import warnings
try:
    from ruamel import yaml
except ImportError:
    import yaml

# Must be set before JAX/XLA is imported to prevent GPU OOM on first JIT compile.
# GTX 1050 has only 4 GiB; without this JAX pre-allocates all of it and crashes.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.75')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_strict_conv_algorithm_picker=false')
# Verbose CUDA error reporting so OOM shows as a message before terminate
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
# CYCLONEDDS_URI is set inside the ROS2 subprocess worker (spot_live.py).
# No DDS config needed in the main training process (rclpy never loads here).
from functools import partial as bind

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent  # now points to dreamerv3/
# Add informed-dreamer root to path so that "dreamerv3" imports resolve to source, not installed
directory_root = directory.parent  # now points to informed-dreamer/
sys.path.insert(0, str(directory_root))
sys.path.insert(0, str(directory_root.parent))
sys.path.insert(0, str(directory_root.parent.parent))
__package__ = directory.name

import embodied
from embodied import wrappers
from embodied.envs.spot_live import SPOTLive
from embodied.run.train_offline import populate_replay


def main(argv=None):
    from . import agent as agt

    # First pass: parse to extract checkpoint path (if provided)
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    checkpoint_path = None
    for arg in other:
        if arg.startswith('--run.from_checkpoint='):
            checkpoint_path = arg.split('=', 1)[1]
        elif arg == '--run.from_checkpoint' and other.index(arg) + 1 < len(other):
            checkpoint_path = other[other.index(arg) + 1]
    
    # Load checkpoint config if available
    checkpoint_config = None
    if checkpoint_path:
        ckpt_dir = pathlib.Path(checkpoint_path).parent
        ckpt_config_file = ckpt_dir / 'config.yaml'
        if ckpt_config_file.exists():
            print(f'[train_online] Loading checkpoint config from {ckpt_config_file}')
            try:
                yaml_loader = yaml.YAML(typ='safe')
                config_dict = yaml_loader.load(ckpt_config_file.read_text())
                checkpoint_config = embodied.Config(config_dict)
            except Exception as e:
                print(f'[train_online] WARNING: Failed to load checkpoint config ({e}), using defaults')
                checkpoint_config = None
    
    # Start with checkpoint config if available, otherwise defaults
    if checkpoint_config is not None:
        config = checkpoint_config
        # Save the checkpoint's model architecture (RSSM, encoder, decoder, actor, critic)
        # so online config doesn't override it
        saved_rssm = config.get('rssm', {})
        saved_encoder = config.get('encoder', {})
        saved_decoder = config.get('decoder', {})
        saved_actor = config.get('actor', {})
        saved_critic = config.get('critic', {})
    else:
        config = embodied.Config(agt.Agent.configs['defaults'])
        saved_rssm = None
        saved_encoder = None
        saved_decoder = None
        saved_actor = None
        saved_critic = None
    
    # Apply named configs from CLI (these override checkpoint config)
    for name in parsed.configs:
        if name in agt.Agent.configs:
            config = config.update(agt.Agent.configs[name])
        else:
            print(f'[train_online] WARNING: config "{name}" not found, skipping')
    
    # Restore checkpoint's model architecture if we loaded from checkpoint
    # (online configs should not change model, only training parameters)
    if checkpoint_config is not None:
        print('[train_online] Preserving checkpoint model architecture (RSSM, encoder, decoder, actor, critic)')
        config = config.update({
            'rssm': saved_rssm,
            'encoder': saved_encoder,
            'decoder': saved_decoder,
            'actor': saved_actor,
            'critic': saved_critic,
        })
    
    # Apply CLI overrides (highest priority)
    config = embodied.Flags(config).parse(other)
    
    # Explicitly handle --env.spot.data_dir if provided (sometimes embodied.Flags misses nested overrides)
    for i, arg in enumerate(other):
        if arg == '--env.spot.data_dir' and i + 1 < len(other):
            data_dir_override = other[i + 1]
            print(f'[train_online] Applying CLI override: env.spot.data_dir = {data_dir_override}')
            config = config.update({'env': {'spot': {'data_dir': data_dir_override}}})
        # Handle encoder.mlp_keys override but preserve checkpoint if not provided
        elif arg == '--encoder.mlp_keys' and i + 1 < len(other):
            mlp_keys_override = other[i + 1]
            print(f'[train_online] Applying CLI override: encoder.mlp_keys = {mlp_keys_override}')
            config = config.update({'encoder': {'mlp_keys': mlp_keys_override}})

    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
        env=config.env,
    )

    print(config)

    # ── JAX platform detection: fallback to CPU if requested platform unavailable ──
    import jax
    platform = config.jax.get('platform', 'cpu')
    try:
        jax.devices(platform)
        print(f'[JAX] Using platform: {platform}')
    except Exception as e:
        fallback = 'cpu'
        print(f'[JAX] WARNING: platform "{platform}" unavailable ({e.__class__.__name__}), falling back to {fallback}')
        config = config.update({'jax': {'platform': fallback}})

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    config.save(logdir / 'config.yaml')

    step = embodied.Counter()
    logger = _make_logger(parsed, logdir, step, config)

    # ── Live environment ─────────────────────────────────────────────────────
    env = _make_live_env(config)
    agent = agt.Agent(env.obs_space, env.act_space, step, config)

    # ── Replay — pre-populate from offline H5 data ───────────────────────────
    data_dir = _resolve_data_dir(config)
    replay = _make_replay(config, logdir / 'replay', args)

    if data_dir:
        print(f'[train_online] Pre-populating replay from {data_dir} …')
        try:
            from pathlib import Path
            data_path = Path(data_dir)
            if data_path.exists():
                h5_count = len(list(data_path.glob('**/*.h5')))
                print(f'[train_online] Dataset check: {data_path} -> {h5_count} .h5 files')
                shuffle = getattr(args, 'shuffle_episodes', True)
                seed    = getattr(args, 'episode_shuffle_seed', None)
                populate_replay(replay, data_dir, shuffle_episodes=shuffle, seed=seed)
            else:
                print(f'[train_online] WARNING: data_dir does not exist: {data_dir}')
                print('[train_online] Starting with empty replay buffer')
        except AssertionError as e:
            print(f'[train_online] WARNING: {e}')
            print('[train_online] Starting with empty replay buffer')
    else:
        print('[train_online] WARNING: no offline data dir found — '
              'replay starts empty.  Set env.spot.data_dir in your config.')

    # ── Load checkpoint ───────────────────────────────────────────────────────
    # The standard train loop will call checkpoint.load(from_checkpoint) when
    # args.from_checkpoint is set, so we don't duplicate that here.

    # ── Run online training ───────────────────────────────────────────────────
    embodied.run.train(agent, env, replay, logger, args)

    env.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_live_env(config):
    """Instantiate SPOTLive with settings from config.env.spotlive."""
    live_cfg = config.env.get('spotlive', {})

    # Get obs_keys from encoder config, or fall back to spotlive config
    encoder_cfg = config.get('encoder', {})
    encoder_mlp_keys = encoder_cfg.get('mlp_keys', 'velocity|position|orientation|goal')
    obs_keys = live_cfg.get('obs_keys', encoder_mlp_keys)
    
    print(f'[_make_live_env] Using obs_keys: {obs_keys} (from encoder: {encoder_mlp_keys})')

    policy_rate_hz = float(live_cfg.get('policy_rate_hz', 3.5))
    goal_timeout   = float(live_cfg.get('goal_timeout', 300.0))
    max_steps      = int(live_cfg.get('max_episode_steps', 200))
    reach_thr      = float(live_cfg.get('reach_threshold', 0.5))

    # goal_schedule can be provided as a JSON string or a list
    raw_schedule = live_cfg.get('goal_schedule', '')
    if raw_schedule is None or raw_schedule == '' or raw_schedule == 'None':
        goal_schedule = None  # interactive mode
    elif isinstance(raw_schedule, str):
        try:
            goal_schedule = json.loads(raw_schedule)
        except json.JSONDecodeError:
            print(f'[goal_schedule] WARNING: Invalid JSON: {raw_schedule!r}, using interactive mode')
            goal_schedule = None
    else:
        goal_schedule = raw_schedule  # None or list

    env = SPOTLive(
        task='noobs',
        obs_keys=obs_keys,
        policy_rate_hz=policy_rate_hz,
        goal_schedule=goal_schedule,
        goal_timeout=goal_timeout,
        max_episode_steps=max_steps,
        reach_threshold=reach_thr,
    )
    return _wrap_env(env, config)


def _wrap_env(env, config):
    args = config.wrapper
    for name, space in env.act_space.items():
        if name == 'reset':
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = wrappers.NormalizeAction(env, name)
    if args.length:
        env = wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = wrappers.CheckSpaces(env)
    return env


def _resolve_data_dir(config):
    """Return the offline H5 data directory from config (or None)."""
    try:
        data_dir = config.env.spot.data_dir
        if data_dir:
            print(f'[_resolve_data_dir] Found data_dir: {data_dir}')
            return data_dir
    except Exception as e:
        print(f'[_resolve_data_dir] DEBUG: config.env.spot.data_dir lookup failed: {e}')
    try:
        data_dir = config.run.offline_data_dir
        if data_dir:
            print(f'[_resolve_data_dir] Found offline_data_dir: {data_dir}')
            return data_dir
    except Exception as e:
        print(f'[_resolve_data_dir] DEBUG: config.run.offline_data_dir lookup failed: {e}')
    print('[_resolve_data_dir] No data_dir found in config')
    return None


def _make_replay(config, directory, args):
    length = config.batch_length
    size   = config.replay_size
    if config.replay == 'uniform':
        kw = {'online': config.replay_online}
        # Do NOT set samples_per_insert here — it causes a deadlock when
        # offline episodes are pre-populated before any training samples are
        # drawn.  train_ratio is handled by embodied.run.train directly.
        kw['min_size'] = config.batch_size
        return embodied.replay.Uniform(length, size, directory, **kw)
    raise NotImplementedError(f'Unsupported replay type: {config.replay}')


def _make_logger(parsed, logdir, step, config):
    multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
    return embodied.Logger(step, [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
    ], multiplier)


if __name__ == '__main__':
    main()
