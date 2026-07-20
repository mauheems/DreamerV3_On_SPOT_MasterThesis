from pathlib import Path

import embodied
import h5py
import numpy as np


def populate_replay(replay, data_dir, shuffle_episodes=True, seed=None):
    """Load all H5 episodes into the replay buffer using the recorded actions.

    Each episode is assigned a unique worker index so their internal streams
    (sliding deques) stay independent. Actions stored in the replay are the
    actions that were actually taken during data collection, not policy outputs.

    Args:
        replay: Replay buffer to populate
        data_dir: Directory containing episode files
        shuffle_episodes: If True, randomize episode order to prevent trajectory order overfitting
        seed: Random seed for reproducibility (if shuffle_episodes=True)

    NOTE: H5 actions are assumed to be in the same normalised [-1, 1] range as
    the policy action space. If your recorded actions are physical velocities
    (e.g. vel_x in m/s) you must normalise them here before storing.
    """
    data_dir = Path(data_dir)
    episode_files = sorted(
        data_dir.glob('**/*.h5'), key=lambda p: p.stat().st_mtime)
    assert episode_files, f'No .h5 episodes found in {data_dir}'
    
    # Shuffle episodes to randomize injection order and prevent trajectory order overfitting
    if shuffle_episodes:
        rng = np.random.RandomState(seed)
        rng.shuffle(episode_files)
        print(f'[offline] Loading {len(episode_files)} episodes into replay (shuffled)...')
    else:
        print(f'[offline] Loading {len(episode_files)} episodes into replay...')

    for worker, ep_file in enumerate(episode_files):
        with h5py.File(ep_file, 'r') as f:
            velocities = f['observations/velocities'][:]
            states     = f['observations/state'][:]
            actions    = f['actions'][:]
            rewards    = f['rewards'][:].astype(np.float32)

        length   = len(velocities)
        positions = states[:length, :2].astype(np.float32)
        goal_pos  = positions[-1, :2].copy()

        for t in range(length):
            pos = states[t, :2].astype(np.float32)
            qx, qy, qz, qw = states[t, 3:7]
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            orientation = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)

            # velocity: keep only [vx, vy, wz] — drop vz (idx 2), wx/wy pitch+roll rates (idx 3,4)
            vel = velocities[t, [0, 1, 5]].astype(np.float32)

            goal_rel = np.array([goal_pos[0] - pos[0], goal_pos[1] - pos[1]], dtype=np.float32)

            raw_act  = actions[t].astype(np.float32)
            # Noobs dataset is trot-only; normalise to [-1, 1] using trot limits.
            # H5 actions may be 3D [vx, vy, vyaw] or legacy 4D [vx, vy, vyaw, gait].
            # We always take only the first 3 dims to match act_space (3,).
            norm_act = np.array([
                raw_act[0] / 2.0,   # vx_max = 2.0 m/s (trot)
                raw_act[1] / 1.0,   # vy_max = 1.0 m/s (trot)
                raw_act[2] / 1.0,   # vyaw_max = 1.0 rad/s
            ], dtype=np.float32)
            norm_act = np.clip(norm_act, -1.0, 1.0)

            replay.add({
                'velocity':     vel,
                'position':     pos,
                'orientation':  orientation,
                'goal':         goal_rel,
                'reward':       np.float32(rewards[t]),
                'is_first':     np.bool_(t == 0),
                'is_last':      np.bool_(t == length - 1),
                'is_terminal':  np.bool_(t == length - 1),
                'action':       norm_act,
                'reset':        np.bool_(t == length - 1),
            }, worker=worker)

        print(f'  [{worker + 1}/{len(episode_files)}] {ep_file.name} ({length} steps)')

    print(f'[offline] Replay populated: {len(replay)} sequences.')


def train_offline(agent, env, replay, logger, args):
    """Pure offline training loop.

    The replay is pre-populated from H5 files (with recorded actions) before
    training begins. No environment stepping occurs during training — the env
    is only used to supply obs_space / act_space for agent initialisation.

    Step counting: step is incremented by batch_steps (= batch_size *
    batch_length) per gradient update, mirroring the online convention where
    step counts individual environment transitions. Set run.steps accordingly,
    e.g. run.steps: 5e7 gives ~48 800 gradient steps with default batch_steps.
    """
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print('Logdir', logdir)

    should_log  = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_sync = embodied.when.Every(args.sync_every)
    step    = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()

    print('Observation space:', embodied.format(env.obs_space), sep='\n')
    print('Action space:',      embodied.format(env.act_space), sep='\n')

    timer = embodied.Timer()
    timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])

    # Pre-populate replay unless we are resuming from a checkpoint whose saved
    # replay already contains data (handle that case by checking len > 0).
    data_dir = getattr(args, 'offline_data_dir', None) or (
        args.get('env', {}).get('spot', {}).get('data_dir'))
    assert data_dir, (
        'Provide data_dir via env.spot.data_dir in your config or set '
        'run.offline_data_dir explicitly.')

    if len(replay) == 0:
        # Shuffle episodes by default to prevent trajectory order overfitting
        shuffle = getattr(args, 'shuffle_episodes', True)
        seed = getattr(args, 'episode_shuffle_seed', None)
        populate_replay(replay, data_dir, shuffle_episodes=shuffle, seed=seed)
    else:
        print(f'[offline] Replay already has {len(replay)} sequences, skipping load.')

    assert len(replay) >= args.batch_steps, (
        f'Replay has {len(replay)} sequences but needs at least {args.batch_steps} '
        f'(batch_size={args.batch_size} x batch_length={args.batch_length}). '
        'Reduce batch size or collect more data.')

    dataset = agent.dataset(replay.dataset)
    state = [None]
    batch = [None]

    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)

    print('Start offline training loop.')
    while step < args.steps:
        with timer.scope('dataset'):
            batch[0] = next(dataset)
        outs, state[0], mets = agent.train(batch[0], state[0])
        metrics.add(mets, prefix='train')
        if 'priority' in outs:
            replay.prioritize(outs['key'], outs['priority'])
        updates.increment()
        step.increment(args.batch_steps)

        if should_sync(updates):
            agent.sync()
        if should_log(step):
            agg = metrics.result()
            report = agent.report(batch[0])
            report = {k: v for k, v in report.items() if 'train/' + k not in agg}
            logger.add(agg)
            logger.add(report, prefix='report')
            logger.add(replay.stats, prefix='replay')
            logger.add(timer.stats(), prefix='timer')
            logger.write(fps=True)
        if should_save(step):
            checkpoint.save()

    checkpoint.save()
    logger.write()
