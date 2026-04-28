#!/bin/bash
#SBATCH --job-name=dreamer-spot-noobs
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

module use /opt/insy/modulefiles
module load cuda/12.1

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/openbots/mjheemskerk/dreamer_spot.sif"
export LOGDIR="/tudelft.net/staff-umbrella/openbots/mjheemskerk/logdir/dreamer-results-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"

srun apptainer exec --cleanenv --nv \
  -B /home/nfs/mjheemskerk:/mnt/home \
  -B /tudelft.net:/tudelft.net \
  -B "$LOGDIR:/logdir" \
  -B /dev/shm:/dev/shm \
  --env PYTHONNOUSERSITE=1,PYTHONPATH= \
  --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  "$APPTAINER_IMAGE" \
  python3 /mnt/home/Master_thesis_DAIC_code/dreamerv3/train.py \
    --logdir /logdir \
    --configs spot_simple \
    --task spot_nav \
    --jax.platform gpu \
    --jax.prealloc False \
    --data_loaders 4 \
    --envs.amount 1 \
    --env.spot.data_dir=/tudelft.net/staff-umbrella/openbots/mjheemskerk/spot_data/noobs_data \
    \
    --rssm.deter=512 \
    --rssm.units=512 \
    --rssm.stoch=32 \
    --rssm.classes=16 \
    \
    --batch_size=16 \
    --batch_length=64 \
    --replay_size=2e6 \
    \
    --actor.units=256 \
    --actor.layers=3 \
    --critic.units=256 \
    --critic.layers=3 \
    --reward_head.units=256 \
    --reward_head.layers=3 \
    --cont_head.units=256 \
    --cont_head.layers=3 \
    \
    --imag_horizon=20 \
    \
    --loss_scales.vector=1.0 \
    --loss_scales.reward=1.0 \
    --loss_scales.cont=1.0 \
    --loss_scales.dyn=2.0 \
    --loss_scales.rep=0.5 \
    --loss_scales.physics=0.0 \
    --loss_scales.multi_step=0.0 \
    --rep_loss.impl=kl \
    --rep_loss.free=1.0 \
    \
    --run.script=train_offline \
    --run.steps=2e7 \
    --run.log_every=300 \
    --run.save_every=3600 \
    --run.train_ratio=512 \
    --run.shuffle_episodes=True \
    --run.episode_shuffle_seed=0
