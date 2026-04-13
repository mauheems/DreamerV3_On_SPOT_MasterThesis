#!/bin/bash
#SBATCH --job-name=dreamer-spot-training
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96GB
#SBATCH --mail-type=END
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

module use /opt/insy/modulefiles
module load cuda/12.1

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/openbots/mjheemskerk/dreamer_spot.sif"
export LOGDIR="/tudelft.net/staff-umbrella/openbots/mjheemskerk/logdir/dreamer-results-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"

apptainer exec --cleanenv --nv \
  -B /home/nfs/mjheemskerk:/mnt/home \
  -B /tudelft.net:/tudelft.net \
  -B "$LOGDIR:/logdir" \
  --env PYTHONNOUSERSITE=1,PYTHONPATH= \
  "$APPTAINER_IMAGE" \
  python3 /mnt/home/Master_thesis_DAIC_code/dreamerv3/train.py \
    --logdir /logdir \
    --configs spot \
    --task spot_nav \
    --jax.platform gpu \
    --jax.prealloc False \
    --data_loaders 1 \
    --batch_size 2 \
    --batch_length 8 \
    --replay_size 4e4 \
    --envs.amount 1 \
    --env.spot.data_dir=/tudelft.net/staff-umbrella/openbots/mjheemskerk/spot_data/processed_data_with_rewards \
    --encoder.mlp_keys='velocity|position|orientation|goal' \
    --encoder.cnn_keys='image' \
    --decoder.mlp_keys='velocity|position|orientation|goal' \
    --decoder.cnn_keys='image' \
    --encoder.cnn_depth=64 \
    --decoder.cnn_depth=64 \
    --rssm.deter=2048 \
    --rssm.units=768 \
    --actor.units=768 \
    --actor.layers=4 \
    --critic.units=768 \
    --critic.layers=4 \
    --reward_head.units=768 \
    --reward_head.layers=4 \
    --cont_head.units=768 \
    --cont_head.layers=4 \
    --disag_head.units=768 \
    --disag_head.layers=4 \
    --loss_scales.image=1.0 \
    --loss_scales.vector=3.0 \
    --loss_scales.reward=1.0 \
    --loss_scales.cont=1.0 \
    --loss_scales.dyn=2 \
    --loss_scales.rep=1 \
    --loss_scales.physics=0.1 \
    --physics_dt=0.3 \
    --multi_step_horizon=7 \
    --loss_scales.multi_step=0.2 \
    --multi_step_loss.impl=kl \
    --multi_step_loss.free=0.5 \
    --rep_loss.impl=kl \
    --rep_loss.free=1 \
    --run.script=train_offline \
    --run.steps=2e7 \
    --run.log_every=300 \
    --run.save_every=3600 \
    --run.sync_every=10 \
    --run.train_ratio=256 \
    --run.shuffle_episodes=True \
    --run.episode_shuffle_seed=0
