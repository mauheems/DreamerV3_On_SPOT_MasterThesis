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

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/openbots/mjheemskerk/dreamer_spotv5.sif"
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
    --run.train_ratio 16 \
    --env.spot.data_dir=/tudelft.net/staff-umbrella/openbots/mjheemskerk/spot_data/processed_data_with_rewards
