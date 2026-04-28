# SPOT Setup Guide

## 5. Building the Docker Container

Navigate to the workspace and run the build script. The building process can take **more than 30 minutes** on the RPi:

```bash
./docker_build.sh
```

This will:
- Build a Ubuntu 22.04 image (ROS2 Humble)
- Copy the spot_driver inside the container
- Place it in the `externals` directory (accessible within the container only)

### Running the Container

After building, execute the run script, then build and source the workspace:

```bash
./docker_run.sh
# This automatically starts the container with USB device access (/dev/input mounted)
# The --privileged flag is already included in the script

colcon build
source install/setup.bash
```

**Note:** The `docker_run.sh` script already includes USB device mapping (`-v /dev/input:/dev/input`), so you don't need to add `--privileged` separately.

This will:
- Start the container
- Mount the `openbots/src/packages` directory in it
- The `packages` directory contains a custom UDP to ROS2 bridge

---

## 6. Launching Spot Driver

### Environment Variables

First, set the following environment variables:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export BOSDYN_CLIENT_USERNAME=user
export BOSDYN_CLIENT_PASSWORD=corspotuser1
export SPOT_IP=192.168.10.102
```

> **Note:** `SPOT_IP` should be configured for the robot during Teltonika setup.

### Launch Commands

Launch the Spot driver and associated nodes:

```bash
ros2 launch spot_nav spot_driver_nmea.launch.py \
  cameras_used:=frontleft,frontright \
  stitch_front_images:=true \
  config_file:=/home/ob/openbots_ws/src/packages/spot_recorder_config.yaml
```

This launches:
- `joy_node` (joystick input)
- `nmea_udp_to_ros` node
- `nmea_topic_driver` node
- `spot_driver` node
- `spot_local_grid_node`

### Joystick Teleop (2nd terminal)

To enable joystick control, run in a separate terminal (after the spot driver is running):

```bash
ros2 run dataset spot_teleop_joy --ros-args \
  -p recording_mode:=expert \
  -p recording_difficulty:=medium \
  -p output_dir:=/media/external_drive/recorded_data_NoObs

```

This will start accepting PS4 controller input for controlling the robot. Recordings will be saved directly to the external 1TB drive mounted at `/media/external_drive/recorded_data`.

**Joystick Controls:**
- **Left Stick**: Forward/Backward movement
- **Right Stick**: Rotation (Yaw)
- **X button**: Stand
- **Circle button**: Sit
- **L1**: Auto gait
- **L2**: SpeedSelectTrot gait
- **R1**: Crawl gait
- **R2**: Jog gait
- **L3**: SpeedSelectAmble gait
- **Square button**: Start/Stop recording
- **Triangle button**: Discard current recording (without saving)
- **D-Pad Down**: Mark collision event (flagged in rosbag)


### Data Recording (3rd terminal)

In another terminal, access the container:

```bash
docker exec -it openbots_container_new bash
```

Record sensor data:

```bash
python3 -m dataset.spot_data_recorder

cd src/packages/dataset/
python3 episode_bag_recorder.py
```

Record ROS2 bag:

```bash
ros2 bag record -a
```


mount harddrive:
sudo mkdir -p /media/maurits/recorded_data
sudo chown -R $(id -u):$(id -g) /media/external_drive/recorded_data
sudo chmod -R 775 /media/external_drive/recorded_data
# quick test as the same user:
touch /media/external_drive/recorded_data/test && echo OK || echo FAIL


# 3. After recording, convert the bag to HDF5
cd ~/openbots_ws/src/packages/dataset/
python3 convert_bag_to_hdf5.py --batch --harddrive /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e5/recorded_data_NoObs

# Then compute rewards (note: this uses the parent harddrive path, not the NoObs subfolder)
python3 compute_rewards_batch.py --harddrive /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e6
---

## 7. Spot WiFi Network Configuration

On the **Spot admin panel**:

1. Go to **Admin Panel**
2. Change the network name and password:

| Setting | Value |
|---------|-------|
| **SSID** | RUT_1CF1_5G |
| **Password** | OBrutc%)19 |

3. Connect both the Spot and your PC to the new WiFi network



# DAIC 
## upload data to DAIC from harddrive:

rsync -avz --progress --partial --append-verify --no-perms -e "ssh -J mjheemskerk@student-linux.tudelft.nl" /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e5/processed_data_NoObs_with_rewards mjheemskerk@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/openbots/mjheemskerk/spot_data


## build apptainer: 
apptainer build   --tmpdir /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e2/tmp   /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e2/dreamer_spot.sif   docker-daemon://dreamer_spot_cuda12:latest

## copy apptainer to DAIC
rsync -avz --progress --partial --append-verify --no-perms -e "ssh -J mjheemskerk@student-linux.tudelft.nl"  /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e5/dreamer_spot.sif   mjheemskerk@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/openbots/mjheemskerk/spot_data


## sinteractive:
sinteractive  --nodes=1 --ntasks=1 --gres=gpu:1 

## run training in sinteractive: 
apptainer exec --cleanenv --nv \
  -B /home/nfs/mjheemskerk:/mnt/home \
  -B /tudelft.net:/tudelft.net \
  --env PYTHONNOUSERSITE=1,PYTHONPATH= \
  /tudelft.net/staff-umbrella/openbots/mjheemskerk/dreamer_spotv5.sif \
  python3 /mnt/home/Master_thesis_DAIC_code/dreamerv3/train.py \
    --logdir /mnt/home/logdir/test_run \
    --configs spot \
    --task spot_nav \
    --jax.platform gpu \
    --jax.prealloc False \
    --data_loaders 1 \
    --batch_size 2 \
    --batch_length 8 \
    --replay_size 2e4 \
    --envs.amount 1 \
    --run.train_ratio 16 \
    --env.spot.data_dir=/tudelft.net/staff-umbrella/openbots/mjheemskerk/spot_data/processed_data_with_rewards


## pull logdir files from training to local: 
rsync -avz --progress --exclude='replay' \
  daic:/home/nfs/mjheemskerk/logdir/dreamer-results-20260310-160927 \
  /home/maurits-heemskerk/Documents/Uni/Master_Thesis/dreamer_results_local/

## push results directory back to DAIC logdir:
rsync -avz --progress --exclude='replay' \
  /home/maurits-heemskerk/Documents/Uni/Master_Thesis/dreamer_results_local/dreamer-results-YYYYMMDD-HHMMSS-label \
  daic:/home/nfs/mjheemskerk/logdir/


## run policy:

# Checkpoint path is set in params.yaml — just launch:
ros2 launch dreamer_deployment dreamer_deployment.launch.py

# Or override checkpoint and gait selection inline:
ros2 launch dreamer_deployment dreamer_deployment.launch.py \
  checkpoint_path:=/home/ob/dreamer_results_local/h25_i15/checkpoint_float16_inference.ckpt \
  disable_gait_selection:=true \
  fixed_gait_mode:=trot

ros2 launch dreamer_deployment dreamer_deployment_noobs.launch.py \
  checkpoint_path:=/home/ob/dreamer_results_local_noobs/medium_baseline_dyn_rep_rewardsv3/checkpoint.ckpt \
  record_rosbag:=true \
  stop_at_goal:=false



# Set goal — x/y offset in robot BODY frame (x=forward, y=left)
# frame_id='body' (default, recommended): offset is rotated into world frame using current yaw
                                                                                                                                                      

---

## GPU Inference Setup (NVIDIA GTX 1050 / Pascal SM 6.1)

### Problem
The GTX 1050 in the backpack is Pascal architecture (SM 6.1). Modern JAX/jaxlib (≥0.4.35)
bundles cuDNN 9.x which **requires SM ≥ 7.0 (Volta)**. Trying to run jaxlib 0.6.x on this GPU
causes `CUDNN_STATUS_NOT_SUPPORTED` (status 5003) during conv autotuning.

### Fix: downgrade JAX to 0.4.28
Inside the container, install the last jaxlib release that bundles cuDNN 8.9 (supports SM 6.1):

```bash
pip install "jax[cuda12]==0.4.28" \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

This installs `jax==0.4.28` + `jaxlib==0.4.28+cuda12.cudnn89`. Verify:

```python
import jax
print(jax.__version__)            # 0.4.28
print(jax.devices())              # [CudaDevice(id=0)]  ← must be GPU, not CPU
```

### Fix: chex + optax upgrade (prevents `jax.core.Shape` KeyError)
The JAX 0.4.28 API changed `jax.core.Shape`; old chex/optax crash on checkpoint load:

```bash
pip install -U chex optax
```

### Fix: GPU memory pre-allocation (prevents OOM during XLA autotuning)
Set in `dreamer_deployment.launch.py` (already present):

```python
env = {
    'XLA_FLAGS': '--xla_gpu_strict_conv_algorithm_picker=false',
    'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
}
```

`XLA_PYTHON_CLIENT_PREALLOCATE=false` stops JAX from grabbing all 4 GiB upfront so the
conv autotuner has room to work. Without this the GTX 1050 OOMs.

### Fix: enable GPU in docker_run.sh
The container must be started with GPU access:

```bash
# In docker_run.sh — add to docker run args:
--gpus all
```

Also requires NVIDIA Container Toolkit on the host:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Fix: install NVIDIA Container Toolkit inside container (if needed)
If `nvidia-smi` works on the host but not inside the container:

```bash
# Inside container:
apt-get update && apt-get install -y nvidia-container-toolkit
```

### Expected performance after setup
- **First call**: ~15s (one-time JIT compilation / XLA tuning)
- **Subsequent calls**: ~20–30 ms per policy step (fast enough for 3.5 Hz)
- **Model memory note**: Both `xlargerssm` (deter=4096) and `dyn1.0` (also deter=4096) 
  need ~2.9 GiB at float32. GTX 1050 has only 4 GiB total. Set `jax_precision: float16` 
  in params.yaml to reduce to ~1.5 GiB. Checkpoint must be trained with matching precision.

### Params in params.yaml for GPU mode

```yaml
jax_platform: gpu
jax_precision: float16    # Required for dyn1.0 / xlargerssm on GTX 1050 (both deter=4096)
```

### Summary of version pins (inside openbots container)

| Package | Version | Reason |
|---------|---------|--------|
| `jax` | 0.4.28 | Last version with cuDNN 8.9 (supports SM 6.1 Pascal) |
| `jaxlib` | 0.4.28+cuda12.cudnn89 | Paired with jax 0.4.28 |
| `chex` | latest (≥0.1.86) | Fixes `jax.core.Shape` KeyError on checkpoint load |
| `optax` | latest (≥0.2.x) | Same fix as chex |


## upgrade for jax for container
pip3 install --no-cache-dir \
    "numpy==1.23.5" \
    "jax[cuda12]==0.4.28" \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

---

## 8. Online Finetuning on SPOT

Online finetuning lets you adapt the policy in real-time on the robot using live data.

### ⚠️ CRITICAL SETUP (April 21, 2026)

**Docker Mount**: Ensure harddrive is mounted correctly in `docker_run.sh`:
```bash
-v /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e5:/media/external_drive
```
*(The UUID ends with `e5`, not `e4`)*

**Action Space**: Changed from 4D to 3D (removed gait dimension). Actions are now `[vx, vy, yaw_rate]` only.
- File: `embodied/envs/spot_live.py` line 229, `_publish_action()` method
- This **must** match your checkpoint training (medium_baseline was trained with 3D actions)

**Geofence & Safety**: 5m radius boundary with 0.5m drift tolerance added to `spot_live.py` `_build_obs()`:
- Goals are clamped to stay within 5m from episode start
- Robot gets penalty reward if it strays beyond 5.5m
- Prevents collision with real obstacles and sensor drift issues

**Observation Space**: Checkpoint uses `velocity|orientation|goal` (NO position). The online training script auto-detects this.

### Setup Overview

**Terminal 1: SPOT Driver** (always first — supplies odometry/velocity)
```bash
ros2 launch spot_nav spot_driver_nmea.launch.py \
  cameras_used:=frontleft,frontright \
  stitch_front_images:=true \
  config_file:=/home/ob/openbots_ws/src/packages/spot_recorder_config.yaml
```

**Terminal 2: Online Finetuning** (trains actor/critic in real-time on robot)

### Two Finetuning Modes

**Mode B: Full Online Training** (recommended here - trains both WM + AC)
- Use when both policy and world model need real-world correction
- New live data progressively improves the world model
- Slower than frozen mode, but closest to full online adaptation

```bash
cd src/dreamer_SPOT_implementation/informed-dreamer

python dreamerv3/train_online.py \
  --configs spot_live_frozen \
  --run.from_checkpoint /home/ob/dreamer_results_local_noobs/medium_dyn_rep_newrewards/checkpoint.ckpt \
  --logdir ./online_runs/frozen_$(date +%Y%m%d_%H%M%S) \
  --env.spot.data_dir /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e6/processed_data_NoObs_with_rewards
```

### Configuration Details

| Parameter | Frozen Mode | Full Mode | Meaning |
|-----------|------------|-----------|---------|
| `spot_live_frozen` | ✓ | - | Freezes world model losses |
| `spot_live_full` | - | ✓ | Trains world model alongside AC |
| Batch Size | 8 | 8 | Replay buffer batch |
| Train Ratio | 32 | 32 | Gradient steps per environment step |
| Steps | 5e5 | 5e5 | Total training steps |
| Precision | float16 (GPU) | default | Memory optimization |

### Goal Schedule Format

Autonomous goal sequence (JSON format):
```python
--env.spotlive.goal_schedule "[[5,0],[3,2],[-4,1],[0,0]]"
```
Each `[x, y]` is a goal in **world frame** (meters). Robot visits each sequentially.

Leave `--env.spotlive.goal_schedule` empty to use manual goal mode.
The live env will then wait for a `/spot/policy/goal` message at the start of
each episode.
```bash
# In separate terminal while training is running:
ros2 topic pub --once /spot/policy/goal \
  geometry_msgs/msg/PointStamped \
  "{header: {frame_id: 'body'}, point: {x: 3.0, y: 0.0, z: 0.0}}"
```

### Automatic Config Loading

The `train_online.py` script now **automatically loads the checkpoint's config** (RSSM architecture, layer sizes, etc.). You only specify:
- `--configs spot_live_full` — full online learning mode
- `--run.from_checkpoint` — path to the checkpoint
- `--env.spot.data_dir` — offline H5 replay seed
- omit `--env.spotlive.goal_schedule` for manual goals

The checkpoint's RSSM config (stoch, deter, classes) is preserved exactly, preventing shape mismatches.

### Environment Setup

Set JAX/XLA flags before running:
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export XLA_FLAGS='--xla_gpu_strict_conv_algorithm_picker=false'
export JAX_TRACEBACK_FILTERING=off  # Optional: full error traces
```

If you have processed data (H5 files) from earlier training, pre-populate the replay buffer:
```bash
--env.spot.data_dir /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e6/processed_data_NoObs_with_rewards
```

Without `--env.spot.data_dir`, training starts with an empty replay buffer and learns purely from live robot data (slower convergence).

### Data Flow

The `SPOTLive` environment:
- **Subscribes to**: `/odometry` (position, orientation, velocity), `/spot/policy/goal`
- **Publishes to**: `/cmd_vel` (velocity commands to robot)
- **Pre-loads** (optional): offline H5 data from `--env.spot.data_dir` into replay buffer
  - Prevents catastrophic forgetting of offline training
  - Seeds training with diverse behaviors from the offline dataset
  - Omit this flag to train purely from live robot data (higher risk of forgetting, but discovers new behaviors)
- **Records live data**: automatically collected during rollouts and mixed into replay buffer

### Workflow

1. **Collect offline dataset** → Process to HDF5 with rewards
2. **Train offline model** → Get baseline checkpoint
3. **Optional: Frozen finetuning** → Adapt policy only (fast, safe)
4. **Optional: Full finetuning** → Adapt full model (powerful, slower)
5. **Collect more data** → Record episodes for next iteration
6. **Export checkpoint** → Use new weights in `dreamer_policy_node_noobs.launch.py`

### Performance Notes

- **First step**: ~5–10s (JIT compilation, device memory allocation)
- **Steady state**: ~10–15 ms per policy step (~100 Hz internally, downsampled to 3.5 Hz for robot)
- **Replay buffer**: Automatically balanced between offline data and live data
- **Checkpoints**: Saved periodically to `logdir/` — can be used immediately in deployment

### Quick Repeat Checklist (Tomorrow)

1. ✓ Start SPOT driver in Terminal 1:
   ```bash
   ros2 launch spot_nav spot_driver_nmea.launch.py cameras_used:=frontleft,frontright stitch_front_images:=true
   ```

2. ✓ Inside container, run in Terminal 2:
   ```bash
   cd /home/ob/openbots_ws/src/dreamer_SPOT_implementation/informed-dreamer
   
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
   
   python dreamerv3/train_online.py \
     --configs spot_live_full \
     --run.from_checkpoint /home/ob/dreamer_results_local_noobs/medium_baseline/checkpoint.ckpt \
     --logdir ./online_runs/full_$(date +%Y%m%d_%H%M%S) \
     --env.spot.data_dir /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e6/processed_data_NoObs_with_rewards
   ```

   Then set the target yourself from a third terminal with `ros2 topic pub`.

3. ✓ Training auto-loads checkpoint config (RSSM, encoder, decoder, actor, critic)
4. ✓ Offline replay is prefilled from `/media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e6/processed_data_NoObs_with_rewards`
5. ✓ Geofence enforced (5m radius, 0.5m tolerance)
6. ✓ Manual goals are enabled by omitting `goal_schedule`
7. ✓ Observation space auto-detected from checkpoint (velocity|orientation|goal, no position)

You can now record a rosbag with velocity, target goal, orientation, and actions by adding the flag `record_rosbag:=true` to your launch command:

```bash
ros2 launch dreamer_deployment dreamer_deployment_noobs.launch.py \
  checkpoint_path:=/home/ob/dreamer_results_local_noobs/medium_dyn_rep_newrewards/checkpoint.ckpt \
  dreamerv3_root:=/home/ob/informed-dreamer \
  record_rosbag:=true
```

This will save `/cmd_vel`, `/spot/policy/goal`, `/odometry`, and `/spot/policy_action_debug` to a bag directory named `rosbag/` next to the checkpoint path, for example:

`/home/ob/dreamer_results_local_noobs/medium_dyn_rep_newrewards/rosbag`
