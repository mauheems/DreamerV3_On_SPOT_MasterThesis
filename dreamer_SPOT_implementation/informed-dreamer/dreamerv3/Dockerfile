# 1. Test setup:
# docker run -it --rm --gpus all nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 nvidia-smi
#
# If the above does not work, try adding the --privileged flag
# and changing the command to `sh -c 'ldconfig -v && nvidia-smi'`.
#
# 2. Start training:
# docker build -f  dreamerv3/Dockerfile -t img . && \
# docker run -it --rm -v ~/logdir:/logdir dreamer_spot_img python3 dreamerv3/train.py --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" --configs spot --task spot_nav

# 3. See results:
# tensorboard --logdir ~/logdir

# System
# System
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1

# Basic utilities
RUN apt-get update && apt-get install -y \
  ffmpeg git python3-pip vim \
  && apt-get clean
RUN pip3 install --upgrade pip

# Base ML dependencies (install before JAX so pip does not downgrade/replace JAX later)
RUN pip3 install "numpy<2" tensorflow_probability optax cloudpickle ruamel.yaml rich zmq msgpack h5py

# Known-working CUDA JAX stack
RUN pip3 install --upgrade \
  "jax==0.6.2" \
  "jaxlib==0.6.2" \
  "jax-cuda12-plugin[with-cuda]==0.6.2" \
  "jax-cuda12-pjrt==0.6.2"

# Copy your code
COPY . /embodied
RUN chown -R 1000:root /embodied && chmod -R 775 /embodied

WORKDIR /embodied