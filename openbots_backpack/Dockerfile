FROM ubuntu:22.04

#RMW ZENOH experimental flag
ARG EXPERIMENTAL_ZENOH_RMW=FALSE

ENV DEBIAN_FRONTEND=noninteractive

ENV SHELL=/bin/bash
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    locales \
    curl \
    git \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    
    
# Add ROS 2 apt repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update

# Install system + ROS dependencies
RUN apt-get update -q && \
    apt-get install -yq --no-install-recommends \
    wget \
    software-properties-common \
    python3-pip \
    python-is-python3 \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    libpython3-dev \
    python3-tk \
    ros-humble-ros-base \
    ros-dev-tools \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-tf-transformations \
    ros-humble-nmea-navsat-driver \
    ros-humble-teleop-twist-keyboard \
    ros-humble-teleop-twist-joy \
    netcat \
    nano \
    sudo \
    $(if [ "$EXPERIMENTAL_ZENOH_RMW" = "TRUE" ]; then echo "ros-humble-rmw-zenoh-cpp"; fi) \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip3 install --no-cache-dir \
    bosdyn-client==5.0.0 \
    bosdyn-mission==5.0.0 \
    bosdyn-choreography-client==5.0.0 \
    aiortc \
    h5py>=3.6.0 \
    Pillow==12.1.1 \
    "numpy<2.0" \
    "opencv-python<4.10"

# DreamerV3 / JAX inference dependencies
# Pin to a ROS-compatible stack:
# - NumPy 1.x is required by cv_bridge/spot_driver and old transforms3d usage.
# - jax[cuda12]==0.4.28 is the last release bundling cuDNN 8.9, which supports
#   Pascal GPUs (SM 6.1, e.g. GTX 1050 in the backpack).  Newer jaxlib (>=0.4.35)
#   bundles cuDNN 9.x which requires Volta (SM>=7.0) and fails with status 5003.
# - chex + optax at latest fix the jax.core.Shape KeyError on checkpoint load.
# - TensorFlow 2.11.x in this image requires protobuf<3.20.
RUN pip3 install --no-cache-dir \
    "numpy==1.23.5" \
    "jax[cuda12]==0.4.28" \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    "ruamel.yaml" \
    "tensorflow_probability==0.23.0" \
    "protobuf==3.19.6" \
    rich \
    cloudpickle

# chex and optax must be installed AFTER jax so they pin against the correct version.
# Latest versions fix the jax.core.Shape KeyError that occurs when loading checkpoints
# produced with newer jax on an older deployment install.
RUN pip3 install --no-cache-dir -U chex optax

# Smoke-test core ML imports at build time so version conflicts fail early.
# Force CPU platform so the build host doesn't need a GPU.
RUN JAX_PLATFORM_NAME=cpu python3 - <<'PY'
import numpy
import jax
import tensorflow_probability as tfp
from tensorflow_probability.substrates import jax as tfp_jax
print('numpy', numpy.__version__)
print('jax', jax.__version__)
print('tfp', tfp.__version__)
print('tfp_jax', getattr(tfp_jax, '__name__', 'ok'))
# Sanity-check that optax and chex load without the jax.core.Shape crash
import optax, chex
print('optax', optax.__version__)
print('chex', chex.__version__)
PY

COPY fix_hosts.sh /usr/local/bin/fix_hosts.sh
RUN chmod +x /usr/local/bin/fix_hosts.sh
RUN useradd -m -s /bin/bash ob && \
    echo "ob ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/ob && \
    chmod 0440 /etc/sudoers.d/ob

COPY --chown=ob:ob bashrclr /home/ob/.bashrc

RUN mkdir -p /home/ob/openbots_ws/src
WORKDIR /home/ob/openbots_ws

COPY --chown=ob:ob src/externals /home/ob/openbots_ws/src/externals
COPY --chown=ob:ob src/packages /home/ob/openbots_ws/src/packages

USER ob
# install script is provided by spot_ros2 package
RUN cd /home/ob/openbots_ws/src/externals/spot_ros2 && \
    ./install_spot_ros2.sh 

RUN sudo chown -R ob:ob /home/ob/openbots_ws

RUN source /opt/ros/humble/setup.bash && \
    cd /home/ob/openbots_ws && \
    colcon build --symlink-install

# Re-pin JAX after install_spot_ros2.sh and colcon build, which may upgrade jax/jaxlib.
# jax[cuda12]==0.4.28 is the last release bundling cuDNN 8.9 (supports Pascal SM 6.1).
# numpy==1.23.5 is pinned for ROS cv_bridge compatibility.
RUN sudo pip3 install --no-cache-dir \
    "numpy==1.23.5" \
    "jax[cuda12]==0.4.28" \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN echo "source /opt/ros/humble/setup.bash" >> /home/ob/.bashrc

ENTRYPOINT ["/usr/local/bin/fix_hosts.sh"]
CMD ["bash"]
