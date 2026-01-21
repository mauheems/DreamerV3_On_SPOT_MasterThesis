FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

SHELL ["/bin/bash", "-c"]

# --------------------------------------------------
# 1. Basic system dependencies
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    locales \
    build-essential \
    wget \
    git \
    python3-pip \
    python-is-python3 \
    software-properties-common \
    && locale-gen en_US.UTF-8

# --------------------------------------------------
# 2. Add ROS 2 apt repository
# --------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list

# --------------------------------------------------
# 3. Install ROS2 Humble + ROS tools
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-nav-msgs \
    ros-humble-cv-bridge \
    python3-rosdep \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# 4. Spot Python SDK (no robot required yet)
# --------------------------------------------------
RUN pip install --no-cache-dir \
    bosdyn-client==5.0.1 \
    bosdyn-core==5.0.1 \
    bosdyn-api==5.0.1

# --------------------------------------------------
# 5. ROS environment
# --------------------------------------------------
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

CMD ["bash"]
