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
    Pillow \
    "numpy<2.0" \
    "opencv-python<4.10"

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

RUN echo "source /opt/ros/humble/setup.bash" >> /home/ob/.bashrc

ENTRYPOINT ["/usr/local/bin/fix_hosts.sh"]
CMD ["bash"]
