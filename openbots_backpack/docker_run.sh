#!/bin/bash

IMAGE_NAME="openbots"

echo "Launching Docker container from image: $IMAGE_NAME"

docker run -it \
    --hostname ob_container \
    --network host \
    -e TERM=xterm-256color \
    -v $(pwd)/src/packages:/home/ob/openbots_ws/src/packages \
    -v /dev/input:/dev/input \
    --device-cgroup-rule='c 13:* rmw' \
    -w /home/ob/openbots_ws \
    --name openbots_container_new \
    $IMAGE_NAME
