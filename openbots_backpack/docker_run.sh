#!/bin/bash

IMAGE_NAME="openbots"

echo "Launching Docker container from image: $IMAGE_NAME"

docker run -it \
    --hostname ob_container \
    --network host \
    -e TERM=xterm-256color \
    -v $(pwd)/src/packages:/home/ob/openbots_ws/src/packages \
    -v /dev/input:/dev/input \
    -v /media/maurits-heemskerk/69987a47-b840-4db7-9f8b-7cc05f14d09e1:/media/external_drive \
    --device-cgroup-rule='c 13:* rmw' \
    --group-add=995 \
    -w /home/ob/openbots_ws \
    --name openbots_container_new \
    $IMAGE_NAME
