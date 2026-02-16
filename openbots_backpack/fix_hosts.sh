#!/bin/bash
# Fix /etc/hosts to include hostname mapping to 127.0.1.1
HOSTNAME=$(hostname)
if ! grep -q "$HOSTNAME" /etc/hosts; then
  echo "127.0.1.1 $HOSTNAME" | sudo tee -a /etc/hosts
fi

exec "$@"
