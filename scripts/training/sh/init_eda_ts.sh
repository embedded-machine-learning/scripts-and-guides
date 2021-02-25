#!/bin/bash

echo "Start script with . /[PATH_TO_SCRIPT], not with ./"

echo "=== Init task spooler ==="
echo "Setup task spooler socket for GPU."

export TS_SOCKET="/srv/ts_socket/GPU.socket"
chmod 777 /srv/ts_socket/GPU.socket
export TS_TMPDIR=~/logs
echo task spooler output directory: ~/logs

echo "Task spooler initialized $TS_SOCKET"
