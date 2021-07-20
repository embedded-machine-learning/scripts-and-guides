#!/bin/bash

echo "Start script with . /[PATH_TO_SCRIPT], not with ./"

echo "=== Init task spooler ==="
echo "Setup task spooler socket for CPU."

export TS_SOCKET="/srv/ts_socket/CPU.socket"
chmod 777 /srv/ts_socket/CPU.socket
export TS_TMPDIR=~/logs
echo task spooler output directory: ~/logs

echo "Task spooler initialized $TS_SOCKET"
