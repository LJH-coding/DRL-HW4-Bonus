#!/bin/bash
set -euo pipefail

DISPLAY_NUM="${DISPLAY_NUM:-99}"
SCREEN_RES="${SCREEN_RES:-1280x720x24}"

if ! command -v Xvfb >/dev/null 2>&1; then
  echo "Xvfb is not installed. Install it first, e.g.:"
  echo "  sudo apt-get install -y xvfb"
  exit 1
fi

if ! pgrep -f "Xvfb :${DISPLAY_NUM}" >/dev/null 2>&1; then
  Xvfb ":${DISPLAY_NUM}" -screen 0 "${SCREEN_RES}" &
  XVFB_PID=$!
  trap 'kill ${XVFB_PID} >/dev/null 2>&1 || true' EXIT
  sleep 1
fi

export DISPLAY=":${DISPLAY_NUM}"

if [ "$#" -eq 0 ]; then
  python BTR.py
else
  "$@"
fi
