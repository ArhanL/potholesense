#!/usr/bin/env bash
# Restart the dev server cleanly (kills whatever holds the port).
PORT="${1:-8000}"
for p in $(cd /proc && ls -d [0-9]* 2>/dev/null); do
  [ -r "/proc/$p/cmdline" ] || continue
  if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "run\.py --port $PORT"; then
    kill -9 "$p" 2>/dev/null
  fi
done
sleep 2
cd "$(dirname "$0")/.."
POTHOLESENSE_STUB="${POTHOLESENSE_STUB:-}" setsid nohup python3 run.py --port "$PORT" \
  > /tmp/server.log 2>&1 < /dev/null &
for i in $(seq 1 20); do
  sleep 1
  curl -s -m 2 "localhost:$PORT/health" > /dev/null 2>&1 && { echo "server up on $PORT"; exit 0; }
done
echo "server failed to start"; tail -20 /tmp/server.log; exit 1
