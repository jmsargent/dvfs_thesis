#!/usr/bin/env bash
set -euo pipefail

# gsync.sh — wrapper to run gsync_broker.py as a long-lived daemon.
#
# Usage:
#   ./gsync.sh start <repo_dir>
#   ./gsync.sh stop <repo_dir>
#   ./gsync.sh status <repo_dir>
#   ./gsync.sh run <repo_dir>       # foreground, auto-restart on crash
#
# You can also pass the target repository with REPO_DIR=/path/to/repo.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BRANCH="${BRANCH:-main}"
REMOTE="${REMOTE:-origin}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"
RESULT_INTERVAL="${RESULT_INTERVAL:-${STATUS_INTERVAL:-5}}"
PUSH_INTERVAL="${PUSH_INTERVAL:-30}"
RESTART_DELAY="${RESTART_DELAY:-10}"
GIT_RECOVERY_FAILURES="${GIT_RECOVERY_FAILURES:-5}"

PY="${PY:-python3}"
BROKER_PY="${BROKER_PY:-$SCRIPT_DIR/gsync_broker.py}"

usage() {
  cat >&2 <<'USAGE'
Usage:
  ./gsync.sh start <repo_dir>
  ./gsync.sh stop <repo_dir>
  ./gsync.sh status <repo_dir>
  ./gsync.sh run <repo_dir>

Alternative:
  REPO_DIR=/path/to/repo ./gsync.sh <start|stop|status|run>
USAGE
}

CMD="${1:-}"
REPO_ARG="${2:-}"
if [[ -z "$CMD" ]]; then
  usage
  exit 2
fi

resolve_repo_dir() {
  local raw="${REPO_DIR:-$REPO_ARG}"
  if [[ -z "$raw" ]]; then
    echo "Missing repo_dir. Pass <repo_dir> or set REPO_DIR." >&2
    usage
    exit 2
  fi
  if [[ ! -d "$raw" ]]; then
    echo "Repo directory does not exist: $raw" >&2
    exit 2
  fi
  local abs
  abs="$(cd "$raw" && pwd)"
  if [[ ! -d "$abs/.git" ]]; then
    echo "Not a git repository: $abs" >&2
    exit 2
  fi
  echo "$abs"
}

case "$CMD" in
  start|stop|status|run) ;;
  *)
    usage
    exit 2
    ;;
esac

REPO_DIR="$(resolve_repo_dir)"
LOG_DIR="${LOG_DIR:-$REPO_DIR/.broker}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/broker.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/broker.pid}"

cleanup_pid_file_if_owned() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" || true)"
    if [[ "$pid" == "$$" ]]; then
      rm -f "$PID_FILE"
    fi
  fi
}

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

cmd_start() {
  if is_running; then
    echo "gsync broker already running (pid=$(cat "$PID_FILE"))."
    exit 0
  fi

  rm -f "$PID_FILE"
  nohup "$0" run "$REPO_DIR" >>"$LOG_FILE" 2>&1 &
  echo "$!" > "$PID_FILE"
  echo "Started gsync broker. pid=$! repo=$REPO_DIR log=$LOG_FILE"
}

cmd_stop() {
  if ! [[ -f "$PID_FILE" ]]; then
    echo "No pid file: $PID_FILE (already stopped?)"
    exit 0
  fi
  local pid
  pid="$(cat "$PID_FILE" || true)"
  if [[ -z "$pid" ]]; then
    echo "Empty pid file. Remove it: $PID_FILE"
    exit 1
  fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid"
    echo "Sent SIGTERM to pid=$pid"
    local i
    for ((i = 0; i < 30; i++)); do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "gsync broker did not exit within 30s; send SIGKILL manually if needed."
      exit 1
    fi
  else
    echo "Process not running. Removing pid file."
  fi
  rm -f "$PID_FILE"
}

cmd_status() {
  if is_running; then
    echo "RUNNING pid=$(cat "$PID_FILE") repo=$REPO_DIR log=$LOG_FILE"
  else
    echo "STOPPED (pid file: $PID_FILE)"
    exit 1
  fi
}

cmd_run() {
  local child_pid=""
  local rc=0
  local stop_requested=0

  trap '' HUP
  trap '
    stop_requested=1
    echo "[gsync.sh] stop requested"
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" >/dev/null 2>&1; then
      kill "$child_pid" >/dev/null 2>&1 || true
    fi
  ' TERM INT
  trap 'cleanup_pid_file_if_owned' EXIT

  echo "[gsync.sh] starting run loop: repo=$REPO_DIR branch=$BRANCH remote=$REMOTE"
  echo "[gsync.sh] log=$LOG_FILE"
  echo "[gsync.sh] result_interval=${RESULT_INTERVAL}s"
  echo "[gsync.sh] restart_delay=${RESTART_DELAY}s"
  echo "[gsync.sh] git_recovery_failures=${GIT_RECOVERY_FAILURES}"

  while (( ! stop_requested )); do
    "$PY" "$BROKER_PY" \
      --repo "$REPO_DIR" \
      --remote "$REMOTE" \
      --branch "$BRANCH" \
      --poll-interval "$POLL_INTERVAL" \
      --result-interval "$RESULT_INTERVAL" \
      --push-interval "$PUSH_INTERVAL" \
      --git-recovery-failures "$GIT_RECOVERY_FAILURES" &
    child_pid=$!

    if wait "$child_pid"; then
      rc=0
    else
      rc=$?
    fi
    child_pid=""

    if (( stop_requested )); then
      break
    fi

    echo "[gsync.sh] gsync_broker.py exited rc=$rc; restarting in ${RESTART_DELAY}s..."
    local slept=0
    while (( slept < RESTART_DELAY )) && (( ! stop_requested )); do
      sleep 1 || true
      slept=$((slept + 1))
    done
  done

  echo "[gsync.sh] run loop stopped."
}

case "$CMD" in
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  run)    cmd_run ;;
esac
