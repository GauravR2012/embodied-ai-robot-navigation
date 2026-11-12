#!/usr/bin/env bash
#
# Stop every leftover Gazebo / SLAM / Nav2 process so the next start is clean.
#
# Stale processes from a previous run (Gazebo bridges, SLAM, or Nav2 servers
# that survived Ctrl+C) cause duplicate action servers (for example two
# /spin servers, which makes Nav2 reject goals) and half-configured nodes.
#
# Usage:
#   scripts/sim_clean.sh --dry-run   # only list what would be stopped
#   scripts/sim_clean.sh             # stop everything, restart the ROS daemon
#
# Run it in a spare terminal BEFORE starting the three launch terminals.
# RViz is intentionally left alone.

set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# Anchored patterns, so an editor that merely has a launch file open (or a
# `tail` on a log) is never matched. Only real launch commands and the actual
# executables are.
PATTERNS=(
  "ros2 launch .*embodied_ai_world\\.launch\\.py"
  "ros2 launch .*online_async_launch\\.py"
  "ros2 launch .*navigation_launch\\.py"
  "(^|/)gz sim"
  "(^|/)parameter_bridge( |$)"
  "(^|/)robot_state_publisher( |$)"
  "(^|/)async_slam_toolbox_node( |$)"
  "(^|/)sync_slam_toolbox_node( |$)"
  "(^|/)controller_server( |$)"
  "(^|/)smoother_server( |$)"
  "(^|/)planner_server( |$)"
  "(^|/)route_server( |$)"
  "(^|/)behavior_server( |$)"
  "(^|/)bt_navigator( |$)"
  "(^|/)waypoint_follower( |$)"
  "(^|/)velocity_smoother( |$)"
  "(^|/)collision_monitor( |$)"
  "(^|/)lifecycle_manager( |$)"
)

# This script, its parent shell(s) and the terminal are never stopped, even if
# their command line happens to mention one of the patterns.
PROTECTED=" $$ "
parent=$PPID
while [ -n "$parent" ] && [ "$parent" -gt 1 ] 2>/dev/null; do
  PROTECTED="$PROTECTED$parent "
  parent=$(ps -o ppid= -p "$parent" 2>/dev/null | tr -d ' ')
done

collect_pids() {
  for pattern in "${PATTERNS[@]}"; do
    pgrep -f "$pattern" 2>/dev/null
  done | sort -un | while read -r pid; do
    case "$PROTECTED" in
      *" $pid "*) ;;
      *) echo "$pid" ;;
    esac
  done
}

PIDS=$(collect_pids)

if [ -z "$PIDS" ]; then
  echo "No leftover simulation / Nav2 processes found."
else
  echo "Processes matched:"
  # shellcheck disable=SC2086
  ps -o pid=,etime=,args= -p $(echo $PIDS | tr ' ' ',') 2>/dev/null | cut -c1-120
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo
  echo "(dry run: nothing was stopped)"
  exit 0
fi

if [ -n "$PIDS" ]; then
  echo
  echo "Stopping (SIGTERM)..."
  # shellcheck disable=SC2086
  kill -TERM $PIDS 2>/dev/null || true
  sleep 3

  LEFT=$(collect_pids)

  if [ -n "$LEFT" ]; then
    echo "Still running, forcing (SIGKILL)..."
    # shellcheck disable=SC2086
    kill -KILL $LEFT 2>/dev/null || true
    sleep 1
  fi
fi

if command -v ros2 >/dev/null 2>&1; then
  echo "Restarting the ROS 2 daemon..."
  ros2 daemon stop >/dev/null 2>&1 || true
  sleep 1
  ros2 daemon start >/dev/null 2>&1 || true
  sleep 1

  NODES=$(ros2 node list 2>/dev/null || true)

  if [ -n "$NODES" ]; then
    echo
    echo "WARNING: these ROS nodes are still visible (another machine, a"
    echo "terminal you forgot, or ROS_DOMAIN_ID shared with other robots?):"
    echo "$NODES"
    exit 1
  fi
else
  echo "(ros2 not found in this shell; source ROS to restart the daemon)"
fi

echo
echo "Clean. Start the terminals in order: Gazebo, SLAM, Nav2."
