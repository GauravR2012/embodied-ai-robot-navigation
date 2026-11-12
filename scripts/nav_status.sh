#!/usr/bin/env bash
#
# Is Nav2 really up? Thin wrapper around scripts/nav_status.py, which checks
# everything inside ONE ROS process (the old version called the slow `ros2`
# command-line tool nine times and gave false alarms on a loaded laptop).
#
# Usage (ROS sourced):
#   bash scripts/nav_status.sh              # check once
#   bash scripts/nav_status.sh --wait 90    # keep checking up to 90 s
#
# Exit code 0 = healthy, 1 = problem.

exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nav_status.py" "$@"
