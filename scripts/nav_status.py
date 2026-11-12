#!/usr/bin/env python3
"""
Is Nav2 really up?  (single-process replacement for the old shell checker)

Checks, in ONE ROS process (fast even on a loaded laptop):

  1. every Nav2 lifecycle node reports the state "active"
  2. /spin and /navigate_to_pose each have exactly ONE action server
  3. no node name appears twice (a stale second Nav2 / SLAM instance)

Usage (ROS sourced):

    python3 scripts/nav_status.py                # check once
    python3 scripts/nav_status.py --wait 90      # keep checking up to 90 s

(`bash scripts/nav_status.sh ...` does the same thing.)

Exit code 0 = healthy, 1 = problem.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

NODES = [
    "controller_server",
    "smoother_server",
    "planner_server",
    "route_server",
    "behavior_server",
    "velocity_smoother",
    "collision_monitor",
    "bt_navigator",
    "waypoint_follower",
]

ACTIONS = ["/spin", "/navigate_to_pose"]

SELF_NAME = "nav_status_check"


def full_name(namespace: str, name: str) -> str:
    return f"{namespace.rstrip('/')}/{name}"


def wait_until(predicate, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec

    while True:
        if predicate():
            return True

        if time.monotonic() >= deadline:
            return False

        time.sleep(0.05)


def check_once(node, clients, get_state_type, timeout_sec: float) -> int:
    """Print one full report. Returns the number of problems."""

    problems = 0

    # -- 1. lifecycle states -------------------------------------------
    print("Lifecycle state of the Nav2 nodes:")

    wait_until(
        lambda: all(c.service_is_ready() for c in clients.values()),
        timeout_sec,
    )

    futures = {
        name: client.call_async(get_state_type.Request())
        for name, client in clients.items()
        if client.service_is_ready()
    }

    wait_until(lambda: all(f.done() for f in futures.values()), timeout_sec)

    for name in NODES:
        future = futures.get(name)

        if future is None:
            print(f"  [FAIL] {name:<18} not reachable (node not running?)")
            problems += 1
        elif not future.done():
            print(f"  [FAIL] {name:<18} did not answer within "
                  f"{timeout_sec:.0f} s")
            problems += 1
        else:
            label = future.result().current_state.label

            if label == "active":
                print(f"  [ OK ] {name:<18} {label}")
            else:
                print(f"  [FAIL] {name:<18} {label}")
                problems += 1

    # -- 2. action servers ---------------------------------------------
    print("\nAction servers (must be exactly 1 each):")

    for action in ACTIONS:
        endpoints = node.get_publishers_info_by_topic(
            f"{action}/_action/status"
        )

        owners = sorted(
            full_name(e.node_namespace, e.node_name) for e in endpoints
        )

        if len(owners) == 1:
            print(f"  [ OK ] {action:<18} 1 server ({owners[0]})")
        else:
            detail = ", ".join(owners) if owners else "none"
            print(f"  [FAIL] {action:<18} {len(owners)} server(s): {detail}")
            problems += 1

    # -- 3. duplicate node names ------------------------------------------
    print("\nDuplicate node names:")

    names = [
        full_name(namespace, name)
        for name, namespace in node.get_node_names_and_namespaces()
        if name != SELF_NAME
    ]

    duplicates = sorted({n for n in names if names.count(n) > 1})

    if not duplicates:
        print("  [ OK ] none")
    else:
        for name in duplicates:
            print(f"  [FAIL] {name} appears {names.count(name)} times")
            problems += 1

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nav2 health check.")

    parser.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="Keep retrying for up to this many seconds (default: once).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for each group of answers (default 10).",
    )

    args = parser.parse_args(argv)

    try:
        import rclpy
        from lifecycle_msgs.srv import GetState
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
    except Exception as exc:
        print(
            f"ROS Python packages not available ({exc}).\n"
            "Source ROS first: source /opt/ros/<distro>/setup.bash"
        )
        return 1

    rclpy.init()

    node = Node(SELF_NAME)
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    clients = {
        name: node.create_client(GetState, f"/{name}/get_state")
        for name in NODES
    }

    try:
        # Give DDS discovery a moment before the first look.
        time.sleep(2.0)

        deadline = time.monotonic() + args.wait

        while True:
            problems = check_once(node, clients, GetState, args.timeout)

            if problems == 0:
                print("\nNav2 is healthy: all nodes active, one server "
                      "per action, no duplicates.")
                return 0

            if time.monotonic() >= deadline:
                print(f"\nNav2 is NOT healthy ({problems} problem(s)).")
                print("Look at the Nav2 launch terminal for the first "
                      "[ERROR] line.")
                print("If nodes or servers are duplicated, run "
                      "scripts/sim_clean.sh and restart the terminals.")
                return 1

            print("\n--- not ready yet, retrying in 3 s ---\n")
            time.sleep(3.0)

    finally:
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
