#!/usr/bin/env python3

import rospy
import socket
from geometry_msgs.msg import PoseStamped

# ROS Publisher
rospy.init_node("robot_navigation")
pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)

# Socket Server Configuration
HOST = "0.0.0.0"  # Listen on all network interfaces
PORT = 5555       # Must match the port used in the backend script

# Predefined target locations (x, y, z) in map frame
locations = {
    "Station 1": (1.5, 0.017, 0.00294),
    "Station 2": (-1.4, -0.058,0.0021),
    "Base Station": (-2.21, -0.397, 0.00157),
}

def send_goal(x, y, z):
    """Sends a navigation goal to move_base."""
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "map"
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    pose.pose.orientation.w = 1.0  # Neutral orientation

    pub.publish(pose)
    rospy.loginfo(f"Navigating to: X={x}, Y={y}, Z={z}")

def handle_client(client_socket):
    """Handles incoming station commands from the backend."""
    with client_socket:
        data = client_socket.recv(1024).decode("utf-8").strip()
        if data in locations:
            rospy.loginfo(f"Received command: {data}")
            x, y, z = locations[data]
            send_goal(x, y, z)
        else:
            rospy.logwarn(f"Unknown location received: {data}")

# Create a socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)  # Allow up to 5 connections

rospy.loginfo(f"Listening for station commands on {HOST}:{PORT}")

while not rospy.is_shutdown():
    client_socket, _ = server_socket.accept()
    handle_client(client_socket)

