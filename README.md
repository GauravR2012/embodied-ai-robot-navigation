# LLM-Controlled Robot Navigation

This repository enables **Large Language Model (LLM)-based natural language control of a robot** using a modular backend and frontend architecture. The backend parses natural language instructions and interfaces with ROS, while the frontend provides a web-based UI to send commands and visualize robot behavior.

---

## Project Structure

- `LLM_backend.py` – Handles language input, interprets commands, and communicates with ROS.  
- `I3D_web_app.py` – Web-based frontend using Gradio, receives instructions and displays feedback.  
- `robot_control.py` – Implements the robot’s low-level control logic using ROS publishers/subscribers.

---

## Requirements

- Python 3.8 or higher  
- ROS (Robot Operating System) installed and configured on the robot machine  
- Python packages:
  - `gradio`
  - `transformers`
  - `torch`
  - `rospy`

Install required Python packages with:
```bash
pip install gradio transformers torch
```

## Setup Instructions
You can run both backend and UI on the same system (for development) or on different systems (for deployment). Follow one of the setups below based on your requirement.

Case 1: Backend and UI on the same system
Open a terminal and run:
```bash
python3 LLM_backend.py
```
Ensure the following values are set in LLM_backend.py:
```python
GRADIO_HOST = "localhost"
BACKEND_HOST = "localhost"
ROS_HOST = "10.114.55.251"  # Replace with your ROS machine's IP
```
In a second terminal, run:
```bash
python3 I3D_web_app.py
```
Ensure the following values are set in I3D_web_app.py:
```python
BACKEND_HOST = "localhost"
server_socket.bind(("localhost", GRADIO_RECEIVE_PORT))
```
After launch, Gradio will print a local URL in the terminal. Open it in your browser.

Case 2: Backend and UI on different systems
On the backend system:
```bash
python3 LLM_backend.py
```
In LLM_backend.py, set:
```python
GRADIO_HOST = "<IP_of_UI_system>"           # IP of the machine running I3D_web_app.py
BACKEND_HOST = "<IP_of_backend_system>"     # IP of this backend machine
ROS_HOST = "10.114.55.251"                  # Replace with your ROS machine's IP
```
On the UI system:
```bash
python3 I3D_web_app.py
```
In I3D_web_app.py, set:
```python
BACKEND_HOST = "<IP_of_backend_system>"     # IP of the backend machine
server_socket.bind(("<IP_of_UI_system>", GRADIO_RECEIVE_PORT))
```
Make sure both systems are on the same local network and can communicate using the specified IPs.

# Citation
If you use this code or build upon this work, please cite:

[1] Mukund Mitra, Yashaswi Sinha, Arushi Khokhar, Sairam Jinkala, and Pradipta Biswas. 2025. LMD-FISH: Language Model Driven - Framework for Intelligent Scheduling of Heterogenous Systems. In Companion Proceedings of the 30th International Conference on Intelligent User Interfaces (IUI '25 Companion). Association for Computing Machinery, New York, NY, USA, 74–77. https://doi.org/10.1145/3708557.3716333
