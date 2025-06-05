import os
import sys
import time
import re
import json
import socket  # For socket communication
import threading
import tempfile
import base64
import cv2  # Import cv2 for image decoding

from PIL import Image
import ollama
import numpy as np

# ==========================
# Configuration Parameters
# ==========================

# Socket Configuration
GRADIO_HOST = "localhost"    # Host where the Gradio webapp socket listener is running
GRADIO_PORT = 12345          # Port where the Gradio socket listener is listening
BACKEND_HOST = "localhost" #"192.168.0.162"   # Host for the backend socket server (self) (this system)
BACKEND_PORT = 12346          # Port for the backend socket server to listen on

# LLM Model Configuration
LLM_MODEL = "llama3.1:8b-instruct-fp16"

# ==========================
# Initialize Global Variables
# ==========================
intelligence = None

# ==========================
# Socket Communication Function
# ==========================

def send_message_to_gradio(message):
    """
    Sends a message to the Gradio webapp via socket, appending a newline delimiter.
    
    Parameters:
        message (str): The message to send.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((GRADIO_HOST, GRADIO_PORT))
            s.sendall((message + '\n').encode('utf-8'))  # Append newline delimiter
            print(f"Sent message to Gradio: {message}")
    except Exception as e:
        print(f"Error sending message to Gradio: {e}")


def send_to_ros_machine(station):
    ROS_HOST = "192.168.0.174"  # Replace with the IP address of the ROS machine (laptop)
    ROS_PORT = 5555           # Replace with the port the ROS script is listening on

    print("INSIDE SEND TO ROS MACHINE")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ROS_HOST, ROS_PORT))
            s.sendall(station.encode('utf-8'))
            print(f"Sent {station} to ROS machine.")
    except Exception as e:
        print(f"Error sending data to ROS machine: {e}")

# ==========================
# Central Intelligence Class
# ==========================

class CentralIntelligence:
    def __init__(self, model):
        self.model = model

    def interpret_task(self, user_sentence):
        """
        Interprets the task based on user input to determine the relevant storage location.

        Parameters:
            user_sentence (str): User-provided sentence about the tool.

        Returns:
            str: Relevant location or 'None'.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
You are an assistant trained to identify locations given an input.
Based on the input provided, choose the **single most relevant location**. If the input is related to tools like hammer, screwdriver,
etc., choose Station 1. If the input is related to hazardous materials or anything related to hazard like danger, caution, etc., choose Station 2.
If the if the input is related to base station or home position or charging station choose Base Station.
Answer only from the given options.
- Station 1
- Station 2
- Base Station
Sentence: "{user_sentence}"

Respond with **only one option** from the list or 'None' if no location is relevant.
"""
                    },
                ]
            )
            response_content = response.get("message", {}).get("content", "").strip()
            if not response_content:
                print("No valid content received from the model.")
                return "None"
            return response_content
        except KeyError as e:
            print(f"Error parsing response: {e}")
            return "None"
        except Exception as e:
            print(f"Error during task interpretation: {e}")
            return "None"

def object_detection_ollama(image):
    """
    Detects objects in the provided image using Ollama's llama3.2-vision model.

    Parameters:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        str: Name(s) of the detected object(s) or an error message.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
            temp_image_path = temp_image.name
            # Convert the NumPy array to PIL Image and save it
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pil_image.save(temp_image_path)

        # Define a concise prompt
        concise_prompt = "List only the names of the objects present in this image, separated by commas. Do not include any additional information, descriptions, adjectives, or possessive terms."

        # Send the image to Ollama's vision model with the concise prompt
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': concise_prompt,
                'images': [temp_image_path]
            }]
        )

        # Clean up the temporary file
        os.remove(temp_image_path)

        # Process the response
        if 'message' in response and 'content' in response['message']:
            detected_objects = response['message']['content'].strip().rstrip('.!?')
            return detected_objects
        else:
            return "Ollama did not return a valid response."

    except Exception as e:
        return f"Error during object detection with Ollama: {e}"



def backend_socket_server():
    """
    Backend socket server that listens for messages from the Gradio webapp,
    processes them, and sends back responses.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.bind((BACKEND_HOST, BACKEND_PORT))
    except socket.error as e:
        print(f"Backend socket binding failed: {e}")
        sys.exit()
    server_socket.listen()
    print(f"Backend socket server started on {BACKEND_HOST}:{BACKEND_PORT}")

    while True:
        try:
            client_socket, addr = server_socket.accept()
            threading.Thread(target=handle_client, args=(client_socket, addr), daemon=True).start()
        except Exception as e:
            print(f"Error accepting connections: {e}")
            continue

def handle_client(client_socket, addr):
    """
    Handles incoming messages from the Gradio webapp.
    
    Parameters:
        client_socket (socket.socket): The client socket.
        addr (tuple): The address of the client.
    """
    global intelligence  # Access the global instance
    buffer = ""
    delimiter = "\n"  # Define message delimiter
    
    with client_socket:
        try:
            while True:
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    # No more data from client
                    break
                buffer += data
                while delimiter in buffer:
                    # Split messages by delimiter
                    message, buffer = buffer.split(delimiter, 1)
                    if not message:
                        continue
                    print(f"Received message from {addr}: {message}")
                    payload = json.loads(message)
                    
                    # Process the payload based on its type
                    if payload['type'] == 'image':
                        # Decode base64 string back to image
                        img_base64 = payload['data']
                        img_bytes = base64.b64decode(img_base64)
                        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is None:
                            send_message_to_gradio("Failed to decode image.")
                            continue
                        
                        detected_object = object_detection_ollama(img)
                        send_message_to_gradio(f"Ollama Detection: {detected_object}")
                        
                        # Interpret task based on detected object
                        relevant_location = intelligence.interpret_task(detected_object)
                        send_message_to_gradio(f"Relevant Location: {relevant_location}")
                        send_to_ros_machine(f"{relevant_location}")
                        
                        # Perform planning based on relevant location
                        if relevant_location != "None":
                            task_description = f"Store the {detected_object} in {relevant_location}."
                        else:
                            task_description = f"No relevant location found for {detected_object}."
                        
                        
                    elif payload['type'] == 'text':
                        user_text = payload['data']
                        print(f"Processing text input: {user_text}")
                        relevant_location = intelligence.interpret_task(user_text)
                        send_message_to_gradio(f"Relevant Location: {relevant_location}")
                        send_to_ros_machine(f"{relevant_location}")
                        
                        # Perform planning based on relevant location
                        if relevant_location != "None":
                            task_description = f"Store the tool in {relevant_location}."
                        else:
                            task_description = f"No relevant location found for the tool."
                        
                        
                    else:
                        send_message_to_gradio("Unknown payload type received.")
        except json.JSONDecodeError as e:
            print(f"JSON decoding error from {addr}: {e}")
            send_message_to_gradio(f"JSON decoding error: {e}")
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
            send_message_to_gradio(f"Error processing input: {e}")

# ==========================
# Main Execution Flow
# ==========================

def main():
    """
    Main function to initialize components and start the backend socket server.
    """
    global intelligence
    intelligence = CentralIntelligence(LLM_MODEL)

    # Start the backend socket server
    backend_socket_server()

if __name__ == "__main__":
    main()