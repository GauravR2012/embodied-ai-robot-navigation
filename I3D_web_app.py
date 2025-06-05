import gradio as gr
import socket
import threading
import queue
import json
import base64
from io import BytesIO
from PIL import Image
from transformers import pipeline
import time

# ========== Configuration ==========
BACKEND_HOST = "localhost"   #system where backend is running
BACKEND_PORT = 12346
GRADIO_RECEIVE_PORT = 12345
messages_queue = queue.Queue()
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# ========== Socket Listener ==========
def backend_listener():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("localhost", GRADIO_RECEIVE_PORT))      #this system ip
    server_socket.listen()
    print(f"[Gradio UI] Listening on port {GRADIO_RECEIVE_PORT} for backend messages...")
    while True:
        client_socket, _ = server_socket.accept()
        with client_socket:
            buffer = ""
            while True:
                data = client_socket.recv(4096).decode("utf-8")
                if not data:
                    break
                buffer += data
                while "\n" in buffer:
                    message, buffer = buffer.split("\n", 1)
                    msg = message.strip()
                    if msg:
                        print(f"[Gradio UI] Received: {msg}")
                        messages_queue.put(msg)

threading.Thread(target=backend_listener, daemon=True).start()

# ========== Send to Backend ==========
def send_to_backend(payload):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((BACKEND_HOST, BACKEND_PORT))
            message = json.dumps(payload) + "\n"
            s.sendall(message.encode("utf-8"))
            print(f"[Gradio UI] Sent to backend: {message.strip()}")
    except Exception as e:
        print(f"[Gradio UI] Error sending to backend: {e}")
        gr.Warning("Backend unreachable.")

# ========== Transcribe Voice ==========
def transcribe_audio(audio_file):
    if audio_file is None:
        return ""
    try:
        result = asr_pipeline(audio_file)
        return result["text"]
    except Exception as e:
        print(f"[ASR Error] {e}")
        return ""

# ========== Process Input ==========
def process_user_input(image, user_text, voice_text):
    voice_text = voice_text.strip()
    user_text = user_text.strip()
    final_text = voice_text if voice_text else user_text

    if image is not None:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        payload = {
            "type": "image",
            "data": base64.b64encode(buffered.getvalue()).decode("utf-8")
        }
        send_to_backend(payload)
        gr.Info("📤 Image input sent!", duration=8)
    elif final_text:
        payload = {
            "type": "text",
            "data": final_text
        }
        send_to_backend(payload)
        if voice_text:
            gr.Info("🎤 Audio sent!", duration=2)
        else:
            gr.Info(f"Sent text: {final_text}", duration=2)
    else:
        gr.Info("Provide image or text input.", duration=3)

    return (
        None,
        "",
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=""),
        gr.update(visible=False)
    )

# ========== Refresh Backend Messages with Custom HTML Toast ==========
def refresh_messages():
    if not messages_queue.empty():
        msg = messages_queue.get()
        html = f"""
        <div id=\"backend-toast\" style=\"
    position: fixed;
    top: 40%;
    left: 50%;
    transform: translate(-50%, -50%);
    padding: 20px 40px;
    background-color: #2196F3;
    color: white;
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    border-radius: 10px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    z-index: 9999;
">
             {msg}
        </div>
        <script>
            setTimeout(function() {{
                var toast = document.getElementById('backend-toast');
                if (toast) {{
                    toast.style.display = 'none';
                }}
            }}, 2000);
        </script>
        """
        return html
    else:
        return """
        <script>
            var toast = document.getElementById('backend-toast');
            if (toast) {
                toast.style.display = 'none';
            }
        </script>
        """

# ========== Gradio UI ==========
with gr.Blocks(css="""
    .compact textarea,
    .compact input[type='text'],
    .compact .gr-button,
    .compact .gr-audio,
    .compact canvas {
        height: 40px !important;
        font-size: 14px !important;
        padding: 4px 6px !important;
    }

    .compact .gr-audio {
        height: 60px !important;
    }

    .compact label {
        font-size: 14px;
        margin-bottom: 2px;
    }

    .compact canvas {
        height: 100px !important;
    }

    .center-button {
        display: flex;
        justify-content: center;
        margin-top: 10px;
    }
               
.align-center {
    margin-left: auto;
    margin-right: auto;
}
               
#image-btn button, #audio-btn button, .gr-button {
    aspect-ratio: 4 / 3 !important;
    width: 160px !important;
    font-size: 20px !important;
}      

""") as demo:

    gr.Image(
        value="/home/nahar/git repos/depth_comp/UniDepth/scripts/banner_1.png",
        interactive=False,
        show_label=False,
        container=False
    )

    gr.HTML("""
        <h1 style="
            text-align: center;
            font-size: 64px;
            font-weight: 900;
            color: #1a237e;
            margin-top: 10px;
            margin-bottom: 20px;
        ">
        </h1>
    """)

    image_input = gr.Image(visible=False, type="pil", container=False, elem_id="hidden_image")
    voice_input = gr.Audio(visible=False, type="filepath", container=False, elem_id="hidden_audio")
    transcribed_text = gr.Textbox(visible=False, container=False, elem_id="hidden_transcription")

    with gr.Row():
        with gr.Column(scale=0.3, elem_classes="align-center"):            
            text_input = gr.Textbox(label="Text Input", lines=1)
            image_icon = gr.Button(" Image Input", elem_id="image-btn")
            audio_icon = gr.Button(" Audio Input", elem_id="audio-btn")



    with gr.Row():
        with gr.Column(visible=False, scale=0.3, elem_classes="align-center") as image_section:
            image_display = gr.Image(label="Selected Image", type="pil")
            clear_image = gr.Button("Clear Image")

        with gr.Column(visible=False, scale=0.3, elem_classes="align-center") as audio_section:
            audio_display = gr.Audio(label="Upload/Record Audio", type="filepath")
            transcribed_display = gr.Textbox(label="Transcribed Text", interactive=False)
            clear_audio = gr.Button("Clear Audio")

    with gr.Row():
        with gr.Column(scale=0.2, elem_classes="align-center"):
            process_button = gr.Button("Process Input")

    dummy_output = gr.Textbox(visible=False)
    toast_output = gr.HTML()
    backend_toast = gr.HTML()
    refresh_button = gr.Button(" Refresh Backend Messages", visible=False)

    image_icon.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        None,
        [image_section, audio_section]
    )
    audio_icon.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        None,
        [image_section, audio_section]
    )

    clear_image.click(lambda: None, None, image_display)
    clear_audio.click(lambda: (None, ""), None, [audio_display, transcribed_display])
    audio_display.change(fn=transcribe_audio, inputs=audio_display, outputs=transcribed_display)

    process_button.click(
        fn=process_user_input,
        inputs=[image_display, text_input, transcribed_display],
        outputs=[image_display, text_input, transcribed_display, image_section, audio_section, image_input, voice_input, transcribed_text, dummy_output]
    )

    refresh_button.click(fn=refresh_messages, inputs=None, outputs=[toast_output])

    demo.load(None, None, None, js="""
        function() {
            setInterval(() => {
                const btns = Array.from(document.querySelectorAll("button"));
                const refreshBtn = btns.find(b => b.innerText.includes("Refresh Backend Messages"));
                if (refreshBtn) {
                    refreshBtn.click();
                }
            }, 2000);
        }
    """)

demo.launch()