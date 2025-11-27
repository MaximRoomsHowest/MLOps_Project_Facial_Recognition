import gradio as gr
import requests
from PIL import Image
import io
import os

# Backend URL from environment docker-compose
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:7860")
# Your FastAPI endpoint
FASTAPI_URL = f"{BACKEND_URL}/upload/image"

def classify_image(image: Image.Image):
    # Convert PIL Image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    
    # Send image to FastAPI
    files = {"img": ("image.png", buffered, "image/png")}
    response = requests.post(FASTAPI_URL, files=files)
    
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        return f"Error: {response.status_code}"

# Create Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Predicted Emotion"),
    title="Facial Emotion Classifier",
    description="Upload an image of a face and get the predicted emotion."
)

iface.launch(server_name="0.0.0.0", server_port=7861)