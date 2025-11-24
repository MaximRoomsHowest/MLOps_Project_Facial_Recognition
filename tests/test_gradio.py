import gradio as gr
from PIL import Image
import numpy as np

def predict(image: Image.Image):
    img_array = np.array(image)
    return f"Image shape: {img_array.shape}"

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text"
)

iface.launch()