from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
import numpy as np
from PIL import Image
import os

# Labels for your model
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
# Number of classes in your classification
num_classes = 5  # Adjust to your dataset

class ResNetWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights=None)  # Don't load pretrained here unless you want
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
# Path to the model inside the container
model_path = os.path.join("model", "facial_expression_model.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNetWrapper(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()  # Important for inference

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/upload/image')
async def upload_image(img: UploadFile = File(...)):
    # Load image
    original_image = Image.open(img.file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    resized_image = transform(original_image).unsqueeze(0)
    img_tensor = resized_image.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return {"prediction": EMOTIONS[predicted_class]}