import onnxruntime as ort
import numpy as np
import gradio as gr
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# Load ONNX model
onnx_path = r"C:\Users\STIC-11\Desktop\Sk2\checkpoints\model.onnx"

if not os.path.exists(onnx_path):
    raise FileNotFoundError(f"❌ ONNX model file not found: {onnx_path}")

session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"])

# Class names
classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5353, 0.3628, 0.2486], std=[0.2126, 0.1586, 0.1401])
])


def predict(image):
    image = Image.open(image).convert("RGB")
    image = transform(image).unsqueeze(0).numpy()

    # Run inference
    outputs = session.run(None, {"input": image})
    prediction = torch.nn.functional.softmax(torch.tensor(outputs[0]), dim=1).squeeze(0)

    # Convert to dictionary format
    confidences = {classes[i]: float(prediction[i]) for i in range(len(classes))}
    
    return confidences


# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(num_top_classes=5),
    title="Diabetic Retinopathy Classifier (ONNX)",
    description="Upload a retina image, and the ONNX model will predict the severity of DR.",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()
