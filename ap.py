import onnxruntime as ort
import numpy as np
import gradio as gr
from PIL import Image
import os
device = "cuda" if ort.get_device() == "GPU" else "cpu"
print(f"Using device: {device}")


onnx_path = r"/Users/macm1/Desktop/HUB/dr-onnx/dr-model.onnx"

if not os.path.exists(onnx_path):
    raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

# Load ONNX Model
session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"])


classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Mean and Std for Normalization
mean = np.array([0.5353, 0.3628, 0.2486], dtype=np.float32)
std = np.array([0.2126, 0.1586, 0.1401], dtype=np.float32)

# Preprocessing Function
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize to 224x224
    image = np.array(image, dtype=np.float32) / 255.0  # Convert to float32 and normalize
    image = (image - mean) / std  # Apply normalization
    image = np.transpose(image, (2, 0, 1))  # Change shape to (C, H, W)
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
    return image

# Prediction Function
def predict(image_path):
    image = preprocess(image_path)
    outputs = session.run(None, {"input": image})[0]  # Run inference

    # Apply Softmax using NumPy
    exp_outputs = np.exp(outputs - np.max(outputs))  # Stability trick
    prediction = exp_outputs / np.sum(exp_outputs, axis=-1, keepdims=True)
    
    # Convert to dictionary format
    confidences = {classes[i]: float(prediction[0, i]) for i in range(len(classes))}
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
