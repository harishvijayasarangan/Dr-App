import torch
import torchvision.models as models
import os
model = models.densenet121(weights=None)
model.classifier = torch.nn.Linear(1024, 5)

model_path = r"C:\Users\STIC-11\Desktop\Sk2\checkpoints\best_model.ckpt"

if not os.path.exists(model_path):
    raise FileNotFoundError(f" Model file not found: {model_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# Dummy input tensor for ONNX export
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Export to ONNX format
onnx_path = r"C:\Users\STIC-11\Desktop\Sk2\checkpoints\model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"]
)

print(f" Model successfully converted to ONNX and saved at: {onnx_path}")
