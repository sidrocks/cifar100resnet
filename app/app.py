import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
from model import resnet18  # Ensure this matches your model definition file

# Load CIFAR-100 class names
with open("cifar100_classes.txt") as f:
    CIFAR100_CLASSES = [line.strip() for line in f.readlines()]

# Load trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(num_classes=100)
checkpoint=torch.load("resnet18_cifar100_best.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
model.to(DEVICE)

# Define preprocessing
transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

def predict(image):
    img = Image.fromarray(image).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        class_name = CIFAR100_CLASSES[pred.item()]
        confidence = conf.item()   # Normalize to 0-100%
    return {f"{class_name}": round(confidence, 2)}

# Gradio UI
title = "CIFAR-100 Image Classifier"
description = "Upload an image (32x32 or larger). The model will predict the top class with confidence score."

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Label(num_top_classes=1, label="Prediction"),
    title=title,
    description=description,
    examples=[["examples/1.jpg"], ["examples/2.jpg"]],
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()