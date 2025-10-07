import torch
import matplotlib.pyplot as plt
import numpy as np
from model import resnet18
from gradcam import GradCAM
from torchvision import transforms
from PIL import Image
import argparse

torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device( 'cpu')
parser = argparse.ArgumentParser(description='Gradcam of trained resnet model on CIFAR-100')
parser.add_argument('--imagepath', type=str,
                        help='Path to the input image')
args = parser.parse_args()

# Load model and weights
model = resnet18().to(device)
#model.load_state_dict(torch.load('resnet18_cifar100_best.pth'))
checkpoint = torch.load('resnet18_cifar100_best.pth', map_location='cpu')
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# Prepare input image (example: PIL image)
# Replace with your image loading code
img = Image.open(args.imagepath).convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
input_tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, 32, 32]

# GradCAM
target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)
cam = gradcam(input_tensor)

# Display GradCAM overlay
img_np = np.array(img.resize((24, 24))) / 255.0
plt.imshow(img_np)
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.title('Grad-CAM')
plt.axis('off')
plt.show()
