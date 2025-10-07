# ResNet CIFAR-100 Classifier (PyTorch)

This repository provides a modular pipeline for training, evaluating, and visualizing a ResNet model on the CIFAR-100 dataset using PyTorch.

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
resnet-cifar100/
├── main.py          # Complete all-in-one training script
├── train.py         # Modular training script (with resume functionality)
├── evalgradcam.py   # Grad-CAM visualizationn script
├── gradcam.py       # Grad-CAM utility
├── model.py         # ResNet model definition
├── requirements.txt #Python dependencies
└── README.md        # README

```


## 🏋️ Training (`train.py`)

Train a ResNet (default: ResNet-18) on CIFAR-100.

**Usage:**
```bash
python train.py --epochs 50 --batch-size 128 --lr 0.1
```

**Arguments:**
- `--epochs`: Number of epochs (default: 30)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.1)
- `--resume`: Path to checkpoint to resume training

Checkpoints and best models are saved automatically.

---

## 🏗️ Model Architecture (`model.py`)

Defines ResNet architectures for CIFAR-100:
- `resnet18`
- `resnet34`
- `resnet50`

Includes modular blocks (`BasicBlock`, `Bottleneck`) for easy extension.

---

## 🔥 Grad-CAM Visualization (`gradcam.py`, `evalgradcam.py`)

Visualize model attention using Grad-CAM.

**Usage:**
```bash
python evalgradcam.py --imagepath path/to/image.png
```

- Loads a trained model and overlays Grad-CAM heatmap on the input image.
- Automatically uses GPU if available.

---

## 🧪 Evaluation

Evaluate model accuracy and visualize Grad-CAM for any image.

---

## 📊 Results

Typical performance after 50 epochs:
- **Best Test Accuracy**: ~73.53%
- **Training Time**: ~20 minutes (on GPU)
- **Best Model**: Saved as `resnet18_cifar100_best.pth`

---

## 📝 Notes

- The CIFAR-100 dataset is downloaded automatically.
- For best performance, use a CUDA-enabled GPU.
- You can extend data augmentation and regularization in `train.py` for better generalization.

---

## 📄 License

MIT License

### Model Training Logs

https://github.com/sidrocks/cifar100resnet/blob/main/traininglog.md
