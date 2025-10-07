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


        Epoch: 1/50
        Epoch: 1 | Batch: 0/391 | Loss: 4.686
        Epoch: 1 | Batch: 100/391 | Loss: 4.223
        Epoch: 1 | Batch: 200/391 | Loss: 3.926
        Epoch: 1 | Batch: 300/391 | Loss: 3.763
        Train Loss: 3.955 | Train Acc: 9.03%
        Test Loss: 3.636 | Test Acc: 13.80% | LR: 0.100000
        New best model saved with accuracy: 13.80%

        Epoch: 2/50
        Epoch: 2 | Batch: 0/391 | Loss: 3.428
        Epoch: 2 | Batch: 100/391 | Loss: 3.540
        Epoch: 2 | Batch: 200/391 | Loss: 3.359
        Epoch: 2 | Batch: 300/391 | Loss: 3.054
        Train Loss: 3.372 | Train Acc: 17.66%
        Test Loss: 3.227 | Test Acc: 21.22% | LR: 0.100000
        New best model saved with accuracy: 21.22%

        Epoch: 3/50
        Epoch: 3 | Batch: 0/391 | Loss: 3.153
        Epoch: 3 | Batch: 100/391 | Loss: 3.132
        Epoch: 3 | Batch: 200/391 | Loss: 2.905
        Epoch: 3 | Batch: 300/391 | Loss: 2.801
        Train Loss: 2.917 | Train Acc: 25.84%
        Test Loss: 2.859 | Test Acc: 27.85% | LR: 0.100000
        New best model saved with accuracy: 27.85%

        Epoch: 4/50
        Epoch: 4 | Batch: 0/391 | Loss: 2.527
        Epoch: 4 | Batch: 100/391 | Loss: 2.429
        Epoch: 4 | Batch: 200/391 | Loss: 2.606
        Epoch: 4 | Batch: 300/391 | Loss: 2.252
        Train Loss: 2.476 | Train Acc: 34.67%
        Test Loss: 2.476 | Test Acc: 35.40% | LR: 0.100000
        New best model saved with accuracy: 35.40%

        Epoch: 5/50
        Epoch: 5 | Batch: 0/391 | Loss: 2.365
        Epoch: 5 | Batch: 100/391 | Loss: 2.338
        Epoch: 5 | Batch: 200/391 | Loss: 2.192
        Epoch: 5 | Batch: 300/391 | Loss: 1.919
        Train Loss: 2.141 | Train Acc: 41.82%
        Test Loss: 2.077 | Test Acc: 43.99% | LR: 0.100000
        New best model saved with accuracy: 43.99%

        Epoch: 6/50
        Epoch: 6 | Batch: 0/391 | Loss: 1.663
        Epoch: 6 | Batch: 100/391 | Loss: 1.802
        Epoch: 6 | Batch: 200/391 | Loss: 1.917
        Epoch: 6 | Batch: 300/391 | Loss: 1.598
        Train Loss: 1.915 | Train Acc: 47.27%
        Test Loss: 2.374 | Test Acc: 40.09% | LR: 0.100000

        Epoch: 7/50
        Epoch: 7 | Batch: 0/391 | Loss: 2.084
        Epoch: 7 | Batch: 100/391 | Loss: 1.911
        Epoch: 7 | Batch: 200/391 | Loss: 1.817
        Epoch: 7 | Batch: 300/391 | Loss: 1.833
        Train Loss: 1.769 | Train Acc: 50.81%
        Test Loss: 1.960 | Test Acc: 46.80% | LR: 0.100000
        New best model saved with accuracy: 46.80%

        Epoch: 8/50
        Epoch: 8 | Batch: 0/391 | Loss: 1.467
        Epoch: 8 | Batch: 100/391 | Loss: 1.672
        Epoch: 8 | Batch: 200/391 | Loss: 1.583
        Epoch: 8 | Batch: 300/391 | Loss: 1.795
        Train Loss: 1.649 | Train Acc: 53.67%
        Test Loss: 1.813 | Test Acc: 50.13% | LR: 0.100000
        New best model saved with accuracy: 50.13%

        Epoch: 9/50
        Epoch: 9 | Batch: 0/391 | Loss: 1.701
        Epoch: 9 | Batch: 100/391 | Loss: 1.697
        Epoch: 9 | Batch: 200/391 | Loss: 1.855
        Epoch: 9 | Batch: 300/391 | Loss: 1.436
        Train Loss: 1.570 | Train Acc: 55.83%
        Test Loss: 1.788 | Test Acc: 51.17% | LR: 0.100000
        New best model saved with accuracy: 51.17%

        Epoch: 10/50
        Epoch: 10 | Batch: 0/391 | Loss: 1.527
        Epoch: 10 | Batch: 100/391 | Loss: 1.579
        Epoch: 10 | Batch: 200/391 | Loss: 1.348
        Epoch: 10 | Batch: 300/391 | Loss: 1.528
        Train Loss: 1.500 | Train Acc: 57.26%
        Test Loss: 1.737 | Test Acc: 52.53% | LR: 0.100000
        New best model saved with accuracy: 52.53%

        Epoch: 11/50
        Epoch: 11 | Batch: 0/391 | Loss: 1.306
        Epoch: 11 | Batch: 100/391 | Loss: 1.498
        Epoch: 11 | Batch: 200/391 | Loss: 1.894
        Epoch: 11 | Batch: 300/391 | Loss: 1.053
        Train Loss: 1.454 | Train Acc: 58.76%
        Test Loss: 1.967 | Test Acc: 48.49% | LR: 0.100000

        Epoch: 12/50
        Epoch: 12 | Batch: 0/391 | Loss: 1.293
        Epoch: 12 | Batch: 100/391 | Loss: 1.488
        Epoch: 12 | Batch: 200/391 | Loss: 1.256
        Epoch: 12 | Batch: 300/391 | Loss: 1.457
        Train Loss: 1.401 | Train Acc: 60.05%
        Test Loss: 1.732 | Test Acc: 53.06% | LR: 0.100000
        New best model saved with accuracy: 53.06%

        Epoch: 13/50
        Epoch: 13 | Batch: 0/391 | Loss: 1.367
        Epoch: 13 | Batch: 100/391 | Loss: 1.585
        Epoch: 13 | Batch: 200/391 | Loss: 1.402
        Epoch: 13 | Batch: 300/391 | Loss: 1.383
        Train Loss: 1.363 | Train Acc: 60.74%
        Test Loss: 1.884 | Test Acc: 51.21% | LR: 0.100000

        Epoch: 14/50
        Epoch: 14 | Batch: 0/391 | Loss: 1.318
        Epoch: 14 | Batch: 100/391 | Loss: 1.100
        Epoch: 14 | Batch: 200/391 | Loss: 1.292
        Epoch: 14 | Batch: 300/391 | Loss: 1.429
        Train Loss: 1.332 | Train Acc: 61.99%
        Test Loss: 1.815 | Test Acc: 51.60% | LR: 0.100000

        Epoch: 15/50
        Epoch: 15 | Batch: 0/391 | Loss: 1.336
        Epoch: 15 | Batch: 100/391 | Loss: 1.096
        Epoch: 15 | Batch: 200/391 | Loss: 1.253
        Epoch: 15 | Batch: 300/391 | Loss: 1.152
        Train Loss: 1.302 | Train Acc: 62.78%
        Test Loss: 1.591 | Test Acc: 56.04% | LR: 0.010000
        New best model saved with accuracy: 56.04%

        Epoch: 16/50
        Epoch: 16 | Batch: 0/391 | Loss: 1.120
        Epoch: 16 | Batch: 100/391 | Loss: 0.805
        Epoch: 16 | Batch: 200/391 | Loss: 0.939
        Epoch: 16 | Batch: 300/391 | Loss: 0.659
        Train Loss: 0.865 | Train Acc: 75.06%
        Test Loss: 1.058 | Test Acc: 69.33% | LR: 0.010000
        New best model saved with accuracy: 69.33%

        Epoch: 17/50
        Epoch: 17 | Batch: 0/391 | Loss: 0.571
        Epoch: 17 | Batch: 100/391 | Loss: 0.667
        Epoch: 17 | Batch: 200/391 | Loss: 0.676
        Epoch: 17 | Batch: 300/391 | Loss: 0.726
        Train Loss: 0.719 | Train Acc: 78.75%
        Test Loss: 1.007 | Test Acc: 71.04% | LR: 0.010000
        New best model saved with accuracy: 71.04%

        Epoch: 18/50
        Epoch: 18 | Batch: 0/391 | Loss: 0.669
        Epoch: 18 | Batch: 100/391 | Loss: 0.683
        Epoch: 18 | Batch: 200/391 | Loss: 0.380
        Epoch: 18 | Batch: 300/391 | Loss: 0.710
        Train Loss: 0.643 | Train Acc: 80.83%
        Test Loss: 1.001 | Test Acc: 71.22% | LR: 0.010000
        New best model saved with accuracy: 71.22%

        Epoch: 19/50
        Epoch: 19 | Batch: 0/391 | Loss: 0.579
        Epoch: 19 | Batch: 100/391 | Loss: 0.583
        Epoch: 19 | Batch: 200/391 | Loss: 0.649
        Epoch: 19 | Batch: 300/391 | Loss: 0.655
        Train Loss: 0.601 | Train Acc: 81.99%
        Test Loss: 1.001 | Test Acc: 71.77% | LR: 0.010000
        New best model saved with accuracy: 71.77%

        Epoch: 20/50
        Epoch: 20 | Batch: 0/391 | Loss: 0.645
        Epoch: 20 | Batch: 100/391 | Loss: 0.575
        Epoch: 20 | Batch: 200/391 | Loss: 0.590
        Epoch: 20 | Batch: 300/391 | Loss: 0.511
        Train Loss: 0.547 | Train Acc: 83.55%
        Test Loss: 0.990 | Test Acc: 71.83% | LR: 0.010000
        New best model saved with accuracy: 71.83%

        Epoch: 21/50
        Epoch: 21 | Batch: 0/391 | Loss: 0.480
        Epoch: 21 | Batch: 100/391 | Loss: 0.463
        Epoch: 21 | Batch: 200/391 | Loss: 0.520
        Epoch: 21 | Batch: 300/391 | Loss: 0.607
        Train Loss: 0.506 | Train Acc: 84.90%
        Test Loss: 1.006 | Test Acc: 71.98% | LR: 0.010000
        New best model saved with accuracy: 71.98%

        Epoch: 22/50
        Epoch: 22 | Batch: 0/391 | Loss: 0.382
        Epoch: 22 | Batch: 100/391 | Loss: 0.453
        Epoch: 22 | Batch: 200/391 | Loss: 0.577
        Epoch: 22 | Batch: 300/391 | Loss: 0.529
        Train Loss: 0.466 | Train Acc: 86.03%
        Test Loss: 1.000 | Test Acc: 71.91% | LR: 0.010000

        Epoch: 23/50
        Epoch: 23 | Batch: 0/391 | Loss: 0.385
        Epoch: 23 | Batch: 100/391 | Loss: 0.367
        Epoch: 23 | Batch: 200/391 | Loss: 0.496
        Epoch: 23 | Batch: 300/391 | Loss: 0.400
        Train Loss: 0.433 | Train Acc: 87.04%
        Test Loss: 1.026 | Test Acc: 71.54% | LR: 0.010000

        Epoch: 24/50
        Epoch: 24 | Batch: 0/391 | Loss: 0.340
        Epoch: 24 | Batch: 100/391 | Loss: 0.336
        Epoch: 24 | Batch: 200/391 | Loss: 0.418
        Epoch: 24 | Batch: 300/391 | Loss: 0.415
        Train Loss: 0.392 | Train Acc: 88.18%
        Test Loss: 1.043 | Test Acc: 71.76% | LR: 0.010000

        Epoch: 25/50
        Epoch: 25 | Batch: 0/391 | Loss: 0.310
        Epoch: 25 | Batch: 100/391 | Loss: 0.352
        Epoch: 25 | Batch: 200/391 | Loss: 0.348
        Epoch: 25 | Batch: 300/391 | Loss: 0.330
        Train Loss: 0.364 | Train Acc: 89.21%
        Test Loss: 1.050 | Test Acc: 71.66% | LR: 0.001000

        Epoch: 26/50
        Epoch: 26 | Batch: 0/391 | Loss: 0.386
        Epoch: 26 | Batch: 100/391 | Loss: 0.280
        Epoch: 26 | Batch: 200/391 | Loss: 0.198
        Epoch: 26 | Batch: 300/391 | Loss: 0.231
        Train Loss: 0.282 | Train Acc: 92.11%
        Test Loss: 0.988 | Test Acc: 73.05% | LR: 0.001000
        New best model saved with accuracy: 73.05%

        Epoch: 27/50
        Epoch: 27 | Batch: 0/391 | Loss: 0.226
        Epoch: 27 | Batch: 100/391 | Loss: 0.265
        Epoch: 27 | Batch: 200/391 | Loss: 0.218
        Epoch: 27 | Batch: 300/391 | Loss: 0.244
        Train Loss: 0.258 | Train Acc: 93.04%
        Test Loss: 0.985 | Test Acc: 73.26% | LR: 0.001000
        New best model saved with accuracy: 73.26%

        Epoch: 28/50
        Epoch: 28 | Batch: 0/391 | Loss: 0.199
        Epoch: 28 | Batch: 100/391 | Loss: 0.228
        Epoch: 28 | Batch: 200/391 | Loss: 0.330
        Epoch: 28 | Batch: 300/391 | Loss: 0.250
        Train Loss: 0.246 | Train Acc: 93.46%
        Test Loss: 0.986 | Test Acc: 73.37% | LR: 0.001000
        New best model saved with accuracy: 73.37%

        Epoch: 29/50
        Epoch: 29 | Batch: 0/391 | Loss: 0.175
        Epoch: 29 | Batch: 100/391 | Loss: 0.230
        Epoch: 29 | Batch: 200/391 | Loss: 0.196
        Epoch: 29 | Batch: 300/391 | Loss: 0.228
        Train Loss: 0.235 | Train Acc: 93.96%
        Test Loss: 0.991 | Test Acc: 73.41% | LR: 0.001000
        New best model saved with accuracy: 73.41%

        Epoch: 30/50
        Epoch: 30 | Batch: 0/391 | Loss: 0.222
        Epoch: 30 | Batch: 100/391 | Loss: 0.253
        Epoch: 30 | Batch: 200/391 | Loss: 0.239
        Epoch: 30 | Batch: 300/391 | Loss: 0.279
        Train Loss: 0.228 | Train Acc: 94.12%
        Test Loss: 0.994 | Test Acc: 73.33% | LR: 0.001000

        Epoch: 31/50
        Epoch: 31 | Batch: 0/391 | Loss: 0.208
        Epoch: 31 | Batch: 100/391 | Loss: 0.281
        Epoch: 31 | Batch: 200/391 | Loss: 0.199
        Epoch: 31 | Batch: 300/391 | Loss: 0.276
        Train Loss: 0.222 | Train Acc: 94.24%
        Test Loss: 1.001 | Test Acc: 73.23% | LR: 0.001000

        Epoch: 32/50
        Epoch: 32 | Batch: 0/391 | Loss: 0.171
        Epoch: 32 | Batch: 100/391 | Loss: 0.199
        Epoch: 32 | Batch: 200/391 | Loss: 0.240
        Epoch: 32 | Batch: 300/391 | Loss: 0.179
        Train Loss: 0.217 | Train Acc: 94.54%
        Test Loss: 0.997 | Test Acc: 73.26% | LR: 0.001000

        Epoch: 33/50
        Epoch: 33 | Batch: 0/391 | Loss: 0.292
        Epoch: 33 | Batch: 100/391 | Loss: 0.176
        Epoch: 33 | Batch: 200/391 | Loss: 0.175
        Epoch: 33 | Batch: 300/391 | Loss: 0.203
        Train Loss: 0.208 | Train Acc: 94.80%
        Test Loss: 0.999 | Test Acc: 73.49% | LR: 0.001000
        New best model saved with accuracy: 73.49%

        Epoch: 34/50
        Epoch: 34 | Batch: 0/391 | Loss: 0.170
        Epoch: 34 | Batch: 100/391 | Loss: 0.242
        Epoch: 34 | Batch: 200/391 | Loss: 0.131
        Epoch: 34 | Batch: 300/391 | Loss: 0.249
        Train Loss: 0.203 | Train Acc: 94.90%
        Test Loss: 1.005 | Test Acc: 73.51% | LR: 0.001000
        New best model saved with accuracy: 73.51%

        Epoch: 35/50
        Epoch: 35 | Batch: 0/391 | Loss: 0.287
        Epoch: 35 | Batch: 100/391 | Loss: 0.164
        Epoch: 35 | Batch: 200/391 | Loss: 0.200
        Epoch: 35 | Batch: 300/391 | Loss: 0.207
        Train Loss: 0.197 | Train Acc: 95.24%
        Test Loss: 1.004 | Test Acc: 73.48% | LR: 0.001000

        Epoch: 36/50
        Epoch: 36 | Batch: 0/391 | Loss: 0.252
        Epoch: 36 | Batch: 100/391 | Loss: 0.257
        Epoch: 36 | Batch: 200/391 | Loss: 0.119
        Epoch: 36 | Batch: 300/391 | Loss: 0.212
        Train Loss: 0.195 | Train Acc: 95.17%
        Test Loss: 1.005 | Test Acc: 73.41% | LR: 0.001000

        Epoch: 37/50
        Epoch: 37 | Batch: 0/391 | Loss: 0.185
        Epoch: 37 | Batch: 100/391 | Loss: 0.165
        Epoch: 37 | Batch: 200/391 | Loss: 0.143
        Epoch: 37 | Batch: 300/391 | Loss: 0.206
        Train Loss: 0.187 | Train Acc: 95.54%
        Test Loss: 1.014 | Test Acc: 73.26% | LR: 0.001000

        Epoch: 38/50
        Epoch: 38 | Batch: 0/391 | Loss: 0.169
        Epoch: 38 | Batch: 100/391 | Loss: 0.220
        Epoch: 38 | Batch: 200/391 | Loss: 0.150
        Epoch: 38 | Batch: 300/391 | Loss: 0.123
        Train Loss: 0.184 | Train Acc: 95.59%
        Test Loss: 1.008 | Test Acc: 73.30% | LR: 0.001000

        Epoch: 39/50
        Epoch: 39 | Batch: 0/391 | Loss: 0.214
        Epoch: 39 | Batch: 100/391 | Loss: 0.192
        Epoch: 39 | Batch: 200/391 | Loss: 0.229
        Epoch: 39 | Batch: 300/391 | Loss: 0.197
        Train Loss: 0.183 | Train Acc: 95.61%
        Test Loss: 1.008 | Test Acc: 73.53% | LR: 0.001000
        New best model saved with accuracy: 73.53%

        Epoch: 40/50
        Epoch: 40 | Batch: 0/391 | Loss: 0.137
        Epoch: 40 | Batch: 100/391 | Loss: 0.130
        Epoch: 40 | Batch: 200/391 | Loss: 0.145
        Epoch: 40 | Batch: 300/391 | Loss: 0.200
        Train Loss: 0.175 | Train Acc: 95.94%
        Test Loss: 1.012 | Test Acc: 73.48% | LR: 0.001000

        Epoch: 41/50
        Epoch: 41 | Batch: 0/391 | Loss: 0.144
        Epoch: 41 | Batch: 100/391 | Loss: 0.138
        Epoch: 41 | Batch: 200/391 | Loss: 0.137
        Epoch: 41 | Batch: 300/391 | Loss: 0.142
        Train Loss: 0.172 | Train Acc: 96.06%
        Test Loss: 1.011 | Test Acc: 73.37% | LR: 0.001000

        Epoch: 42/50
        Epoch: 42 | Batch: 0/391 | Loss: 0.207
        Epoch: 42 | Batch: 100/391 | Loss: 0.179
        Epoch: 42 | Batch: 200/391 | Loss: 0.152
        Epoch: 42 | Batch: 300/391 | Loss: 0.152
        Train Loss: 0.167 | Train Acc: 96.13%
        Test Loss: 1.016 | Test Acc: 73.44% | LR: 0.001000

        Epoch: 43/50
        Epoch: 43 | Batch: 0/391 | Loss: 0.110
        Epoch: 43 | Batch: 100/391 | Loss: 0.135
        Epoch: 43 | Batch: 200/391 | Loss: 0.147
        Epoch: 43 | Batch: 300/391 | Loss: 0.114
        Train Loss: 0.162 | Train Acc: 96.39%
        Test Loss: 1.020 | Test Acc: 73.32% | LR: 0.001000

        Epoch: 44/50
        Epoch: 44 | Batch: 0/391 | Loss: 0.090
        Epoch: 44 | Batch: 100/391 | Loss: 0.247
        Epoch: 44 | Batch: 200/391 | Loss: 0.178
        Epoch: 44 | Batch: 300/391 | Loss: 0.140
        Train Loss: 0.157 | Train Acc: 96.44%
        Test Loss: 1.023 | Test Acc: 73.29% | LR: 0.001000

        Epoch: 45/50
        Epoch: 45 | Batch: 0/391 | Loss: 0.104
        Epoch: 45 | Batch: 100/391 | Loss: 0.154
        Epoch: 45 | Batch: 200/391 | Loss: 0.158
        Epoch: 45 | Batch: 300/391 | Loss: 0.166
        Train Loss: 0.153 | Train Acc: 96.58%
        Test Loss: 1.030 | Test Acc: 73.23% | LR: 0.001000

        Epoch: 46/50
        Epoch: 46 | Batch: 0/391 | Loss: 0.159
        Epoch: 46 | Batch: 100/391 | Loss: 0.146
        Epoch: 46 | Batch: 200/391 | Loss: 0.148
        Epoch: 46 | Batch: 300/391 | Loss: 0.088
        Train Loss: 0.151 | Train Acc: 96.61%
        Test Loss: 1.029 | Test Acc: 73.37% | LR: 0.001000

        Epoch: 47/50
        Epoch: 47 | Batch: 0/391 | Loss: 0.160
        Epoch: 47 | Batch: 100/391 | Loss: 0.104
        Epoch: 47 | Batch: 200/391 | Loss: 0.155
        Epoch: 47 | Batch: 300/391 | Loss: 0.140
        Train Loss: 0.147 | Train Acc: 96.84%
        Test Loss: 1.031 | Test Acc: 73.26% | LR: 0.001000

        Epoch: 48/50
        Epoch: 48 | Batch: 0/391 | Loss: 0.164
        Epoch: 48 | Batch: 100/391 | Loss: 0.149
        Epoch: 48 | Batch: 200/391 | Loss: 0.109
        Epoch: 48 | Batch: 300/391 | Loss: 0.165
        Train Loss: 0.144 | Train Acc: 96.90%
        Test Loss: 1.032 | Test Acc: 73.39% | LR: 0.001000

        Epoch: 49/50
        Epoch: 49 | Batch: 0/391 | Loss: 0.186
        Epoch: 49 | Batch: 100/391 | Loss: 0.166
        Epoch: 49 | Batch: 200/391 | Loss: 0.137
        Epoch: 49 | Batch: 300/391 | Loss: 0.100
        Train Loss: 0.138 | Train Acc: 97.19%
        Test Loss: 1.034 | Test Acc: 73.39% | LR: 0.001000

        Epoch: 50/50
        Epoch: 50 | Batch: 0/391 | Loss: 0.183
        Epoch: 50 | Batch: 100/391 | Loss: 0.100
        Epoch: 50 | Batch: 200/391 | Loss: 0.117
        Epoch: 50 | Batch: 300/391 | Loss: 0.177
        Train Loss: 0.135 | Train Acc: 97.10%
        Test Loss: 1.041 | Test Acc: 73.37% | LR: 0.001000

        Training completed in 19.31 minutes
        Best test accuracy: 73.53%


