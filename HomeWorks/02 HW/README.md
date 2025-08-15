# Fashion-MNIST CNN (PyTorch + Colab)

A clean, end-to-end **Convolutional Neural Network** pipeline for **Fashion-MNIST** using PyTorch.  
This project shows best practices for dataset handling, model design with BatchNorm/Dropout, a tidy training/evaluation loop with `tqdm`, and clear visualization of results (loss/accuracy curves and a confusion matrix).

> **Highlights**
> - Minimal, well-documented CNN (2×Conv + BN + ReLU + MaxPool ➜ Dropout + Linear)
> - Reproducible training loop with progress bars
> - Evaluation utilities (accuracy, confusion matrix, sample predictions)
> - Colab-friendly (includes Google Drive mounting notes) or local run

---

## 📦 Dataset

We use **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)** (60,000 train / 10,000 test, grayscale 28×28, 10 classes).  
Images are normalized to mean `0.5` and std `0.5`.

Default class names used in this project:

```
["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

> **Note:** In the notebook, an alternative path layout (`DIR_TRAIN`, `DIR_TEST`) is shown for Drive-based datasets. This project ultimately trains from `torchvision.datasets.FashionMNIST` (auto-download) with transforms.

---

## 🗂️ Project Structure

```
.
├─ advanced_pytorch.ipynb         # Main code                          
├─ saved_model                    # model weights (created by torch.save)
├─ README.md                      # this file
```

---

## 🧪 Environment & Requirements

- Python 3.8+
- PyTorch, Torchvision, TQDM, NumPy, Matplotlib, Seaborn, scikit-learn
- (Optional) OpenCV & Pillow (used for general image utility; not required to train)

Install (local):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or cpu wheels
pip install tqdm numpy matplotlib seaborn scikit-learn pillow opencv-python
```

Google Colab users typically have most dependencies preinstalled.

---

## ⚙️ Data Loading & Transforms

```python
Transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

train_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=True, download=True, transform=Transforms
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=False, download=True, transform=Transforms
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)
```

---

## 🧱 Model Architecture

A compact CNN with two 3×3 convolutions (same channels), BatchNorm, ReLU, and a 2×2 MaxPool. The classifier is a Dropout(0.5) + Linear head.

**Input:** `(N, 1, 28, 28)`  
**Feature extractor:**  
`Conv(1→10, 3×3, pad=1)` → `BN(10)` → `ReLU` → `Conv(10→10, 3×3, pad=1)` → `BN(10)` → `ReLU` → `MaxPool(2×2)`  
**Classifier:**  
`Flatten` → `Dropout(0.5)` → `Linear(10×14×14 → 10)`

```python
class CNN_Model(nn.Module):
    def __init__(self, input_channel=1, hidden_unit=10, output_shape=10):
        super().__init__()
        self.CNN_block = nn.Sequential(
            nn.Conv2d(input_channel, hidden_unit, 3, 1, 1),
            nn.BatchNorm2d(hidden_unit),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_unit, hidden_unit, 3, 1, 1),
            nn.BatchNorm2d(hidden_unit),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_unit * 14 * 14, output_shape)
        )

    def forward(self, x):
        x = self.CNN_block(x)
        return self.classifier(x)
```

---

## 🚀 Training & Evaluation

**Loss/Optimizer/Device**
```python
loss_fn  = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**Training loop** (with `tqdm` progress-bars)
- Tracks **average loss** and **accuracy** per epoch
- Properly zeros grads, backprops, and steps optimizer
- Uses `model.train()` for training, `model.eval()` for testing

**Accuracy function**
```python
def accuracy_fn(true, pred_logits):
    pred = F.softmax(pred_logits, dim=1).argmax(dim=1)
    return (pred == true).float().mean()
```

**Results (20 epochs, batch=32, lr=1e-4)**

- **Train Acc:** ↑ from ~0.77 to **~0.94**
- **Test Acc:** peaks around **~0.909** (20th epoch: `0.9086`)
- **Generalization:** stable; light overfitting controlled by BN + Dropout

> *Your exact numbers may vary depending on the runtime seed/hardware.*

**Saving the model**
```python
torch.save(model.state_dict(), "saved_model")
```

---

## 📊 Visualizations

- **Loss & Accuracy curves** over epochs (Matplotlib + Seaborn)
- **Confusion Matrix** (`sklearn.metrics.confusion_matrix` + Seaborn heatmap)

Example of confusion you may observe (typical for Fashion-MNIST):
- **Shirt ↔ T-shirt/top** mix-ups
- **Pullover/Coat** overlaps
- Strong classes like **Sandal**, **Sneaker**, **Ankle boot** usually classify well

---

## 🔎 Sample Inference

```python
model.eval()
sample, label = test_dataset[0]          # (1, 28, 28)
with torch.no_grad():
    logits = model(sample.unsqueeze(0).to(device))
pred = logits.softmax(dim=1).argmax(dim=1).item()
print("True:", label, "Pred:", pred)
```

---



## 🛠️ Extending This Baseline

- Increase `hidden_unit` (e.g., 32/64) or add an extra conv block
- Add data augmentation (RandomCrop, RandomHorizontalFlip)
- Try schedulers (e.g., `StepLR`, `CosineAnnealingLR`)
- Swap to **SGD + momentum** for comparison
- Experiment with **weight decay** and different `Dropout` rates
- Try a deeper head (e.g., `Linear → ReLU → Linear`)

---

## 📝 Notes on Google Drive

If you keep a custom dataset on Drive (e.g., class folders per label), mount and point to the directories:
```python
from google.colab import drive
drive.mount('/content/drive')

DIR_TRAIN = "/content/drive/MyDrive/fashionmnist/train/"
DIR_TEST  = "/content/drive/MyDrive/fashionmnist/test/"
# Then use ImageFolder with appropriate transforms
```
This project *ultimately* trains using `torchvision.datasets.FashionMNIST` for simplicity and reproducibility.

---

## 📄 Citation

- **Fashion-MNIST**: Han Xiao, Kashif Rasul, Roland Vollgraf.  
  *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.*  
  https://github.com/zalandoresearch/fashion-mnist

---

## 📜 License

This project is released under the **MIT License** (feel free to adapt).

---

## 🙌 Acknowledgements

- PyTorch & TorchVision teams
- `tqdm`, `scikit-learn`, `seaborn`, and the open-source community
- You, for training models and sharing results 🎉
