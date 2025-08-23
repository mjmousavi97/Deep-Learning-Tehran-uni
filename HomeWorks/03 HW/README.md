# Deep-Learning-Tehran-uni — HW03

A compact PyTorch project comparing **three shallow CNN variants** on **MNIST**, **Fashion‑MNIST**, and **CIFAR‑10**.  
It includes reusable dataloaders, training/evaluation utilities, result plotting, and saved checkpoints for all runs.
## Reference

This project is a reimplementation/simulation of the following paper:

> Fangyuan Lei, Xun Liu, Qingyun Dai, Bingo Wing-Kuen Ling,  
> *"Shallow convolutional neural network for image classification"*,  
> Published online: 17 December 2019, Springer Nature Switzerland AG.  
> [Link to the paper](https://scispace.com/pdf/shallow-convolutional-neural-network-for-image-8twgtut756.pdf)

The main goal of this project is to reproduce the results and gain a better understanding of the proposed method.

## Highlights

- Three model families with small footprints:
  - **SCNNB** – with BatchNorm in the conv stack  
  - **SCNNB‑a** – BN after the 2nd conv and a clean `Sequential` head  
  - **SCNNB‑b** – no BatchNorm (baseline)
- Unified **training loop** with tqdm progress, accuracy tracking, and timing
- **Data pipeline**: `ToTensor` → `RandomHorizontalFlip(p=0.5)` → `Normalize(mean=0.5, std=0.5)`
- Ready‑to‑use **checkpoints** (`.pth`) for all trained models
- One notebook to run everything: `src/main.ipynb`

---

## Repository Structure

```
03 HW/
├─ checkpoints/
│  ├─ trained_model1.pth   # MNIST — SCNNB
│  ├─ trained_model2.pth   # FashionMNIST — SCNNB
│  ├─ trained_model3.pth   # CIFAR-10 — SCNNB
│  ├─ trained_model4.pth   # MNIST — SCNNB-a
│  ├─ trained_model5.pth   # FashionMNIST — SCNNB-a
│  ├─ trained_model6.pth   # CIFAR-10 — SCNNB-a
│  ├─ trained_model7.pth   # MNIST — SCNNB-b
│  ├─ trained_model8.pth   # FashionMNIST — SCNNB-b
│  └─ trained_model9.pth   # CIFAR-10 — SCNNB-b
└─ src/
   └─ main.ipynb           # End-to-end experiments (train/eval/plots)
```

> Notes  
> • Each “family” is trained once per dataset (grayscale models use `in_channels=1`, CIFAR‑10 uses `in_channels=3`).  
> • Models output raw logits; training uses `CrossEntropyLoss`.

---

## Models

All variants share the same macro‑shape:

- **Feature extractor (2× Conv3×3 + ReLU + MaxPool)**  
- **Classifier**: Flatten → Linear(…→1280) → ReLU → Dropout(0.5) → Linear(1280→num_classes)

Differences:

- **SCNNB**  
  - BN after each conv; convs use `bias=False`.  
  - Feature dim computed dynamically to keep the code input‑size agnostic.

- **SCNNB‑a**  
  - Uses a `Sequential` CNN block with BN after the **second** conv.  
  - Fully connected head in a `Sequential`.

- **SCNNB‑b**  
  - **No BatchNorm**, otherwise identical training recipe.

---

## Datasets

- **MNIST** (1×28×28), **Fashion‑MNIST** (1×28×28), **CIFAR‑10** (3×32×32)  
- Common transform for all three:
  ```python
  transforms.Compose([
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.Normalize((0.5,), (0.5,))
  ])
  ```

---

## Training Setup

- **Optimizer**: SGD(lr=0.02, momentum=0.9, weight_decay=5e‑6)  
- **Loss**: CrossEntropyLoss  
- **Batch size**: 128  
- **Epochs**:  
  - MNIST: 20  
  - Fashion‑MNIST: 30  
  - CIFAR‑10: 40  
- **Device**: auto‑select (`cuda` if available)

Core utilities:

- `train_step`, `test_step` — epoch‑level routines with tqdm progress
- `fit(...)` — end‑to‑end trainer returning loss/acc history + total time
- `accloss_plots(...)` — 4‑panel comparison (train/test × loss/accuracy)

---

## Results (Test Accuracy)

| Dataset        | SCNNB | SCNNB‑a | SCNNB‑b |
|----------------|:-----:|:-------:|:-------:|
| **MNIST**      | **0.991** | 0.990 | 0.990 |
| **Fashion‑MNIST** | 0.916 | 0.927 | **0.923** |
| **CIFAR‑10**   | **0.768** | **0.775** | **0.794**|


---

## Quickstart

### 1) Environment

```bash
git clone https://github.com/mjmousavi97/Deep-Learning-Tehran-uni.
cd Deep-Learning-Tehran-uni/HomeWorks/03\ HW

```

**Suggested `requirements.txt`:**
```
torch>=2.1
torchvision>=0.16
tqdm
matplotlib
seaborn
numpy
pillow
opencv-python  # optional; imported in the notebook
```

### 2) Run the notebook

Open `src/main.ipynb` and execute cells to:
- download datasets
- train one or more model variants
- evaluate on the test split
- save checkpoints into `checkpoints/`
- plot the learning curves (`accloss_plots`)

---

## Using the Checkpoints

Example: load a trained MNIST model and run inference on a single image.

```python
import torch
from models import SCNNB  # (or define SCNNB inline as in the notebook)

model = SCNNB(in_channels=1, num_classes=10, input_size=28)
model.load_state_dict(torch.load("checkpoints/trained_model1.pth", map_location="cpu"))
model.eval()

# x: a normalized tensor of shape (1, 1, 28, 28)
with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(dim=1).item()
print("Predicted class:", pred)
```

> Make sure the **model class definition matches** the checkpoint you load (SCNNB / SCNNB‑a / SCNNB‑b and the correct `in_channels`/`input_size`).

---

## Reproducibility Notes

- Random seeds are **not** fixed in the current notebook; minor variations are expected.  
- Horizontal flip is applied to **all** datasets for consistency (including MNIST/Fashion‑MNIST).  
- Feature dimensions are computed dynamically — the same code works for 28×28 and 32×32 inputs.

---

## License

Add your preferred license (e.g., MIT) at the repository root.

---

