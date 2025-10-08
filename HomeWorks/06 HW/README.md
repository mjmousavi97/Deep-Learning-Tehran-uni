# 🧠 COVID-CT Classification using Custom ResNet

This project implements a **ResNet-based Convolutional Neural Network** from scratch in **PyTorch** to classify **CT scan images** as **COVID** or **Non-COVID**.  
It includes data preprocessing, augmentation, model definition, training, and evaluation — all within **Google Colab**.

---

## 🚀 Features
- Custom **ResNet** architecture with Bottleneck blocks (ResNet-50 style)
- Image augmentation (rotation, flipping, noise addition, normalization)
- Dataset loading via `torchvision.datasets.ImageFolder`
- Automatic train/validation/test splitting
- Training loop with accuracy, precision, and F1-score tracking
- GPU (CUDA) support
- Model weights saved in **`checkpoints/`** folder
- Ready-to-run on **Google Colab**

---

## 📂 Dataset Structure

Before running, ensure your dataset is structured as follows:

```
data/
│
├── CT_COVID/
│   ├── img001.png
│   ├── img002.png
│   └── ...
│
└── CT_NonCOVID/
    ├── img001.png
    ├── img002.png
    └── ...
```

Each class should be stored in its own folder.

### 📎 Dataset Link
Download dataset from GitHub:  
🔗 [UCSD-AI4H COVID-CT Dataset](https://github.com/desaisrkr/https-github.com-UCSD-AI4H-COVID-CT/tree/master)

---

## 🧰 Requirements

This code runs on **Google Colab** or any environment with PyTorch installed.

### Python Libraries
```bash
torch
torchvision
numpy
matplotlib
scikit-learn
```

### Optional (for local setup)
```bash
pip install torch torchvision matplotlib scikit-learn
```

---

## 📦 Setup and Data Preparation

Upload your `.zip` files containing the datasets (`CT_COVID.zip`, `CT_NonCOVID.zip`) to Colab:

```python
from google.colab import files
files.upload()
```

Then extract them:
```python
import zipfile, shutil

with zipfile.ZipFile('CT_COVID.zip', 'r') as covid_zip:
    covid_zip.extractall('/content/data')

with zipfile.ZipFile('CT_NonCOVID.zip', 'r') as noncovid_zip:
    noncovid_zip.extractall('/content/data')

# Remove unnecessary folders (MacOS artifacts)
shutil.rmtree('/content/data/__MACOSX', ignore_errors=True)
```

---

## 🔧 Data Loading and Augmentation

We use a `torchvision.transforms.Compose` pipeline that includes:
- Resizing to 256×256  
- Adding Gaussian noise  
- Random rotation and horizontal flip  
- Normalization

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),
    transforms.RandomRotation((-20, 20)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

---

## 🧠 Model Architecture

The model follows a **ResNet-50** style structure implemented from scratch:
- Bottleneck residual blocks with expansion factor = 4  
- 4 main layers with `[3, 4, 6, 3]` blocks respectively  
- Global average pooling and a fully connected classification layer

```python
Resnet_model = ResNet(block=bottleneck, layers=[3, 4, 6, 3], num_classes=2)
```

---

## ⚙️ Training

We train the model using:
- **Adam optimizer** (`lr = 0.0001`)
- **CrossEntropyLoss**
- Training and validation metrics (Accuracy, Precision, F1-Score)

```python
criterion = nn.CrossEntropyLoss()
optimizer = Adam(Resnet_model.parameters(), lr=0.0001)
train_loss_history, val_loss_history, train_acc_history, val_acc_history =     train_and_validation(Resnet_model, train_loader, val_loader,
                         epochs=100, optimizer=optimizer,
                         criterion=criterion, device=device)
```

During training, model checkpoints are saved automatically in the **`checkpoints/`** folder.

---

## 📊 Example Output

Training progress will show epoch-wise updates:

```
Epoch 37/100 - Train Loss: 0.3505 - Train Acc: 0.8412 - Val Loss: 0.4647 - Val Acc: 0.7785
Precision: 0.8029 - F1 Score: 0.7806 - Accuracy: 0.7785
```

You can visualize accuracy/loss curves using `matplotlib` later for report purposes.

---

## 🧪 Evaluation

After training, the best model can be evaluated on the **test set** using the same metrics as validation.

---

## 🧵 Notes

- Make sure to delete the `__MACOSX` folder if extracted automatically.
- The learning rate may be adjusted based on dataset size and performance.
- GPU acceleration significantly speeds up training — Colab GPU is recommended.

---

## 📈 Results Summary (Example)

| Metric        | Train | Validation |
|----------------|--------|-------------|
| Accuracy       | ~84%   | ~78%        |
| Precision      | ~80%   | ~78%        |
| F1 Score       | ~78%   | ~77%        |

---

## 📜 Citation / Credits

If you use this code or modify it for your research, please cite appropriately or provide a link to this repository.

Dataset source: [UCSD-AI4H COVID-CT](https://github.com/desaisrkr/https-github.com-UCSD-AI4H-COVID-CT/tree/master)
