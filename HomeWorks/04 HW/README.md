# Automated Diagnosis of Pneumonia from Chest X‑Rays with EfficientNet‑B2

> Transfer‑learning baseline on the **Chest X‑Ray Images (Pneumonia)** dataset using TensorFlow/Keras.  
> Implements augmentation, class‑imbalance weighting, fine‑tuning of **EfficientNet‑B2**, and evaluation with confusion matrix & ROC.

---

## 🔎 Overview

Pneumonia is a leading cause of morbidity and mortality worldwide. Chest X‑ray (CXR) imaging is widely available, but manual interpretation can be challenging. This project trains a CNN classifier—**EfficientNet‑B2** with transfer learning—to distinguish **PNEUMONIA** vs **NORMAL** CXRs.

This repository is inspired by the paper:

> *Automated Diagnosis of Pneumonia from Classification of Chest X‑Ray Images using EfficientNet* (ICICT4SD 2021).  
> IEEE Xplore: https://ieeexplore.ieee.org/document/9397055

---

## 🗂️ Data

**Dataset:** *Chest X‑Ray Images (Pneumonia)* by Paul Mooney (Kaggle).  
Kaggle page: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Folder structure after extraction:
```
chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

> **Note:** In this project we read all images from `train/`, `val/`, `test/` into memory, then perform a **stratified re‑split** to 60%/20%/20% (train/val/test) for reproducibility across runs.

---

## ⚙️ Setup (Google Colab)

1) Upload your Kaggle API token (`kaggle.json`) to Colab, then run:
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

2) Download the dataset:
```bash
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip -q chest-xray-pneumonia.zip -d /content
```

3) (Optional) Verify base directory:
```python
base_directory = '/content/chest_xray/'
```

Colab comes with recent TensorFlow; this project was tested on **Python 3.10+** and **TensorFlow 2.15+**.

---

## 🧪 Preprocessing & Augmentation

- **Read & resize** to `128×128`
- **RGB conversion** (OpenCV loads as BGR; we convert to RGB)
- **Normalization** to `[0,1]`
- **Data augmentation** via `ImageDataGenerator`:
  - random rotation (±30°), width/height shift (±0.2), zoom (±0.2), shear (0.2), horizontal flip
  - `fill_mode='constant'`, `cval=0.0`

**Class imbalance:** We compute `class_weight` from the training labels (sklearn `compute_class_weight`) and pass it to `model.fit(...)`.

> ⚠️ **Important:** Ensure deterministic class index mapping. Use:
> ```python
> classes = sorted(os.listdir(os.path.join(base_directory, 'train')))
> # Recommended mapping (by alphabetical order):
> # 0 -> NORMAL, 1 -> PNEUMONIA
> ```
> If you keep the raw `os.listdir` order, document the mapping you observe.

---

## 🧠 Model

**Backbone:** `tf.keras.applications.EfficientNetB2` (pretrained on ImageNet, `include_top=False`).  
**Head:**
```text
GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3)
                      → Dense(64, ReLU)  → Dropout(0.2)
                      → Dense(1, Sigmoid)
```
**Loss:** Binary cross‑entropy  
**Metric:** Accuracy (plus the standard classification report & ROC AUC at eval time)

**Optimization:** Adam with a piecewise constant learning‑rate schedule:
- boundaries = `[1210, 1500]` (in training **steps**)
- values = `[5e-6, 3e-6, 1.5e-6]`

**Fine‑tuning policy:** Freeze early EfficientNet layers at first, then unfreeze later for full fine‑tune if needed.

**Callbacks:**
- `EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)`
- `ModelCheckpoint('tmp/checkpoint.keras', monitor='val_accuracy', save_best_only=True)`

---

## 🚀 Training 

```python
# Generators
batch_size = 32
train_generator = generator.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
val_generator   = generator.flow(X_val,   y_val,   batch_size=batch_size, shuffle=False)

# Compile
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=['accuracy'],
)

# Train
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoints],  # ← make sure to include callbacks
)
```


---




## 🧯 Known Pitfalls & Fixes

- **Label encoding mix‑up**: If you create `y_train_encoded = np.where(y_train == 'NORMAL', 0, 1)` while your labels are integers, you will silently get all 1s. Keep labels consistently **int** or **str**, not both.  
- **Class order non‑determinism**: Use `classes = sorted(os.listdir(...))` and map explicitly. Document it in the README.
- **Callbacks not used**: Make sure `callbacks=[early_stopping, checkpoints]` is passed to `model.fit(...)`.
- **Learning‑rate boundaries**: In `PiecewiseConstantDecay`, `boundaries` are **steps**, not epochs. Adjust based on `steps_per_epoch = ceil(len(X_train)/batch_size)`.
- **Transfer‑learning inference flag**: When calling the base model, prefer `training=False` during feature extraction; set `layer.trainable=True` and re‑compile before fine‑tuning.


---

## 📚 Citation

If you use this repository in academic work, please cite the referenced paper and the dataset authors.

- Paper: *Automated Diagnosis of Pneumonia from Classification of Chest X‑Ray Images using EfficientNet*, ICICT4SD 2021.  
  IEEE Xplore: https://ieeexplore.ieee.org/document/9397055
- Dataset: *Chest X‑Ray Images (Pneumonia)* (Kaggle) — derived from pediatric CXR data curated by Kermany et al. Please consult the Kaggle page for license/terms.

---

## 🙏 Acknowledgments

- Paul Mooney & contributors for releasing the CXR dataset on Kaggle.
- The EfficientNet authors and TensorFlow/Keras team.

---

## 📄 License

This code is made available for research and educational purposes. **Dataset licensing may impose additional restrictions**—please review the Kaggle dataset license before redistribution.

---
