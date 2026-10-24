# 🩺 Skin Disease Detection System

### Multi-class skin disease classification using Deep Learning (ResNet18 + Transfer Learning) on a synthetic dataset

**Python | PyTorch | ResNet18 | Google Colab | Academic Project**

> ⚠️ **Disclaimer:** This is an **educational mini project** trained on a **synthetic dataset**
> (solid colours + random noise). It is **not a medical device** and must **never** be used for
> real diagnosis. Always consult a qualified doctor for any skin condition.

---

## 📑 Table of Contents
1. [Overview](#-overview)
2. [Objectives](#-objectives)
3. [Features](#-features)
4. [Tech Stack](#-tech-stack)
5. [System Architecture](#-system-architecture)
6. [Model Architecture](#-model-architecture)
7. [Dataset](#-dataset)
8. [Project Structure](#-project-structure)
9. [Installation & Setup](#-installation--setup)
10. [How to Run](#-how-to-run)
11. [How the Code Works](#-how-the-code-works)
12. [Sample Output](#-sample-output)
13. [Results](#-results)
14. [Test Cases](#-test-cases)
15. [Limitations](#-limitations)
16. [Future Work](#-future-work)
17. [Documentation](#-documentation)
18. [Authors](#-authors)
19. [References](#-references)

---

## 📌 Overview
Skin diseases are among the most common health problems worldwide, and early diagnosis plays a
major role in effective treatment. Manual diagnosis needs expert dermatologists and can be slow
and error-prone.

This project builds an automated **image classification system** that takes a skin image and
predicts the disease class along with a **confidence score**. It uses a **ResNet18** model
pretrained on ImageNet (transfer learning) with a new final layer for 4 skin-disease classes.

## 🎯 Objectives
- Build a deep-learning-based **multi-class** skin disease classifier
- Use **CNN with transfer learning** (ResNet18) for classification
- Provide a **confidence score** for every prediction
- Show a **warning message** when confidence is low
- Demonstrate the potential of AI in healthcare and early screening

## ✨ Features
- 🧪 Synthetic dataset generator (4 classes, no download needed)
- 🧠 ResNet18 pretrained on ImageNet, backbone frozen, new classification head
- 📈 Training and validation loop with per-epoch accuracy
- 🔍 Prediction function returning **class + confidence**
- 🚦 Confidence-based warning system
- 💾 Model saving (`skin_model.pth`)
- 📊 Visualization of prediction confidence per class

## 🛠 Tech Stack
| Component | Technology |
|---|---|
| Language | Python |
| Deep learning framework | PyTorch, TorchVision |
| Model | ResNet18 (transfer learning) |
| Image handling | Pillow (PIL), NumPy |
| Visualization | Matplotlib |
| Environment | Google Colab / Jupyter Notebook |

## 🏗 System Architecture
The system follows a simple layered pipeline:

```
Input Image  →  Preprocessing  →  ResNet18 Model  →  Softmax  →  Prediction + Confidence
 (JPG/PNG)     (Resize 224×224,     (feature          (class        (class name,
                ToTensor)            extraction + FC)  probabilities) score, warning)
```

1. **Input Layer** – accepts a skin image (JPG/PNG)
2. **Preprocessing Layer** – resizes to 224×224 and converts to a tensor
3. **Model Layer** – ResNet18 extracts features and classifies
4. **Output Layer** – predicted class, confidence score and warning message

## 🧠 Model Architecture
- **Base model:** ResNet18 (pretrained on ImageNet)
- **Transfer learning:** all pretrained layers are **frozen**
- **Custom head:** final fully connected layer replaced with `Linear(512, 4)`
- **Output:** Softmax probabilities over 4 classes
- **Loss:** Cross-Entropy Loss
- **Optimizer:** Adam (learning rate = 0.001)
- **Batch size:** 16  |  **Epochs:** 10  |  **Input size:** 224 × 224

## 🗂 Dataset
The dataset is **synthetic** and generated automatically with Python (NumPy + PIL).
Each class is a solid base colour with random noise added.

| Class | Base colour (RGB) | Train images | Validation images |
|---|---|---|---|
| Melanoma | (120, 50, 50) | 50 | 10 |
| Nevus | (180, 120, 90) | 50 | 10 |
| Eczema | (200, 120, 120) | 50 | 10 |
| Psoriasis | (230, 230, 230) | 50 | 10 |
| **Total** | | **200** | **40** |

Folder layout created by the code:
```
dataset/
├── train/
│   ├── melanoma/  nevus/  eczema/  psoriasis/
└── val/
    ├── melanoma/  nevus/  eczema/  psoriasis/
```
> Because classes differ only by colour, the task is very easy, so accuracy is unrealistically high.

## 📁 Project Structure
```
Skin-Disease-Detection-System/
├── Skin_Disease_Detection.ipynb       # Google Colab notebook (step-by-step)
├── skin_disease_detection.py          # Same project as a single Python script
├── requirements.txt                   # Python dependencies
├── Skin_Disease_Detection_Report.pdf  # Full project report
├── .gitignore                         # Ignores dataset and model files
└── README.md                          # Project documentation
```

## ⚙️ Installation & Setup

### Option 1: Google Colab (easiest)
1. Open [Google Colab](https://colab.research.google.com).
2. Click **File → Upload notebook** and select `Skin_Disease_Detection.ipynb`.
3. Click **Runtime → Run all**.

No installation is needed; Colab already includes PyTorch.

### Option 2: Run locally
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ How to Run
**Notebook:** run the cells from top to bottom.

**Script (local):**
```bash
python skin_disease_detection.py
```
This will create the dataset, train the model for 10 epochs, evaluate it, save
`skin_model.pth`, and print a sample prediction.

## 🔧 How the Code Works
| Step | What happens |
|---|---|
| 1 | Install and import libraries |
| 2 | Select device (GPU if available, else CPU) |
| 3 | Generate the synthetic dataset (train and val folders) |
| 4 | Load images with `ImageFolder` and `DataLoader` |
| 5 | Load pretrained ResNet18, freeze layers, replace final layer |
| 6 | Define loss function and optimizer |
| 7 | Train for 10 epochs and print accuracy |
| 8 | Evaluate on the validation set |
| 9 | Save the trained model |
| 10 | `predict_image()` returns class name and confidence |
| 11 | Test on a validation image |
| 12 | Print a warning based on confidence |

**Confidence warning system**
| Confidence | Message |
|---|---|
| < 0.6 | ⚠️ Low Confidence – Consult Doctor |
| 0.6 – 0.8 | ⚠️ Medium Risk |
| ≥ 0.8 | ✅ High Confidence Prediction |

## 🖥 Sample Output
```
Using device: cpu
✅ Dataset Created
Classes: ['eczema', 'melanoma', 'nevus', 'psoriasis']
✅ Model Ready
🚀 Training Started
Epoch 1/10 → Accuracy: 0.6050
Epoch 2/10 → Accuracy: 0.9750
Epoch 3/10 → Accuracy: 0.9800
...
Epoch 10/10 → Accuracy: 0.9900
✅ Training Completed
Validation Accuracy: 1.0
✅ Model Saved
Prediction: melanoma
Confidence: 0.9806
✅ High Confidence Prediction
```
*(Exact numbers vary slightly between runs because the data is random.)*

## 📊 Results
| Metric | Value |
|---|---|
| Training accuracy | ~99% |
| Validation accuracy | 100% (synthetic data) |
| Prediction confidence | High |

> These results are high **only because the synthetic classes are trivially separable by colour**.
> Performance on real skin images would be much lower, which indicates a risk of overfitting to
> simple, controlled data.

## ✅ Test Cases
| ID | Input | Expected output |
|---|---|---|
| TC1 | Valid image | Correct prediction |
| TC2 | Random image | Low confidence |
| TC3 | Clear image | High confidence |
| TC4 | Invalid file | Error message |

## ⚠️ Limitations
- **Synthetic dataset**, not real clinical images
- **Overfitting risk**, since very high accuracy comes from simple patterns
- Only **4 classes** are supported
- **Not clinically verified**; cannot replace a doctor
- May fail with different lighting, skin tones or camera quality
- The "risk" message is based on model confidence, not an actual medical risk level
- The Grad-CAM explainability mentioned in the report is planned but not yet implemented

## 🚀 Future Work
- Train on real datasets such as **HAM10000** and **ISIC**
- Add **Grad-CAM** visualizations to highlight important image regions
- Add data augmentation and ImageNet normalization
- Report precision, recall, F1-score and a confusion matrix
- Support more skin disease classes
- Deploy as a **web / mobile app** with real-time camera detection
- Test fairness across different skin tones

## 📄 Documentation
Full project report (objective, architecture, implementation, results, conclusion):
[Skin_Disease_Detection_Report.pdf](Skin_Disease_Detection_Report.pdf)

## 👥 Authors
Add the names of everyone who worked on this project. The project report lists:
- **Shrishti Gupta**
- **Samridhi Kumari**

*Subject: Deep Learning | Academic Year: 2025–2026*

## 📚 References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ResNet in PyTorch Hub](https://pytorch.org/hub/pytorch_vision_resnet/)
- [HAM10000 paper](https://arxiv.org/abs/1803.10417)
- [ISIC Archive](https://www.isic-archive.com/)
- [Kaggle: Skin Disease Detection Dataset (HAM10000 + ISIC)](https://www.kaggle.com/datasets/nour12347653/skin-disease-detection-dataset-ham10000-isic)
- [Deep Learning Book – Ian Goodfellow](https://www.deeplearningbook.org)

---
⭐ If you found this project useful, consider giving the repository a star!
