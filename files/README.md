# Skin Disease Detection System (Deep Learning)

A multi-class image classifier built with **PyTorch** and **ResNet18 transfer learning**.
It predicts one of four classes (melanoma, nevus, eczema, psoriasis) and returns a softmax
confidence score with a simple confidence-based warning.

> ⚠️ **Disclaimer:** This is an educational mini project trained on a **synthetic dataset**
> (solid colours + noise). It is **not** a medical device and must not be used for diagnosis.

## Features
- Synthetic dataset generator (4 classes, 50 train / 10 val images per class)
- ResNet18 pretrained on ImageNet, backbone frozen, new fully connected head
- Training and validation loop with accuracy logging
- Prediction function returning class + confidence
- Confidence warning: `< 0.6` low, `< 0.8` medium, otherwise high

## Project structure
```
.
├── skin_disease_detection.py   # dataset generation, training, evaluation, prediction
├── requirements.txt
├── docs/                       # put the project report PDF here
└── README.md
```

## Setup & run
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python skin_disease_detection.py
```

## Results (synthetic data)
| Metric | Value |
|---|---|
| Training accuracy | ~99% |
| Validation accuracy | 100% |

These numbers are high **only because the synthetic classes differ by base colour**, which is
trivially separable. Performance on real skin images would be much lower.

## Limitations
- Synthetic data, not real clinical images
- Only 4 classes; not clinically validated
- No robustness testing across lighting, skin tones or cameras
- Confidence score is the softmax probability, which is not a calibrated medical risk level

## Future work
- Train on real datasets (HAM10000, ISIC)
- Add Grad-CAM explainability
- Mobile app / real-time camera detection

## Documentation
Full project report: [`docs/Skin_Disease_Detection_Report.pdf`](docs/)

## Authors
Add the names of everyone who contributed, e.g. the authors listed on the report:
Shrishti Gupta, Samridhi Kumari (and any other teammates).

## References
- PyTorch docs: https://pytorch.org/docs/stable/index.html
- HAM10000 paper: https://arxiv.org/abs/1803.10417
- ISIC Archive: https://www.isic-archive.com/
