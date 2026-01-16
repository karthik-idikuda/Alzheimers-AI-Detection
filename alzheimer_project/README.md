# NeuroXAI: Explainable AI for Alzheimer's Detection

An interpretable deep learning framework for Alzheimer's detection from MRI scans.

## 🧠 Overview

This project provides:
- **CNN-based classification** into 4 dementia stages
- **SHAP explainability** for visual interpretation
- **Clinical dashboard** for side-by-side comparison

## 📁 Project Structure

```
alzheimer_project/
├── Datasets/
│   ├── MRI Dataset/
│   │   ├── train.parquet
│   │   └── test.parquet
│   └── ALZ_Variant Datset/
│       └── preprocessed_alz_data.npz
├── data_utils.py          # Data loading utilities
├── model.py               # CNN model architecture
├── explainability.py      # SHAP visualization
├── AI4Alzheimers_Notebook.ipynb  # Main notebook
├── AI4Alzheimers_Report.md       # Project report
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook AI4Alzheimers_Notebook.ipynb
```

### 3. Or Run Training Script
```python
from data_utils import load_mri_data, get_class_weights
from model import build_model, compile_model, train_model

# Load data
X_train, y_train = load_mri_data('Datasets/MRI Dataset/train.parquet')
X_test, y_test = load_mri_data('Datasets/MRI Dataset/test.parquet')

# Build and train model
model = build_model()
model = compile_model(model)
class_weights = get_class_weights(y_train)
history = train_model(model, X_train, y_train, X_test, y_test, class_weights)
```

## 📊 Classes

| Label | Description |
|-------|-------------|
| 0 | Non-Demented |
| 1 | Very Mild Dementia |
| 2 | Mild Dementia |
| 3 | Moderate Dementia |

## 🔬 Key Features

### CNN Architecture
- 3 Convolutional blocks (32→64→128 filters)
- BatchNormalization + Dropout
- Dense layers (256→128→4)

### SHAP Explainability
- GradientExplainer for saliency maps
- Visual overlay on MRI scans
- Clinical dashboard generation

## 📝 Hackathon Submission

**AI 4 Alzheimer's Hackathon**
- Deadline: January 1, 2026
- Team: Karthik Idikuda, Praveen Kumar, Jakka Siva

## 📄 License

MIT License
