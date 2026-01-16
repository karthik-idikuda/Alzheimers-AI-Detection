# NeuroXAI: Explainable AI for Alzheimer's Detection

**AI 4 Alzheimer's Hackathon Submission**

---

## 1. Problem Statement

Alzheimer's disease affects millions globally, yet its progression is often too subtle to track until irreversible damage occurs. While deep learning models can detect Alzheimer's with high accuracy, they suffer from the "Black Box" problem—doctors cannot treat patients based solely on a probability without understanding *why* the AI made its decision.

**Our Goal**: Bridge this trust gap by building an interpretable AI system that not only diagnoses but also explains its decisions by highlighting the exact brain regions that drove its conclusion.

---

## 2. Methods

### 2.1 Dataset
- **MRI Brain Scans**: 5,120 training images + 1,280 test images
- **128×128 grayscale** images extracted from Parquet format
- **4 Classes**: Non-Demented, Very Mild, Mild, Moderate Dementia

### 2.2 Model Architecture
We designed a custom CNN with three convolutional blocks:

| Layer Block | Filters | Features |
|-------------|---------|----------|
| Block 1 | 32 | Conv → BatchNorm → ReLU → MaxPool → Dropout (25%) |
| Block 2 | 64 | Conv → BatchNorm → ReLU → MaxPool → Dropout (25%) |
| Block 3 | 128 | Conv → BatchNorm → ReLU → MaxPool → Dropout (25%) |
| Dense | 256→128 | BatchNorm → ReLU → Dropout (50%) |
| Output | 4 | Softmax activation |

**Key Features**:
- BatchNormalization for training stability
- Dropout regularization to prevent overfitting
- Class weights to handle imbalanced data
- Adam optimizer with learning rate scheduling

### 2.3 Explainability (SHAP)
We integrated **SHAP GradientExplainer** to generate saliency maps:
1. Calculate pixel-level importance scores using Shapley values
2. Generate heatmaps highlighting influential brain regions
3. Overlay explanations on original MRI scans

---

## 3. Results

### 3.1 Classification Performance
The model achieves strong performance across all dementia stages:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Non-Demented | High | High | High |
| Very Mild | Moderate | Moderate | Moderate |
| Mild | High | High | High |
| Moderate | High | High | High |

*Note: Very Mild stage is hardest to detect due to subtle brain changes.*

### 3.2 Visual Explanations
Our SHAP visualizations reveal the model focuses on clinically relevant regions:
- **Enlarged Ventricles**: Primary indicator for Moderate Dementia
- **Hippocampus Region**: Early atrophy marker
- **Cortical Gray Matter**: Progressive thinning

---

## 4. Key Findings

1. **Biological Validation**: The model autonomously learned to identify ventricle enlargement—a standard radiological marker for dementia progression.

2. **Trust Through Transparency**: Side-by-side visualization (MRI → SHAP → Overlay → Probability) creates a "visual second opinion" for clinicians.

3. **Class Imbalance Challenge**: Very Mild cases require higher sensitivity due to microscopic changes; class weighting improved detection.

---

## 5. Future Work

- **3D Volumetric Analysis**: Measure exact brain atrophy volumes
- **Longitudinal Tracking**: Predict progression from scan timelines
- **Multimodal Integration**: Combine with genetic biomarkers
- **Mobile Deployment**: TensorFlow Lite for offline clinical use

---

## 6. Conclusion

NeuroXAI demonstrates that explainable AI can bridge the gap between computational accuracy and clinical trust. By combining CNN classification with SHAP-based visual explanations, we provide doctors with interpretable evidence to support Alzheimer's diagnosis—moving AI from a "black box" to a transparent diagnostic partner.

---

**Repository**: [GitHub Link]  
**Team**: Karthik Idikuda, Praveen Kumar, Jakka Siva Subramanyam Guptha  
**Contact**: idikudakarthik55@gmail.com
