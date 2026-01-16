# Alzheimer's AI Detection System

## Overview
A specialized AI application for the early detection of Alzheimer's disease using MRI scan analysis. This project focuses on high-precision processing of neuroimaging data to identify subtle biomarkers often missed by traditional diagnostic methods.

## Features
-   **MRI Analysis**: Deep learning pipelines optimized for brain scan processing.
-   **Early Warning**: Detection of mild cognitive impairment (MCI) indicators.
-   **Batch Processing**: Capability to analyze large datasets of patient scans.
-   **Visual Reports**: Generation of annotated images highlighting areas of concern.
-   **Clinical Dashboard**: Interface for medical professionals to review findings.

## Technology Stack
-   **Deep Learning**: PyTorch / Keras.
-   **Imaging**: NiBabel, SimpleITK.
-   **Backend**: Python Flask.
-   **Frontend**: React / Streamlit.

## Usage Flow
1.  **Ingest**: System receives MRI DICOM/NIfTI files.
2.  **Preprocess**: Skull stripping and image normalization.
3.  **Analyze**: CNN models classify the scan (AD, MCI, CN).
4.  **Report**: Detailed diagnostic report is generated.

## Quick Start
```bash
# Clone the repository
git clone https://github.com/Nytrynox/Alzheimers-AI-Detection.git

# Install dependencies
pip install -r requirements.txt

# Run the analysis server
python app.py
```

## License
MIT License

## Author
**Karthik Idikuda**
