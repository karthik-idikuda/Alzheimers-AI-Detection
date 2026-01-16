"""
Data Utilities for Alzheimer's MRI Classification
Handles loading, preprocessing, and augmentation of MRI data from parquet files.
"""

import numpy as np
import pandas as pd
from PIL import Image
import io
from sklearn.utils.class_weight import compute_class_weight


def extract_bytes(blob):
    """Unwrap a dict-wrapped binary payload if needed."""
    if isinstance(blob, dict):
        for key in ("bytes", "data", "image"):
            if key in blob and isinstance(blob[key], (bytes, bytearray)):
                return blob[key]
        for v in blob.values():
            if isinstance(v, (bytes, bytearray)):
                return v
        raise TypeError(f"No bytes found in dict payload: {list(blob.keys())}")
    return blob


def bytes_to_pixels(b: bytes) -> np.ndarray:
    """Convert raw image bytes (JPEG/PNG) into a 2D numpy array."""
    img = Image.open(io.BytesIO(b)).convert('L')  # Convert to grayscale
    return np.array(img)


def load_mri_data(data_path: str, normalize: bool = True):
    """
    Load MRI data from parquet file.
    
    Args:
        data_path: Path to the parquet file
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        X: numpy array of images (N, 128, 128, 1)
        y: numpy array of labels
    """
    df = pd.read_parquet(data_path)
    
    # Convert bytes to pixel arrays
    images = []
    for _, row in df.iterrows():
        pixels = bytes_to_pixels(extract_bytes(row['image']))
        images.append(pixels)
    
    X = np.array(images)
    y = np.array(df['label'].values)
    
    # Add channel dimension
    X = X.reshape(-1, 128, 128, 1)
    
    if normalize:
        X = X.astype('float32') / 255.0
    
    return X, y


def get_class_weights(y):
    """Calculate class weights for imbalanced dataset."""
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def get_label_names():
    """Return human-readable label names."""
    return {
        0: 'Non-Demented',
        1: 'Very Mild Dementia',
        2: 'Mild Dementia',
        3: 'Moderate Dementia'
    }


def create_data_augmentation():
    """Create data augmentation layer for training."""
    import tensorflow as tf
    
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ], name='data_augmentation')


if __name__ == "__main__":
    # Test data loading
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, "Datasets/MRI Dataset/train.parquet")
    
    if os.path.exists(train_path):
        X, y = load_mri_data(train_path)
        print(f"Loaded {len(X)} training samples")
        print(f"Image shape: {X[0].shape}")
        print(f"Label distribution: {np.bincount(y)}")
        print(f"Class weights: {get_class_weights(y)}")
