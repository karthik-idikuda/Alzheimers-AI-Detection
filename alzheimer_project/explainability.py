"""
SHAP Explainability Module for Alzheimer's Classification
Generates saliency maps and visualizations using SHAP GradientExplainer.
"""

import numpy as np
import matplotlib.pyplot as plt
import shap


def create_explainer(model, background_data):
    """
    Create a SHAP GradientExplainer.
    
    Args:
        model: Trained Keras model
        background_data: Sample of training data for baseline
    
    Returns:
        SHAP GradientExplainer
    """
    # Use a subset of data as background
    if len(background_data) > 100:
        indices = np.random.choice(len(background_data), 100, replace=False)
        background_data = background_data[indices]
    
    explainer = shap.GradientExplainer(model, background_data)
    return explainer


def get_shap_values(explainer, images):
    """
    Calculate SHAP values for given images.
    
    Args:
        explainer: SHAP GradientExplainer
        images: Images to explain (N, 128, 128, 1)
    
    Returns:
        SHAP values array
    """
    shap_values = explainer.shap_values(images)
    return shap_values


def plot_shap_explanation(image, shap_values, predicted_class, true_class, 
                          class_names, save_path=None):
    """
    Create a visualization showing original image and SHAP heatmap.
    
    Args:
        image: Original image (128, 128, 1)
        shap_values: SHAP values for the predicted class
        predicted_class: Predicted class index
        true_class: True class index
        class_names: Dict of class names
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title('Original MRI Scan', fontsize=12)
    axes[0].axis('off')
    
    # SHAP Heatmap
    shap_img = shap_values[predicted_class].squeeze() if isinstance(shap_values, list) else shap_values.squeeze()
    
    # Normalize for visualization
    abs_max = np.abs(shap_img).max()
    if abs_max > 0:
        shap_img = shap_img / abs_max
    
    axes[1].imshow(shap_img, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('SHAP Feature Importance', fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image.squeeze(), cmap='gray', alpha=0.7)
    overlay = axes[2].imshow(shap_img, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay (Attention Regions)', fontsize=12)
    axes[2].axis('off')
    
    # Add colorbar
    plt.colorbar(overlay, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Overall title
    pred_name = class_names.get(predicted_class, f'Class {predicted_class}')
    true_name = class_names.get(true_class, f'Class {true_class}')
    status = "✓ Correct" if predicted_class == true_class else "✗ Incorrect"
    fig.suptitle(f'Prediction: {pred_name} | True: {true_name} | {status}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_clinical_dashboard(images, shap_values, predictions, true_labels, 
                               class_names, num_samples=4, save_path=None):
    """
    Create a clinical dashboard showing multiple samples with explanations.
    
    Args:
        images: Array of images
        shap_values: SHAP values for all images
        predictions: Model predictions (probabilities)
        true_labels: True labels
        class_names: Dict of class names
        num_samples: Number of samples to display
        save_path: Optional path to save
    """
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    # Select diverse samples
    predicted_classes = predictions.argmax(axis=1)
    
    for i in range(min(num_samples, len(images))):
        img = images[i].squeeze()
        pred_class = predicted_classes[i]
        true_class = true_labels[i]
        probs = predictions[i]
        
        # Get SHAP values for predicted class
        if isinstance(shap_values, list):
            shap_img = shap_values[pred_class][i].squeeze()
        else:
            shap_img = shap_values[i].squeeze()
        
        abs_max = np.abs(shap_img).max()
        if abs_max > 0:
            shap_img = shap_img / abs_max
        
        # Original
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title('MRI Scan', fontsize=10)
        axes[i, 0].axis('off')
        
        # SHAP
        axes[i, 1].imshow(shap_img, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 1].set_title('SHAP Importance', fontsize=10)
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(img, cmap='gray', alpha=0.7)
        axes[i, 2].imshow(shap_img, cmap='jet', alpha=0.4)
        axes[i, 2].set_title('Attention Overlay', fontsize=10)
        axes[i, 2].axis('off')
        
        # Probability bar chart
        bars = axes[i, 3].barh(range(len(class_names)), probs, color=['green' if j == pred_class else 'steelblue' for j in range(len(class_names))])
        axes[i, 3].set_yticks(range(len(class_names)))
        axes[i, 3].set_yticklabels([class_names[j] for j in range(len(class_names))], fontsize=8)
        axes[i, 3].set_xlim(0, 1)
        axes[i, 3].set_title(f'Pred: {class_names[pred_class]}\nTrue: {class_names[true_class]}', fontsize=10)
        
        # Highlight correct/incorrect
        if pred_class == true_class:
            axes[i, 3].patch.set_facecolor('#d4edda')
        else:
            axes[i, 3].patch.set_facecolor('#f8d7da')
    
    plt.suptitle('NeuroXAI Clinical Dashboard: Alzheimer\'s Detection with Explainability', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_global_patterns(shap_values, class_names, images, labels, save_path=None):
    """
    Create visualization showing average SHAP patterns per class.
    
    Args:
        shap_values: SHAP values for all samples
        class_names: Dict of class names
        images: All images
        labels: True labels
        save_path: Optional path to save
    """
    fig, axes = plt.subplots(2, len(class_names), figsize=(4*len(class_names), 8))
    
    for class_idx in range(len(class_names)):
        # Get indices for this class
        class_mask = labels == class_idx
        
        if class_mask.sum() > 0:
            # Average image
            avg_image = images[class_mask].mean(axis=0).squeeze()
            axes[0, class_idx].imshow(avg_image, cmap='gray')
            axes[0, class_idx].set_title(f'{class_names[class_idx]}\n(n={class_mask.sum()})', fontsize=10)
            axes[0, class_idx].axis('off')
            
            # Average SHAP
            if isinstance(shap_values, list):
                avg_shap = np.abs(shap_values[class_idx][class_mask]).mean(axis=0).squeeze()
            else:
                avg_shap = np.abs(shap_values[class_mask]).mean(axis=0).squeeze()
            
            axes[1, class_idx].imshow(avg_shap, cmap='hot')
            axes[1, class_idx].set_title('Avg. Feature Importance', fontsize=10)
            axes[1, class_idx].axis('off')
    
    axes[0, 0].set_ylabel('Average MRI', fontsize=12)
    axes[1, 0].set_ylabel('Average SHAP', fontsize=12)
    
    plt.suptitle('Global Pattern Analysis: What the Model Learns per Class', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("SHAP Explainability module loaded successfully.")
    print("Functions available: create_explainer, get_shap_values, plot_shap_explanation, create_clinical_dashboard, plot_global_patterns")
