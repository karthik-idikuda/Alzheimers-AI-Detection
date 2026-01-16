"""
CNN Model for Alzheimer's MRI Classification
Custom architecture with 3 convolutional blocks for 4-class classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks


def build_model(input_shape=(128, 128, 1), num_classes=4):
    """
    Build a CNN model for Alzheimer's classification.
    
    Architecture:
    - 3 Convolutional blocks with BatchNorm and MaxPooling
    - Dense layers with Dropout for regularization
    - Softmax output for 4-class classification
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """Compile the model with Adam optimizer and categorical crossentropy."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(model_path='best_model.keras', patience=5):
    """Get training callbacks for early stopping and checkpointing."""
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]


def train_model(model, X_train, y_train, X_val, y_val, 
                class_weights=None, epochs=20, batch_size=32):
    """
    Train the model with validation.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        class_weights: Dict of class weights for imbalance
        epochs: Number of training epochs
        batch_size: Batch size
    
    Returns:
        history: Training history
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1
    )
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = predictions.argmax(axis=1)
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'predictions': predictions,
        'predicted_classes': predicted_classes
    }


if __name__ == "__main__":
    # Test model building
    model = build_model()
    model = compile_model(model)
    model.summary()
