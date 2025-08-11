from typing import Tuple, Dict

import tensorflow
from scipy.interpolate import interp1d
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sklearn
import cv2
#import matplotlib.pyplot as plt
def early_stopping_ffnn(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                Dict]:
    """Creates one neural network that uses early stopping during training.

    :param n_inputs: The number of inputs to the models (features).
    :param n_outputs: The number of outputs from the models (likelihood of single class).
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
    )
    """
    depth = 5
    width_d = 256
    model_early = keras.Sequential()
    model_early.add(keras.Input(shape = (n_inputs,)))
    for _ in range(depth):
        model_early.add(keras.layers.Dense(width_d, activation = "relu", \
            kernel_initializer=keras.initializers.RandomNormal(stddev = 0.1)))
        width_d = int(width_d/2)
        model_early.add(keras.layers.Dropout(0.1))
    model_early.add(keras.layers.Dense(n_outputs, activation = "sigmoid"))
    
    opt1 = keras.optimizers.Adam(
        learning_rate=.001
    )
    model_early.compile(loss = 'binary_crossentropy', optimizer = opt1)
    callback = keras.callbacks.EarlyStopping(monitor = 'loss', patience = 10)
    args1 = {'callbacks':[callback]}
    print(model_early)
    return (model_early, args1)

def train_frame_classifier(data_training, features_columns, label_column, epochs = 500):
    """Trains a frame classifier on the provided data.

    :param data: DataFrame containing the training data.
    :param features_columns: List of feature column names.
    :param labels_column: label column name
    :param epochs: Number of epochs to train the model.
    :param early_stopping: Whether to use early stopping during training.
    :return: Trained model and training history.
    """
    X = data_training[features_columns].values
    y = data_training[label_column].values

    model, args = early_stopping_ffnn(X.shape[-1], y.shape[-1])[0]

    history = model.fit(X, y, epochs=epochs, validation_split=0.2, callbacks = args)

    return model, history


def interpolate_features(features, original_spacing, target_spacing, method='linear', existing_timestamps=None):
    """
    Interpolates features over time to a new time grid.

    Parameters:
        features (numpy.ndarray): 2D array (N x T) where T is the number of time points, N is the number of features.
        original_spacing (float): Time interval between original frames.
        target_spacing (float): Desired time interval between frames.
        method (str): Interpolation method ('linear', 'quadratic', 'cubic', etc.)
        existing_timestamps (numpy.ndarray, optional): Array of existing timestamps for the original features.

    Returns:
        new_times (numpy.ndarray): The new interpolated time points.
        new_features (numpy.ndarray): The interpolated feature values (T' x N).
    """
    N, T = features.shape

    # Use existing timestamps if provided, otherwise calculate based on spacing
    original_times = existing_timestamps if existing_timestamps is not None else np.arange(T) * original_spacing
    new_times = np.arange(original_times[0], original_times[-1] + target_spacing, target_spacing)

    # Interpolating each feature independently
    new_features = np.zeros((N, len(new_times)))
    for i in range(N):
        interpolator = interp1d(original_times, features[i, :], kind=method, fill_value="extrapolate")
        new_features[i, :] = interpolator(new_times)

    return new_times, new_features

# Example usage
"""timepoints = 20
features = np.random.rand(3,20)  # 10 time points, 3 features
original_spacing = 20  # e.g., frames every 2 seconds
target_spacing = 15    # interpolate to frames every 1 second

new_times, new_features = interpolate_features(features, original_spacing, target_spacing, method = "quadratic")
original_times = np.arange(timepoints) * original_spacing

print("New Time Grid:", new_times)
print("Original Time Grid:", original_times)
print("Interpolated Features:", new_features)
plt.plot(new_times, new_features[1], label = "new")
plt.plot(original_times, features[1,:], label = "orig")
plt.legend()
plt.show()"""
def standardize_cols(data):
    return 0
def min_max_range_norm_cols(data):
    return 0

def em_mat_weights(gt_labels, pred_labels, labels = [0, 1]):
    """Generates an emission matrix from classifier outputs
    :param gt_labels: ground truth labels
    :param pred_labels: predicted labels from a classifier.
    :param labels: list of labels to consider for the confusion matrix.

    :return: An array in the form of a markov emission matrix.  Columns are predicted labels, rows are true labels.
    """
    cm = np.array(sklearn.metrics.confusion_matrix(np.array(gt_labels), np.array(pred_labels).round(), labels = labels))
    row_sums = cm.sum(axis = 1, keepdims = True)
    
    # If no observations in a row, avoid division by zero
    row_sums[row_sums == 0] = 1
    return cm/row_sums
    
def load_nn_model(file_location = "models/death_frame_predictor_auto_20min.keras"):
    return tensorflow.keras.models.load_model(file_location)

### This section for evaluation of single frames after prediction from ffnn ###
def build_cnn_classifier(input_shape=(64, 64, 1)):
    """Builds a simple CNN for binary classification (Dead vs. Mitosing)."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def train_cnn_check(model, train_images, train_labels, val_images, val_labels, epochs=10, batch_size=32):
    """Trains the CNN model on preprocessed data."""
    
    # Preprocess all images
    train_images = np.array([preprocess_roi(img) for img in train_images]).astype("float32") / 255.0
    val_images = np.array([preprocess_roi(img) for img in val_images]).astype("float32") / 255.0

    # Train the model
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save the trained model
    model.save("cnn_classifier.h5")
    return history

def preprocess_roi(roi, target_size=(64, 64)):
    """Resizes and pads ROI to match the target size while maintaining aspect ratio."""
    
    h, w = roi.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)  # Scaling factor to fit within bounds
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize while maintaining aspect ratio
    resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank canvas with the target size
    padded_roi = np.zeros(target_size, dtype=np.uint8)
    
    # Compute padding offsets
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2

    # Place the resized ROI in the center
    padded_roi[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_roi
    
    return np.expand_dims(padded_roi, axis=-1)  # Ensure (64,64,1) shape

def classify_roi(model, roi_image):
    """Classifies a single ROI image as 'Dead' (0) or 'Mitosing' (1), handling any input size."""
    
    processed_roi = preprocess_roi(roi_image).astype("float32") / 255.0  # Normalize
    processed_roi = np.expand_dims(processed_roi, axis=0)  # Add batch dimension

    prediction = model.predict(processed_roi)[0][0]
    
    return "Mitosing" if prediction > 0.5 else "Dead"

