import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Dict, Any, Tuple

from src.training.custom_f1_score import CustomF1Score
from src.training.custom_frame_level_loss import CustomFrameLevelLoss
from src.training.attention import Attention


def get_custom_objects() -> Dict[str, Any]:
    """
    Get custom objects for loading the model. All custom objects used in the model
    should be added to this dictionary, but only the necessary ones will be loaded.

    Returns:
    - custom_objects (Dict[str, Any]): Dictionary of custom objects.
    """
    custom_objects = {
        'Attention': Attention,
        'custom_frame_level_loss': CustomFrameLevelLoss,
        'CustomF1Score': CustomF1Score,
    }

    return custom_objects


class ModelPredictor:
    def __init__(self, model_path, features_dir, output_labels_dir, audio_file, maxlen=None):
        """
        Initialize the model predictor.

        Args:
        - model_path (str): Path to the model file (.keras).
        - features_dir (str): Directory containing .npy feature files.
        - output_labels_dir (str): Directory to save predicted label files.
        - audio_file (str): Path to the audio file (.wav).
        - maxlen (int): Maximum sequence length for padding/truncation.
        """
        self.model_path = model_path
        self.features_dir = features_dir
        self.output_labels_dir = output_labels_dir
        self.audio_file = audio_file

        # Load the model
        self.model = self.load_model()

        # Determine maxlen from the model
        self.maxlen = maxlen if maxlen is not None else self.get_maxlen()

        # Ensure output labels directory exists
        os.makedirs(self.output_labels_dir, exist_ok=True)

    def load_model(self) -> tf.keras.Model:
        """
        Load the model from the model file.

        Returns:
        - model (tf.keras.Model): Model loaded from the model file.
        """
        print("Loading model...")
        model = tf.keras.models.load_model(
            self.model_path,
            custom_objects=get_custom_objects()
        )
        print("Model loaded successfully.")

        return model

    def get_maxlen(self) -> int:
        """
        Get the maximum sequence length for padding/truncation.

        Returns:
        - maxlen (int): Maximum sequence length.
        """
        # Determine maxlen from the model's input shape
        input_shape = self.model.input_shape

        if isinstance(input_shape, list):
            # If model has multiple inputs
            input_shape = input_shape[0]
        maxlen = input_shape[1]

        if maxlen is None:
            print("Model accepts variable-length sequences.")
        else:
            print(f"Model expects sequences of length {maxlen}.")

        return maxlen

    def process_features(self, feature_file_path) -> Tuple[np.ndarray, int]:
        """
        Process the features from the feature file.

        Args:
        - feature_file_path (str): Path to the feature file.

        Returns:
        - features_padded (np.ndarray): Padded or truncated features.
        - original_length (int): Original length of the features before padding.
        """
        # Load features
        features = np.load(feature_file_path)
        original_length = features.shape[0]  # Get original length
        if features.ndim == 2:
            # Shape: (1, timesteps, features)
            features = np.expand_dims(features, axis=0)

        # If maxlen is not None, pad or truncate features
        if self.maxlen is not None:
            features_padded = pad_sequences(
                features,
                maxlen=self.maxlen,
                dtype='float32',
                padding='post',
                truncating='post'
            )
        else:
            features_padded = features  # Use the features as is

        return features_padded, original_length

    def make_predictions(self, features_padded) -> np.ndarray:
        """
        Make predictions on the features.

        Args:
        - features_padded (np.ndarray): Padded or truncated features.

        Returns:
        - frame_predictions_binary (np.ndarray): Binary frame-level predictions.
        - utt_predictions_binary (np.ndarray): Binary utterance-level predictions.
        """
        # Make predictions
        predictions = self.model.predict(features_padded)
        frame_predictions, utt_predictions = predictions

        # Apply threshold to get binary predictions
        threshold = 0.5
        frame_predictions_binary = (frame_predictions >= threshold).astype(int)
        utt_predictions_binary = (utt_predictions >= threshold).astype(int)

        return frame_predictions_binary, utt_predictions_binary

    def save_label_file(self, feature_file_name, frame_predictions_binary, original_length) -> str:
        """
        Save the frame-level predictions as a label file.

        Args:
        - feature_file_name (str): Name of the feature file.
        - frame_predictions_binary (np.ndarray): Binary frame-level predictions.
        - original_length (int): Original length of the features before padding.

        Returns:
        - label_file_path (str): Path to the saved label file.
        """
        # The label file should have the same name as the feature file, but in the labels directory
        label_file_name = feature_file_name.replace('.npy', '_labels.npy')
        label_file_path = os.path.join(self.output_labels_dir, label_file_name)

        # Trim the predictions to the original length
        frame_predictions_trimmed = frame_predictions_binary[0][:original_length]

        # Save the frame-level predictions as labels
        np.save(label_file_path, frame_predictions_trimmed)
        print(f"Label file saved: {label_file_path}")

        return label_file_path

    def run_predictions(self) -> None:
        """
        Run predictions on feature files corresponding to the given audio file.

        Saves the frame-level predictions as label files in the output labels directory.
        """
        # Get the base name of the audio file without extension
        audio_base_name = os.path.splitext(
            os.path.basename(self.audio_file))[0]

        # Iterate over all feature files in the features directory
        for feature_file_name in os.listdir(self.features_dir):
            if feature_file_name.endswith('.npy') and feature_file_name.startswith(audio_base_name):
                feature_file_path = os.path.join(
                    self.features_dir, feature_file_name)

                print(f"Processing {feature_file_name}...")

                # Process features
                features_padded, original_length = self.process_features(
                    feature_file_path)

                # Make predictions
                frame_preds_bin, utt_preds_bin = self.make_predictions(
                    features_padded)

                # Save label file
                self.save_label_file(
                    feature_file_name, frame_preds_bin, original_length)

    @staticmethod
    def from_args(args) -> 'ModelPredictor':
        """
        Create a ModelPredictor object from command-line arguments.

        Args:
        - args: Command-line arguments.

        Returns:
        - model_predictor (ModelPredictor): ModelPredictor object.
        """
        return ModelPredictor(
            model_path=args.model_path,
            features_dir=args.features_dir,
            output_labels_dir=args.output_labels_dir,
            audio_file=args.audio_file
        )
