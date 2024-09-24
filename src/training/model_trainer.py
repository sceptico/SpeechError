"""
model_trainer.py

Class for training models using k-fold cross-validation.
"""
import os
import json
import glob
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import KFold
from typing import List, Dict, Tuple, Any

from data_utils import load_data_from_csv, get_max_sequence_length
from custom_data_generator import CustomDataGenerator
from model_utils import create_model
from parse_config import parse_config
from custom_f1_score import CustomF1Score
from custom_frame_level_loss import CustomFrameLevelLoss
from attention import Attention


class ModelTrainer:
    """
    Class for training models using k-fold cross-validation.

    Attributes:
    - config_paths (List[str]): List of configuration file paths.
    - configs (List[Dict[str, Any]]): List of parsed configurations.
    - config (Dict[str, Any]): Configuration dictionary.
    - training_config (Dict[str, Any]): Training configuration dictionary.
    - log_config (Dict[str, Any]): Log configuration dictionary.
    - frame_loss_function (str): Loss function for frame-level outputs.
    - utt_loss_function (str): Loss function for utterance-level outputs.
    - frame_loss_weight (float): Weight for the frame-level loss.
    - utt_loss_weight (float): Weight for the utterance-level loss.
    - optimizer_name (str): Name of the optimizer to use.
    - learning_rate (float): Learning rate for the optimizer.
    - epochs (int): Number of epochs for training.
    - batch_size (int): Batch size for training.
    - k_folds (int): Number of folds for cross-validation.
    - patience (int): Patience for early stopping.
    - model (keras.Model): Keras model for training.
    - history (keras.callbacks.History): Training history.
    - train_generator (keras.utils.Sequence): Training data generator.
    - eval_generator (keras.utils.Sequence): Evaluation data generator.
    - test_generator (keras.utils.Sequence): Test data generator.

    Methods:
    - __init__: Initialize the ModelTrainer object.
    - initialize_generators: Initialize data generators for training and evaluation.
    - get_callbacks: Get list of callbacks for training the model.
    - create_model: Create and compile the model.
    - train: Train the model.
    - evaluate: Evaluate the model.
    - perform_cross_validation: Perform k-fold cross-validation.
    - save_model_and_history: Save the model and training history to disk.
    - process_cross_validation_metrics: Process cross-validation metrics and save them to a CSV file.
    - train_final_model: Train the final model on the full training and evaluation data.
    - run: Run the training and evaluation process.
    - initialize_fold_log: Initialize the fold log file.
    - update_fold_status: Update the status of a fold in the fold log file.
    - check_fold_status: Check the status of a fold in the fold log file.
    """

    def __init__(self, config_paths: List[str]) -> None:
        """
        Initialize the ModelTrainer object.

        Args:
        - config_paths (List[str]): List of configuration file paths.
        """
        self.configs = [parse_config(cp) for cp in config_paths]
        self.config = self.configs[0]

        self.training_config = self.config['training']
        self.log_config = self.config['log']

        # Extract training parameters
        self.frame_loss_function = self.training_config.get(
            'frame_loss_function', 'binary_crossentropy')
        self.utt_loss_function = self.training_config.get(
            'utt_loss_function', 'binary_crossentropy')
        self.frame_loss_weight = float(
            self.training_config.get('frame_loss_weight', 1.0))
        self.utt_loss_weight = float(
            self.training_config.get('utt_loss_weight', 1.0))
        self.optimizer_name = self.training_config.get('optimizer', 'adam')
        self.learning_rate = float(
            self.training_config.get('learning_rate', 0.001))
        self.epochs = int(self.training_config.get('epochs', 10))
        self.batch_size = int(self.training_config.get('batch_size', 32))
        self.k_folds = int(self.training_config.get('k_fold', 5))
        self.patience = int(self.training_config.get('patience', 5))

        # Initialize other attributes
        self.model = None
        self.history = None
        self.train_generator = None
        self.eval_generator = None
        self.test_generator = None

    def initialize_generators(self) -> None:
        """
        Initialize data generators for training and evaluation.
        """
        # Load datasets
        train_csv = self.config['data']['train_csv']
        eval_csv = self.config['data']['eval_csv']
        test_csv = self.config['data']['test_csv']

        train_features, train_labels = load_data_from_csv(train_csv)
        eval_features, eval_labels = load_data_from_csv(eval_csv)
        test_features, test_labels = load_data_from_csv(test_csv)

        # Combine data for cross-validation
        self.all_features = train_features + eval_features
        self.all_labels = train_labels + eval_labels

        # Calculate maxlen
        self.maxlen = get_max_sequence_length(
            self.all_features + test_features,
            self.all_labels + test_labels
        )

        # Initialize test data generator
        self.test_generator = CustomDataGenerator(
            test_features, test_labels, batch_size=self.batch_size, maxlen=self.maxlen
        )

    def get_callbacks(self, fold_no: int = None) -> List[tf.keras.callbacks.Callback]:
        """
        Get list of callbacks for training the model.

        Args:
        - fold_no (int): Fold number for cross-validation.
        """
        callbacks = []
        model_name = self.log_config["model_name"]
        checkpoint_dir = self.log_config["checkpoint_dir"]
        log_dir = self.log_config["log_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Model checkpointing
        if fold_no is not None:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"{model_name}_fold_{fold_no}_epoch_{{epoch:03d}}.keras")
            log_file = os.path.join(
                log_dir, f"{model_name}_fold_{fold_no}_training_log.csv")

        else:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"{model_name}_epoch_{{epoch:03d}}.keras")
            log_file = os.path.join(log_dir, f"{model_name}_training_log.csv")

        # Model checkpoint callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)

        # CSV logging callback
        csv_logger = tf.keras.callbacks.CSVLogger(log_file, append=True)
        callbacks.append(csv_logger)

        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # TensorBoard callback
        log_dir = self.log_config["log_dir"]
        if fold_no is not None:
            tensorboard_log_dir = os.path.join(
                log_dir, f"tensorboard_logs_fold_{fold_no}")
        else:
            tensorboard_log_dir = os.path.join(log_dir, "tensorboard_logs")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1
        )
        callbacks.append(tensorboard_callback)

        return callbacks

    def create_model(self, input_shape: Tuple[int, int], num_classes: int) -> keras.Model:
        """
        Create and compile the model.

        Args:
        - input_shape (Tuple[int, int]): Shape of the input data (timesteps, features).
        - num_classes (int): Number of output classes.

        Returns:
        - model (keras.Model): Compiled Keras model.
        """
        self.model = create_model(
            input_shape,
            num_classes,
            self.frame_loss_function,
            self.utt_loss_function,
            self.frame_loss_weight,
            self.utt_loss_weight,
            self.optimizer_name,
            self.learning_rate
        )

    def train(self, fold_no: int = None) -> None:
        """
        Train the model.

        Args:
        - fold_no (int): Fold number for cross-validation.
        """
        callbacks = self.get_callbacks(fold_no)
        initial_epoch = 0

        if fold_no is not None:
            # Check for existing checkpoint
            checkpoint_dir = self.log_config["checkpoint_dir"]
            model_name = self.log_config["model_name"]
            pattern = os.path.join(
                checkpoint_dir, f"{model_name}_fold_{fold_no}_epoch_*.keras")
            checkpoint_files = glob.glob(pattern)

            if checkpoint_files:
                # Get the latest checkpoint
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                print(f"Loading model from checkpoint: {latest_checkpoint}")
                self.model = tf.keras.models.load_model(
                    latest_checkpoint, custom_objects=self.get_custom_objects())

                # Extract initial epoch from filename
                epoch_match = re.search(r'epoch_(\d+)', latest_checkpoint)

                if epoch_match:
                    initial_epoch = int(epoch_match.group(1))

            else:
                print("No checkpoint found. Training from scratch.")

        self.history = self.model.fit(
            self.train_generator,
            initial_epoch=initial_epoch,
            epochs=self.epochs,
            validation_data=self.eval_generator,
            callbacks=callbacks
        )

    def evaluate(self, generator: keras.utils.Sequence, fold_no: int = None) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
        - generator (keras.utils.Sequence): Data generator for evaluation.
        - fold_no (int): Fold number for cross-validation.

        Returns:
        - results (Dict[str, float]): Evaluation results.
        """
        results = self.model.evaluate(generator, return_dict=True)
        print(f"Evaluation results for fold {fold_no}: {results}")
        return results

    def perform_cross_validation(self) -> None:
        """
        Perform k-fold cross-validation.

        This method performs k-fold cross-validation using the training and evaluation data
        loaded from the configuration file. It trains a model for each fold and evaluates
        it on the validation data. The evaluation results are saved in a list and processed
        after all folds are completed.

        The method also saves the model and training history for each fold.
        """
        self.initialize_fold_log()
        metrics_per_fold = self.load_metrics_per_fold()
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        fold_no = 1

        for train_idx, val_idx in kfold.split(self.all_features):
            status = self.check_fold_status(fold_no)
            if status == 'Completed':
                print(f"Fold {fold_no} is already completed. Skipping.")
                fold_no += 1
                continue
            else:
                print(f"Starting fold {fold_no}/{self.k_folds}")

            # Split data
            train_features_fold = [self.all_features[i] for i in train_idx]
            train_labels_fold = [self.all_labels[i] for i in train_idx]
            val_features_fold = [self.all_features[i] for i in val_idx]
            val_labels_fold = [self.all_labels[i] for i in val_idx]

            # Initialize data generators
            self.train_generator = CustomDataGenerator(
                train_features_fold, train_labels_fold,
                batch_size=self.batch_size, maxlen=self.maxlen
            )
            self.eval_generator = CustomDataGenerator(
                val_features_fold, val_labels_fold,
                batch_size=self.batch_size, maxlen=self.maxlen
            )

            # Determine input shape and num_classes
            input_shape = self.train_generator.get_input_shape()
            num_classes = self.train_generator.get_num_classes()

            # Create model
            self.create_model(input_shape, num_classes)

            # Train model
            self.train(fold_no)

            # Evaluate model
            results = self.evaluate(self.eval_generator, fold_no)
            metrics_per_fold.append(results)
            self.save_metrics_per_fold(metrics_per_fold)

            # Save model and history
            self.save_model_and_history(fold_no)

            # Update fold status to 'Completed'
            self.update_fold_status(fold_no, 'Completed')

            fold_no += 1

        # After cross-validation, process metrics
        self.process_cross_validation_metrics(metrics_per_fold)

    def save_metrics_per_fold(self, metrics_per_fold: List[Dict[str, float]]) -> None:
        """
        Save metrics per fold to a JSON file.

        Args:
        - metrics_per_fold (List[Dict[str, float]]): List of metrics for each fold.
        """
        metrics_path = os.path.join(
            self.log_config['log_dir'], f'{self.log_config["model_name"]}_metrics_per_fold.json')

        # Convert NumPy arrays to lists for serialization
        serializable_metrics = []
        for result in metrics_per_fold:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value
            serializable_metrics.append(serializable_result)

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f)

    def load_metrics_per_fold(self) -> List[Dict[str, float]]:
        """
        Load metrics per fold from a JSON file.

        Returns:
        - metrics_per_fold (List[Dict[str, float]]): List of metrics for each fold.
        """
        metrics_path = os.path.join(
            self.log_config['log_dir'], f'{self.log_config["model_name"]}_metrics_per_fold.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics_per_fold = json.load(f)
                return metrics_per_fold
        else:
            return []

    def save_model_and_history(self, fold_no: int = None) -> None:
        """
        Save the model and training history to disk.

        Args:
        - fold_no (int): Fold number for cross-validation.
        """
        model_name = self.log_config["model_name"]
        log_dir = self.log_config["log_dir"]
        model_dir = self.log_config["model_dir"]

        if fold_no is not None:
            model_name_log = f"{model_name}_fold_{fold_no}"
        else:
            model_name_log = model_name

        # Save the model
        model_path = os.path.join(model_dir, f"{model_name_log}.keras")
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")

    def process_cross_validation_metrics(self, metrics_per_fold: List[Dict[str, float]]) -> None:
        """
        Process cross-validation metrics and save them to a CSV file.

        Args:
        - metrics_per_fold (List[Dict[str, float]]): List of metrics for each fold.
        """
        metrics_df = pd.DataFrame(metrics_per_fold)
        metrics_df.insert(0, 'fold', range(1, self.k_folds + 1))

        # Calculate average metrics
        metrics_df['fold'] = metrics_df['fold'].astype(object)
        avg_metrics = metrics_df.mean(numeric_only=True)
        avg_metrics['fold'] = 'average'
        avg_metrics_df = avg_metrics.to_frame().T
        avg_metrics_df['fold'] = avg_metrics_df['fold'].astype(object)

        metrics_df = pd.concat([metrics_df, avg_metrics_df], ignore_index=True)

        # Save metrics
        log_dir = self.log_config['log_dir']
        model_name = self.log_config['model_name']
        metrics_file = os.path.join(
            log_dir, f"{model_name}_cross_validation_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Cross-validation metrics saved to {metrics_file}")

    def train_final_model(self) -> None:
        """
        Train the final model on the full training and evaluation data.
        """
        # Initialize data generators
        self.train_generator = CustomDataGenerator(
            self.all_features, self.all_labels, batch_size=self.batch_size, maxlen=self.maxlen
        )

        # Determine input shape and num_classes
        input_shape = self.train_generator.get_input_shape()
        num_classes = self.train_generator.get_num_classes()

        # Create model
        self.create_model(input_shape, num_classes)

        # Train model
        self.train()

        # Evaluate on test set
        results = self.evaluate(self.test_generator)
        print(f"Final model test results: {results}")

        # Save final model and history
        self.save_model_and_history()

    def run(self) -> None:
        """
        Run the training and evaluation process.
        """
        self.initialize_generators()
        self.perform_cross_validation()
        self.train_final_model()

    def get_custom_objects(self) -> Dict[str, Any]:
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

    def initialize_fold_log(self) -> None:
        """
        Initialize the fold log file.
        """
        self.fold_log_path = os.path.join(
            self.log_config['log_dir'], f'{self.log_config["model_name"]}_fold_status.log')
        if not os.path.exists(self.fold_log_path):
            # Initialize log file with all folds marked as Incomplete
            with open(self.fold_log_path, 'w', encoding='utf-8') as log_file:
                for fold_no in range(1, self.k_folds + 1):
                    log_file.write(f"Fold {fold_no}: Incomplete\n")

    def update_fold_status(self, fold_no: int, status: str) -> None:
        """
        Update the status of a fold in the fold log file.

        Args:
        - fold_no (int): Fold number.
        - status (str): New status for the fold.
        """
        # Read current statuses
        with open(self.fold_log_path, 'r', encoding='utf-8') as log_file:
            lines = log_file.readlines()

        # Update the status of the current fold
        with open(self.fold_log_path, 'w', encoding='utf-8') as log_file:
            for line in lines:
                if line.startswith(f"Fold {fold_no}:"):
                    log_file.write(f"Fold {fold_no}: {status}\n")
                else:
                    log_file.write(line)

    def check_fold_status(self, fold_no: int) -> str:
        """
        Check the status of a fold in the fold log file.

        Args:
        - fold_no (int): Fold number.

        Returns:
        - status (str): Status of the fold.
        """
        with open(self.fold_log_path, 'r', encoding='utf-8') as log_file:
            for line in log_file:
                if line.startswith(f"Fold {fold_no}:"):
                    status = line.strip().split(': ')[1]
                    return status

        return 'Incomplete'  # Default if not found
