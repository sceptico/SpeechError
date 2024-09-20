"""
custom_error_rate_metric.py

Custom error rate metric and logging callback for the model training process.
"""

import tensorflow as tf
from tensorflow.keras.metrics import Metric, FalsePositives, FalseNegatives, TruePositives, TrueNegatives

THRESHOLD = 0.5


class CustomErrorRateMetric(Metric):
    """
    Custom error rate metric for the model training process.

    This metric calculates the error rate based on the number of substitutions, deletions, and insertions.

    Attributes:
    - name: Name of the metric.
    - false_positives: FalsePositives metric.
    - false_negatives: FalseNegatives metric.
    - true_positives: TruePositives metric.
    - true_negatives: TrueNegatives metric.

    Methods:
    - __init__: Initializes the metric.
    - update_state: Updates the state of the metric.
    - result: Calculates the error rate.
    - reset_states: Resets the state of the metric.
    """

    def __init__(self, name: str = 'custom_error_rate', **kwargs):
        """
        Initializes the metric.

        Args:
        - name (str): Name of the metric.
        """
        super(CustomErrorRateMetric, self).__init__(name=name, **kwargs)
        self.false_positives = FalsePositives()
        self.false_negatives = FalseNegatives()
        self.true_positives = TruePositives()
        self.true_negatives = TrueNegatives()

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        """
        Updates the state of the metric.

        Args:
        - y_true (tf.Tensor): True labels.
        - y_pred (tf.Tensor): Predicted labels.
        - sample_weight: Optional weighting of samples.
        """
        # Predictions are binary based on the threshold
        y_pred = tf.cast(y_pred >= THRESHOLD, dtype=tf.float32)

        # Update the state of the internal metrics
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self) -> tf.Tensor:
        """
        Calculates the error rate based on the number of substitutions, deletions, and insertions.

        Returns:
        - ER (tf.Tensor): Error rate.
        """
        # Get the results of the internal metrics
        FP = tf.cast(self.false_positives.result(), dtype=tf.float32)
        FN = tf.cast(self.false_negatives.result(), dtype=tf.float32)
        TP = tf.cast(self.true_positives.result(), dtype=tf.float32)
        TN = tf.cast(self.true_negatives.result(), dtype=tf.float32)

        # Calculate substitutions (S), deletions (D), and insertions (I)
        S = tf.minimum(FN, FP)
        D = tf.maximum(0.0, FN - FP)
        I = tf.maximum(0.0, FP - FN)

        # Calculate total number of active sound events (N)
        N = TP + FN

        # Calculate error rate (ER)
        ER = (S + D + I) / (N + tf.keras.backend.epsilon())

        return ER

    def reset_states(self) -> None:
        """
        Reset the state of the internal metrics.
        """
        # Reset the state of the internal metrics
        self.false_positives.reset_states()
        self.false_negatives.reset_states()
        self.true_positives.reset_states()
        self.true_negatives.reset_states()
