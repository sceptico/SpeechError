"""
custom_f1_score.py

This script defines a custom F1 score metric for the model training process.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


class CustomF1Score(tf.keras.metrics.Metric):
    """
    Custom F1 score metric for the model training process.

    Attributes:
    - name: Name of the metric.
    - true_positives: Number of true positives.
    - false_positives: Number of false positives.
    - false_negatives: Number of false negatives.

    Methods:
    - __init__: Initializes the metric.
    - update_state: Updates the state of the metric.
    - result: Calculates the F1 score.
    - reset_states: Resets the state of the metric.
    """

    def __init__(self, name: str = "custom_f1_score", **kwargs) -> None:
        """
        Initializes the metric.

        Args:
        - name (str): Name of the metric.
        """
        super(CustomF1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        """
        Updates the state of the metric.

        Args:
        - y_true (tf.Tensor): True labels.
        - y_pred (tf.Tensor): Predicted labels.
        - sample_weight: Optional weighting of samples.
        """
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_pred = tf.round(y_pred)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum(y_pred) - tp
        fn = tf.reduce_sum(y_true) - tp

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self) -> tf.Tensor:
        """
        Calculates the F1 score.

        Returns:
        - f1_score (tf.Tensor): The computed F1 score.
        """
        precision = self.true_positives / \
            (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / \
            (self.true_positives + self.false_negatives + K.epsilon())
        f1_score = 2 * (precision * recall) / \
            (precision + recall + K.epsilon())
        return f1_score

    def reset_states(self) -> None:
        """
        Resets the state of the metric.
        """
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)
