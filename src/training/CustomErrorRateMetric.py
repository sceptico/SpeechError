import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanMetricWrapper, Metric, FalsePositives, FalseNegatives, TruePositives, TrueNegatives

THRESHOLD = 0.5


class CustomErrorRateMetric(Metric):
    def __init__(self, name='error_rate', **kwargs):
        super(CustomErrorRateMetric, self).__init__(name=name, **kwargs)
        self.false_positives = FalsePositives()
        self.false_negatives = FalseNegatives()
        self.true_positives = TruePositives()
        self.true_negatives = TrueNegatives()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Predictions are binary based on the threshold
        y_pred = tf.cast(y_pred >= THRESHOLD, dtype=tf.float32)

        # Update the state of the internal metrics
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self):
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

    def reset_states(self):
        # Reset the state of the internal metrics
        self.false_positives.reset_states()
        self.false_negatives.reset_states()
        self.true_positives.reset_states()
        self.true_negatives.reset_states()


class ErrorRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        tf.print(
            f"\nBatch {batch}:")
        tf.print("-" * 50)
        # for key, value in logs.items():
        #     tf.print(f"\t{key}: {value}")
        for key in ['output_layer_frame_error_rate',
                    'output_layer_frame_FalseNegatives',
                    'output_layer_frame_FalsePositives',
                    'output_layer_frame_TrueNegatives',
                    'output_layer_frame_TruePositives']:
            tf.print(f"\t{key}: {logs[key]}")
