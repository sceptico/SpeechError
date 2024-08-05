import tensorflow as tf
from tensorflow.keras import backend as K


class CustomF1Score(tf.keras.metrics.Metric):
    def __init__(self, name="custom_f1_score", **kwargs):
        super(CustomF1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_pred = tf.round(y_pred)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum(y_pred) - tp
        fn = tf.reduce_sum(y_true) - tp

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / \
            (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / \
            (self.true_positives + self.false_negatives + K.epsilon())
        f1_score = 2 * (precision * recall) / \
            (precision + recall + K.epsilon())
        return f1_score

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)
