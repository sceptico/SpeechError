"""
This script tests the custom frame-level loss function.
"""

import sys
import os
import unittest
import tensorflow as tf

try:
    from src.training.custom_frame_level_loss import custom_frame_level_loss
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.training.custom_frame_level_loss import custom_frame_level_loss


class TestCustomFrameLevelLoss(unittest.TestCase):

    def test_custom_frame_level_loss_no_mask(self):
        y_true_frame_1 = tf.constant(
            [[[1], [0]], [[1], [0]], [[1], [1]], [[0], [1]]], dtype=tf.float32)
        y_pred_frame_1 = tf.constant(
            [[[0.9], [0.3]], [[0.8], [0.1]], [[0.7], [0.2]], [[0.3], [0.7]]], dtype=tf.float32)
        loss_frame_1 = custom_frame_level_loss(
            y_true_frame_1, y_pred_frame_1).numpy()

        y_true_frame_2 = tf.constant(
            [[[1], [0]], [[1], [0]], [[1], [1]], [[0], [1]]], dtype=tf.float32)
        y_pred_frame_2 = tf.constant(
            [[[0.9], [0.3]], [[0.8], [0.1]], [[0.7], [0.2]], [[0.3], [0.7]]], dtype=tf.float32)
        loss_frame_2 = custom_frame_level_loss(
            y_true_frame_2, y_pred_frame_2).numpy()

        self.assertAlmostEqual(loss_frame_1, loss_frame_2, places=6)

    def test_custom_frame_level_loss_partially_masked(self):
        y_true_frame_1 = tf.constant(
            [[[1], [0]], [[0], [0]], [[1], [1]], [[0], [1]]], dtype=tf.float32)
        y_pred_frame_1 = tf.constant(
            [[[0.9], [0.3]], [[0.8], [0.1]], [[0.7], [0.2]], [[0.3], [0.7]]], dtype=tf.float32)
        loss_frame_1 = custom_frame_level_loss(
            y_true_frame_1, y_pred_frame_1).numpy()

        y_true_frame_2 = tf.constant(
            [[[1], [0]], [[1], [1]], [[0], [1]]], dtype=tf.float32)
        y_pred_frame_2 = tf.constant(
            [[[0.9], [0.3]], [[0.7], [0.2]], [[0.3], [0.7]]], dtype=tf.float32)
        loss_frame_2 = custom_frame_level_loss(
            y_true_frame_2, y_pred_frame_2).numpy()

        self.assertAlmostEqual(loss_frame_1, loss_frame_2, places=6)

    def test_custom_frame_level_loss_all_masked(self):
        y_true_empty = tf.constant(
            [[[0], [0]], [[0], [0]], [[0], [0]], [[0], [0]]], dtype=tf.float32)
        y_pred_empty = tf.constant(
            [[[0.9], [0.3]], [[0.8], [0.1]], [[0.7], [0.2]], [[0.3], [0.7]]], dtype=tf.float32)
        loss_empty = custom_frame_level_loss(
            y_true_empty, y_pred_empty).numpy()

        self.assertAlmostEqual(loss_empty, 1e-7, places=6)

    def test_custom_frame_level_loss_unsupported_rank(self):
        y_true_unsupported = tf.constant([1, 0, 1], dtype=tf.float32)
        y_pred_unsupported = tf.constant([0.9, 0.1, 0.3], dtype=tf.float32)
        with self.assertRaises(ValueError):
            custom_frame_level_loss(
                y_true_unsupported, y_pred_unsupported).numpy()


if __name__ == "__main__":
    print("Running custom frame-level loss tests...")
    print("----------------------------------------")
    unittest.main()
