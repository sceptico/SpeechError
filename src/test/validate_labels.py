import os
import unittest
import numpy as np
import csv
import ast


class TestLabels(unittest.TestCase):

    def setUp(self):
        # Load the labels and info_labels
        self.label_file_no_labels_name = 'data/labels/ac083_2008-04-06_0001_labels.npy'
        self.label_file_one_label_name = 'data/labels/bp249_2011-12-25_0045_labels.npy'
        self.label_info_path = 'data/metadata/label_info.csv'

        # Get the base directory (two level up)
        base_dir = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))

        self.label_file_no_labels_path = os.path.join(
            base_dir, self.label_file_no_labels_name)
        self.label_file_one_label_path = os.path.join(
            base_dir, self.label_file_one_label_name)
        self.label_info_path = os.path.join(base_dir, self.label_info_path)

        self.label_file_no_labels = np.load(self.label_file_no_labels_path)
        self.label_file_one_label = np.load(self.label_file_one_label_path)

        with open(self.label_info_path, 'r') as f:
            self.label_info = list(csv.reader(f))

        self.hop_length_seconds = 0.02
        self.num_classes = self.label_file_no_labels.shape[0]

    def test_no_labels_length(self):
        # Search for the row containing the label file in the label_info file
        start_time = 0
        end_time = 0
        for row in self.label_info:
            if row[1] == self.label_file_no_labels_name:
                start_time = float(row[2])
                end_time = float(row[3])
                break

        # Check if the length of the label file is correct
        expected_length = int((end_time - start_time) /
                              self.hop_length_seconds)
        self.assertAlmostEqual(
            self.label_file_no_labels.shape[0], expected_length, delta=1)

    def test_one_label_length(self):
        # Search for the row containing the label file in the label_info file
        start_time = 0
        end_time = 0
        for row in self.label_info:
            if row[1] == self.label_file_one_label_name:
                start_time = float(row[2])
                end_time = float(row[3])
                break

        # Check if the length of the label file is correct
        expected_length = int((end_time - start_time) /
                              self.hop_length_seconds)
        self.assertAlmostEqual(
            self.label_file_one_label.shape[0], expected_length, delta=1)

    def test_no_labels_values(self):
        # Check if the label file contains only zeros
        self.assertEqual(np.sum(self.label_file_no_labels), 0)

    def test_one_label_values(self):
        # Check if the label file contains the 1st label at the correct index
        # Look for start time in the label_info file
        start_time = 0
        end_time = 0
        for row in self.label_info:
            if row[1] == self.label_file_one_label_name:
                segment_start_time = float(row[2])
                label_list = ast.literal_eval(row[4])
                start_time = float(label_list[0][0]) - segment_start_time
                end_time = float(label_list[0][1]) - segment_start_time
                break

        # Check if the label file contains only zeros except for the label
        # at the correct index
        expected_start_index = int(start_time / self.hop_length_seconds)
        expected_end_index = int(end_time / self.hop_length_seconds)
        expected_array = np.zeros(self.label_file_one_label.shape)
        expected_array[expected_start_index:expected_end_index, 0] = 1

        self.assertTrue(np.array_equal(
            self.label_file_one_label, expected_array))


if __name__ == '__main__':
    unittest.main()
