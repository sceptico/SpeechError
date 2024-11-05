"""
Split data into training, evaluation, and testing sets.

This script reads the label_info_context.csv file and splits the data into training, evaluation, and testing sets.

It only considers samples with events such as 'Phonological Addition', 'Phonological Deletion', and 'Phonological Substitution'.
i.e. sound errors in the speech data.


Usage:
    python split_data.py --label_info_path <label_info_context_path> --output_dir <output_dir> --eval_ratio <eval_ratio> --test_ratio <test_ratio>

    - label_info_path (str): The path to the label_info_context.csv file.
    - output_dir (str): The directory to save the split data.
    - eval_ratio (float): The ratio of samples to include in the evaluation set.
    - test_ratio (float): The ratio of samples to include in the testing set.
    
Example:
    python split_data.py --label_info_path /data/metadata/label_info_context.csv --output_dir /data/metadata --eval_ratio 0.1 --test_ratio 0.1
"""

import os
import csv
import random
from typing import List
import argparse


def split_data(label_info_path: str, output_dir: str, eval_ratio: float = 0.1, test_ratio: float = 0.1) -> None:
    """
    Split the data into training, evaluation, and testing sets.

    Args:
    - label_info_path (str): The path to the label_info_context.csv file.
    - output_dir (str): The directory to save the split CSV files.
    - eval_ratio (float): The ratio of samples to include in the evaluation set.
    - test_ratio (float): The ratio of samples to include in the testing set.
    """
    # Load the label_info file
    with open(label_info_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Get the headers
        label_info = list(reader)

    # Get the list of samples with events
    # Only consider these events: ['Phonological Addition', 'Phonological Deletion', 'Phonological Substitution']
    events_to_consider = ['Phonological Addition',
                          'Phonological Deletion', 'Phonological Substitution']
    label_index = headers.index('label_list')
    samples_with_events = [row for row in label_info if any(
        event in row[label_index] for event in events_to_consider)]
    samples_without_events = [row for row in label_info if all(
        event not in row[label_index] for event in events_to_consider)]

    # Shuffle the samples
    random.shuffle(samples_with_events)
    random.shuffle(samples_without_events)

    # Calculate the number of samples in each set
    num_samples = len(samples_with_events)
    num_eval_samples = int(num_samples * eval_ratio)
    num_test_samples = int(num_samples * test_ratio)
    num_train_samples = num_samples - num_eval_samples - num_test_samples

    # Split the samples into sets
    eval_samples = samples_with_events[:num_eval_samples]
    test_samples = samples_with_events[num_eval_samples:
                                       num_eval_samples + num_test_samples]
    train_samples = samples_with_events[num_eval_samples + num_test_samples:]

    eval_sample_event_size = len(eval_samples)
    test_sample_event_size = len(test_samples)
    train_sample_event_size = len(train_samples)

    # Combine the samples with and without events
    eval_samples += samples_without_events[:int(
        num_eval_samples * len(samples_without_events) / num_samples)]
    test_samples += samples_without_events[int(num_eval_samples * len(samples_without_events) / num_samples):int(
        (num_eval_samples + num_test_samples) * len(samples_without_events) / num_samples)]
    train_samples += samples_without_events[int(
        (num_eval_samples + num_test_samples) * len(samples_without_events) / num_samples):]

    eval_sample_total_size = len(eval_samples)
    test_sample_total_size = len(test_samples)
    train_sample_total_size = len(train_samples)

    print(
        f"Number of samples in the training set: {train_sample_total_size}; (with events: {train_sample_event_size})")
    print(
        f"Number of samples in the evaluation set: {eval_sample_total_size}; (with events: {eval_sample_event_size})")
    print(
        f"Number of samples in the testing set: {test_sample_total_size}; (with events: {test_sample_event_size})")

    # Save the split CSV files with new names to avoid overwriting original files
    save_split_csv(headers, train_samples, os.path.join(output_dir, 'train_context.csv'))
    save_split_csv(headers, eval_samples, os.path.join(output_dir, 'eval_context.csv'))
    save_split_csv(headers, test_samples, os.path.join(output_dir, 'test_context.csv'))


def save_split_csv(headers: List[str], samples: List[List[str]], output_path: str) -> None:
    """
    Save the split samples to a CSV file.

    Args:
    - headers (List[str]): The header row for the CSV file.
    - samples (List[List[str]]): The samples to save.
    - output_path (str): The path to save the CSV file.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # Write the headers
        writer.writerows(samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split data into train, eval, and test sets")
    parser.add_argument('--label_info_path', type=str,
                        required=True, help="Path to the label_info_resampled.csv file")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the split data")
    parser.add_argument('--eval_ratio', type=float,
                        default=0.1, help="Ratio of eval data")
    parser.add_argument('--test_ratio', type=float,
                        default=0.1, help="Ratio of test data")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    label_info_path = args.label_info_path
    output_dir = args.output_dir
    eval_ratio = args.eval_ratio
    test_ratio = args.test_ratio
    seed = args.seed

    random.seed(seed)

    split_data(label_info_path, output_dir, eval_ratio, test_ratio)
