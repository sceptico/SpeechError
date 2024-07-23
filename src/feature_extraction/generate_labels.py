import os
import argparse
import numpy as np
import pandas as pd
import configparser
import multiprocessing
import csv
import math
from typing import List, Tuple


class LabelEncoder:
    """
    The LabelEncoder class is used to generate labels for the audio features.

    Args:
    - config_path (str): The path to the configuration file.
    - annotations_path (str): Path to the csv file containing the annotations.
    - transcript_dir (str): Directory containing the transcript files.
    - feature_dir (str): Directory containing the audio features (.npy files).
    - label_dir (str): Directory where the labels will be saved.
    - label_info_dir (str): Directory where the label information will be saved.
    - n_process (int): Number of processes to use for generating the labels.
    - labels_to_keep (List[str]): The list of labels to keep. Ignoring all other labels.
    - multi_class (bool): Whether to use multi-class labels. If False, one class (speech error or non-speech error) is used.

    Attributes:
    - config_path (str): The path to the configuration file.
    - sr (int): The sample rate of the audio.
    - unique_labels (List[str]): The unique labels in the annotations.
    - hop_length_seconds (float): The hop length in seconds for the audio features.
    - feature_file_list (List[str]): The list of audio feature files.
    - transcript_file_list (List[str]): The list of transcript files.
    - label_info (List[Dict]): Information about the labels for each feature file.
    """

    def __init__(self, config_path: str, annotations_path: str, transcript_dir: str, feature_dir: str, label_dir: str, label_info_dir: str, n_process: int, labels_to_keep: List[str], multi_class: bool = False) -> None:
        self.config_path = config_path
        self.annotations_path = annotations_path
        self.transcript_dir = transcript_dir
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.label_info_dir = label_info_dir
        self.n_process = n_process
        self.labels_to_keep = labels_to_keep
        self.multi_class = multi_class

        self.sr = None
        self.unique_labels = []
        self.hop_length_seconds = None
        self.feature_file_list = []
        self.transcript_file_list = []
        self.label_info = []

        self._init_config()
        self._compile_list_of_feature_files()
        self._compile_list_of_transcript_files()
        self._sort_dataset_by_segment()

    def _init_config(self) -> None:
        """
        Initialize the configuration settings for the LabelEncoder.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found at {self.config_path}")

        config = configparser.ConfigParser()
        config.read(self.config_path)

        if 'feature' not in config:
            raise ValueError(f"Config file does not contain 'feature' section")

        feature_config = config['feature']

        self.sr = int(feature_config.get('sr', 16000))
        self.hop_length_seconds = float(feature_config.get('hop_length', 0.02))

    def _compile_list_of_feature_files(self) -> None:
        """
        Compile a list of audio feature files from the specified directory.

        Raises:
        - FileNotFoundError: If the feature directory is not found.
        """
        if not os.path.exists(self.feature_dir):
            raise FileNotFoundError(
                f"Feature directory not found at {self.feature_dir}")

        self.feature_file_list = []
        for root, _, files in os.walk(self.feature_dir):
            for file in files:
                if file.endswith('.npy'):
                    self.feature_file_list.append(os.path.join(root, file))

    def _compile_list_of_transcript_files(self) -> None:
        """
        Compile a list of transcript files from the specified directory.

        Raises:
        - FileNotFoundError: If the transcript directory is not found.
        """
        if not os.path.exists(self.transcript_dir):
            raise FileNotFoundError(
                f"Transcript directory not found at {self.transcript_dir}")

        self.transcript_file_list = []
        for root, _, files in os.walk(self.transcript_dir):
            for file in files:
                if file.endswith('.csv'):
                    self.transcript_file_list.append(os.path.join(root, file))

    def _sort_dataset_by_segment(self) -> None:
        """
        Sort the dataset by segment, and generate a new label_info file with the following:
        - feature_file, str: The path to the audio feature file, [audio_file]_[segment_number].npy.
        - start_time, float: The start time of the segment.
        - end_time, float: The end time of the segment.
        - label_list, List[(float, float, str)]: The list of labels for the segment:
            - start_time, float: The start time of the label.
            - end_time, float: The end time of the label.
            - label, str: The label for the segment.
        - label_count, int: The number of labels for the segment.

        The label_info file is saved to the label_info directory.

        Raises:
        - FileNotFoundError: If the annotations file is not found.
        - ValueError: If the required columns are not found in the annotations file.
        """
        print("Sorting dataset by segment...")

        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(
                f"Annotations file not found at {self.annotations_path}")

        if not os.path.exists(self.label_info_dir):
            os.makedirs(self.label_info_dir)

        annotations = pd.read_csv(self.annotations_path)
        required_columns = ['file', 'label', 'start', 'end']
        columns = annotations.columns.tolist()

        for column in required_columns:
            if column not in columns:
                raise ValueError(
                    f'{column} not found in the annotations file.')

        label_info = []
        feature_dict = {}

        for feature_file in self.feature_file_list:
            label_file_name = os.path.basename(
                feature_file).replace('.npy', '_labels.npy')
            label_file = os.path.join(self.label_dir, label_file_name)
            label_info.append({
                "feature_file": feature_file,
                "label_file": label_file,
                "start_time": 0,
                "end_time": 0,
                "label_list": [],
                "label_count": 0
            })

            feature_file_name = os.path.basename(feature_file)
            parts = feature_file_name.rsplit('_', 1)
            audio_file_name = parts[0]
            segment_number = int(parts[1].replace('.npy', ''))

            feature_dict.setdefault(audio_file_name, []).append({
                "feature_file": feature_file,
                "segment_number": segment_number - 1,
                "start_time": 0,
                "end_time": 0
            })

        for feature_audio_file, segments in feature_dict.items():
            transcript_file = None
            for file in self.transcript_file_list:
                if feature_audio_file in file:
                    transcript_file = file
                    break

            if transcript_file is None:
                raise FileNotFoundError(
                    f"Transcript file not found for {feature_audio_file}")

            with open(transcript_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    for segment in segments:
                        if reader.line_num - 2 == segment["segment_number"]:
                            segment["start_time"] = float(row[0])
                            segment["end_time"] = float(row[1])
                            for item in label_info:
                                if item['feature_file'] == segment["feature_file"]:
                                    item['start_time'] = segment["start_time"]
                                    item['end_time'] = segment["end_time"]
                                    break

        for index, row in annotations.iterrows():
            print(f"Processing annotation {index + 1}/{len(annotations)}")
            audio_file = os.path.basename(row['file']).split('.')[0]
            start_time = row['start']
            end_time = row['end']
            label = row['label']

            if audio_file in feature_dict:
                for segment in feature_dict[audio_file]:
                    feature_start_time = segment["start_time"]
                    feature_end_time = segment["end_time"]
                    feature_file = segment["feature_file"]

                    if start_time >= feature_start_time and end_time <= feature_end_time:
                        label_start_time = start_time
                        label_end_time = end_time
                        for item in label_info:
                            if item['feature_file'] == feature_file:
                                item['label_list'].append(
                                    (label_start_time, label_end_time, label))
                                item['label_count'] += 1
                                break
                        break

                    elif start_time >= feature_start_time and start_time < feature_end_time and end_time > feature_end_time:
                        label_start_time = start_time
                        label_end_time = feature_end_time
                        for item in label_info:
                            if item['feature_file'] == feature_file:
                                item['label_list'].append(
                                    (label_start_time, label_end_time, label))
                                item['label_count'] += 1
                                break
                        break

                    elif start_time < feature_start_time and end_time <= feature_end_time and end_time > feature_start_time:
                        label_start_time = feature_start_time
                        label_end_time = end_time
                        for item in label_info:
                            if item['feature_file'] == feature_file:
                                item['label_list'].append(
                                    (label_start_time, label_end_time, label))
                                item['label_count'] += 1
                                break
                        break

        self.label_info = label_info
        label_info_df = pd.DataFrame(label_info)
        label_info_path = os.path.join(self.label_info_dir, 'label_info.csv')
        label_info_df.to_csv(label_info_path, index=False)
        print(f"Label information saved to {label_info_path}")

    def _generate_labels_for_single_list(self, list_feature_files: List[str], index: int, file_per_process: int, feature_file_length: int) -> List[Tuple[str, float, float, int]]:
        """
        Generate labels for a list of audio feature files.

        Args:
        - list_feature_files (List[str]): The list of audio feature files to process.
        - index (int): The index of the process.
        - file_per_process (int): The number of files to process per process.
        - feature_file_length (int): The total number of feature files to process.

        Returns:
        - List[Tuple[str, float, float, int]]: List of tuples containing feature file, start time, end time, and label count.
        """
        results = []

        num_classes = len(self.labels_to_keep) if self.multi_class else 1

        for sub_list_index, feature_file in enumerate(list_feature_files):
            current_index = index * file_per_process + sub_list_index
            feature_file = feature_file['feature_file']
            feature_file_name = os.path.basename(feature_file)
            parts = feature_file_name.rsplit('_', 1)
            audio_file_name = parts[0]
            segment_number = int(parts[1].replace('.npy', ''))

            label_list = []
            start_time = 0
            end_time = 0
            label_file = ""

            for item in self.label_info:
                if item['feature_file'] == feature_file:
                    label_list = item['label_list']
                    start_time = item['start_time']
                    end_time = item['end_time']
                    label_file = item['label_file']
                    break

            feature = np.load(feature_file)
            feature_length = feature.shape[0]

            # Validate length of feature array
            feature_length_check = math.ceil(
                (end_time - start_time) / self.hop_length_seconds)
            if not (feature_length == feature_length_check or feature_length == feature_length_check + 1):
                print(
                    f"Feature length mismatch for {feature_file}; expected {feature_length_check}, got {feature_length}")

            labels = np.zeros((feature_length, num_classes))

            for label_start_time, label_end_time, label in label_list:
                label_start_index = int(
                    (label_start_time - start_time) / self.hop_length_seconds)
                label_end_index = int(
                    (label_end_time - start_time) / self.hop_length_seconds)

                if label in self.labels_to_keep:
                    if self.multi_class:
                        label_index = self.unique_labels.index(label)
                        labels[label_start_index:label_end_index, label_index] = 1
                    else:
                        labels[label_start_index:label_end_index, 0] = 1

            np.save(label_file, labels)

            results.append((feature_file, start_time,
                           item['end_time'], item['label_count']))

        print(
            f"Process {index + 1}/{min(self.n_process, feature_file_length)} ({len(list_feature_files)} labels) completed")
        return results

    def generate_all_labels_multiprocessing(self) -> None:
        """
        Generate labels for all the audio features using multiprocessing.

        Raises:
        - FileNotFoundError: If any of the paths do not exist.
        - ValueError: If the required columns are not found in the annotations file.
        """
        feature_file_length = len(self.label_info)
        file_per_process = (feature_file_length +
                            self.n_process - 1) // self.n_process

        pool = multiprocessing.Pool(processes=self.n_process)
        results = []

        for index in range(self.n_process):
            start_index = index * file_per_process
            end_index = min((index + 1) * file_per_process,
                            feature_file_length)

            if start_index < feature_file_length and start_index < end_index:
                sub_list = self.label_info[start_index:end_index]
                result = pool.apply_async(self._generate_labels_for_single_list, args=(
                    sub_list, index, file_per_process, feature_file_length))
                results.append(result)

        pool.close()
        pool.join()

        final_results = [result.get() for result in results]
        final_results = [item for sublist in final_results for item in sublist]

        label_info_df = pd.DataFrame(final_results, columns=[
                                     'feature_file', 'start_time', 'end_time', 'label_count'])
        label_info_path = os.path.join(
            self.label_info_dir, 'final_label_info.csv')
        label_info_df.to_csv(label_info_path, index=False)
        print(f"Final label information saved to {label_info_path}")


def generate_label(config_path: str, annotations_path: str, transcript_dir: str, feature_dir: str, label_dir: str, label_info_dir: str, n_process: int, labels_to_keep: List[str], multi_class: bool) -> None:
    """
    Generate labels for the audio features.

    Args:
    - config_path (str): Path to the configuration file.
    - annotations_path (str): Path to the csv file containing the annotations.
    - transcript_dir (str): Directory containing the transcript files generated by WhisperX.
    - feature_dir (str): Directory containing the audio features (.npy files) generated by the feature extraction script.
    - label_dir (str): Directory where the labels will be saved.
    - label_info_dir (str): Directory where the labels information will be saved.
    - n_process (int): Number of processes to use for feature extraction.
    - labels_to_keep (List[str]): The list of labels to keep. Ignoring all other labels.
    - multi_class (bool): Whether to use multi-class labels. If False, one class (speech error or non-speech error) is used.
    """
    label_encoder = LabelEncoder(config_path, annotations_path, transcript_dir,
                                 feature_dir, label_dir, label_info_dir, n_process, labels_to_keep, multi_class)
    label_encoder.generate_all_labels_multiprocessing()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate labels for the training data.')
    parser.add_argument('--feature_config', type=str,
                        help='Path to the configuration file', required=True)
    parser.add_argument('--annotations_path', type=str,
                        help='Path to the csv file containing the annotations', required=True)
    parser.add_argument('--transcript_dir', type=str,
                        help='Directory containing the transcript files generated by WhisperX', required=True)
    parser.add_argument('--feature_dir', type=str,
                        help='Directory containing the audio features (.npy files) generated by the feature extraction script', required=True)
    parser.add_argument('--label_dir', type=str,
                        help='Directory where the labels will be saved', required=True)
    parser.add_argument('--label_info_dir', type=str,
                        help='Directory where the labels information will be saved', required=True)
    parser.add_argument('--n_process', type=int, dest='n_process',
                        help='Number of processes to use for feature extraction.', default=4)
    parser.add_argument('--multi_class', action='store_true',
                        help='Whether to use multi-class labels. If False, one class (speech error or non-speech error) is used.', default=False)

    arguments = parser.parse_args()

    config_path = arguments.feature_config
    annotations_path = arguments.annotations_path
    transcript_dir = arguments.transcript_dir
    feature_dir = arguments.feature_dir
    label_dir = arguments.label_dir
    label_info_dir = arguments.label_info_dir
    n_process = arguments.n_process
    multi_class = arguments.multi_class

    labels_to_keep = ['Phonological Addition',
                      'Phonological Deletion', 'Phonological Substitution']

    paths = [config_path, annotations_path, transcript_dir,
             feature_dir, label_dir, label_info_dir]

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path "{path}" not found.')

    generate_label(config_path, annotations_path, transcript_dir, feature_dir,
                   label_dir, label_info_dir, n_process, labels_to_keep, multi_class)
