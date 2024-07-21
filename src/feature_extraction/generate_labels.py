import os
import argparse
import numpy as np
import pandas as pd
import configparser
import multiprocessing
import csv
from typing import List, Tuple


class LabelEncoder:
    """
    The LabelEncoder class is used to generate labels for the audio features.

    Args:
    - config_path (str): The path to the configuration file.

    Attributes:
    - config_path (str): The path to the configuration file.
    - sr (int): The sample rate of the audio.
    - unique_labels (List[str]): The unique labels in the annotations.
    - hop_length_seconds (float): The hop length in seconds for the audio features.    
    - feature_file_list (List[str]): The list of audio feature files.
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.sr = None
        self.unique_labels = []
        self.hop_length_seconds = None
        self.feature_file_list = []
        self.transcript_file_list = []
        self._init_config()

    def _init_config(self) -> None:
        """
        Initialize the configuration settings for the LabelEncoder.        
        """
        config_path = self.config_path

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}")

        config = configparser.ConfigParser()
        config.read(config_path)

        if 'feature' not in config:
            raise ValueError(f"Config file does not contain 'feature' section")

        feature_config = config['feature']

        self.sr = int(feature_config.get('sr', 16000))
        self.hop_length_seconds = float(feature_config.get('hop_length', 0.02))

    def compile_list_of_feature_files(self, feature_dir: str) -> None:
        """
        Compile a list of audio feature files from the specified directory.

        Args:
        - feature_dir (str): The directory containing the audio features (.npy files).

        Raises:
        - FileNotFoundError: If the feature directory is not found.
        """
        if not os.path.exists(feature_dir):
            raise FileNotFoundError(
                f"Feature directory not found at {feature_dir}")

        self.feature_file_list = []
        for root, _, files in os.walk(feature_dir):
            for file in files:
                if file.endswith('.npy'):
                    self.feature_file_list.append(os.path.join(root, file))

    def compile_list_of_transcript_files(self, transcript_dir: str) -> None:
        """
        Compile a list of transcript files from the specified directory.

        Args:
        - transcript_dir (str): The directory containing the transcript files.

        Raises:
        - FileNotFoundError: If the transcript directory is not found.
        """
        if not os.path.exists(transcript_dir):
            raise FileNotFoundError(
                f"Transcript directory not found at {transcript_dir}")

        self.transcript_file_list = []
        for root, _, files in os.walk(transcript_dir):
            for file in files:
                if file.endswith('.csv'):
                    self.transcript_file_list.append(os.path.join(root, file))

    def sort_dataset_by_segment(self, annotations_path: str, label_info_dir: str) -> None:
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

        Args:
        - annotations_path (str): The path to the annotations file.
        - feature_dir (str): The directory containing the audio features (.npy files).
        - label_info_dir (str): The directory where the label information will be saved.

        Raises:
        - FileNotFoundError: If the annotations file is not found.
        - ValueError: If the required columns are not found in the annotations file.
        """
        print("Sorting dataset by segment...")
        feature_files = self.feature_file_list
        transcript_files = self.transcript_file_list

        # Check if directories exist
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(
                f"Annotations file not found at {annotations_path}")

        if not os.path.exists(label_info_dir):
            os.makedirs(label_info_dir)

        # Read the annotations file
        annotations = pd.read_csv(annotations_path)
        required_columns = ['file', 'label', 'start', 'end']
        columns = annotations.columns.tolist()

        for column in required_columns:
            if column not in columns:
                raise ValueError(
                    f'{column} not found in the annotations file.')

        # Initialize the label_info dictionary
        label_info = []

        # Create a dictionary for quick lookup
        feature_dict = {}

        for feature_file in feature_files:
            label_info.append({
                "feature_file": feature_file,
                "label_file": None,
                "start_time": 0,
                "end_time": 0,
                "label_list": [],
                "label_count": 0
            })

            feature_file_name = os.path.basename(feature_file)
            parts = feature_file_name.rsplit('_', 1)
            audio_file_name = parts[0]
            segment_number = parts[1].replace('.npy', '')

            feature_dict.setdefault(audio_file_name, []).append({
                "feature_file": feature_file,
                # Segment numbers are 0-indexed in transcripts
                "segment_number": int(segment_number) - 1,
                "start_time": 0,
                "end_time": 0
            })

        # Find start and end time for each segment by reading the transcript files
        for feature_audio_file, segments in feature_dict.items():
            transcript_file = None
            for file in transcript_files:
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

        # Iterate through all the annotations and add labels to the correct segment
        for index, row in annotations.iterrows():
            print(f"Processing annotation {index + 1}/{len(annotations)}")
            audio_file = os.path.basename(row['file']).split('.')[0]
            start_time = row['start']
            end_time = row['end']
            label = row['label']

            # Search for the feature file that contains the annotation based on the audio file name, start time, and end time
            if audio_file in feature_dict:
                for segment in feature_dict[audio_file]:
                    feature_start_time = segment["start_time"]
                    feature_end_time = segment["end_time"]
                    feature_file = segment["feature_file"]

                    # Three cases to consider:
                    # 1. The annotation is within the segment
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

                    # 2. The annotation starts within the segment, but ends after the segment
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

                    # 3. The annotation starts before the segment, but ends within the segment
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

        # Save the label_info to a csv file
        label_info_df = pd.DataFrame(label_info)
        label_info_path = os.path.join(label_info_dir, 'label_info.csv')
        label_info_df.to_csv(label_info_path, index=False)
        print(f"Label information saved to {label_info_path}")

    def _generate_zero_labels(self, num_frames: int) -> np.ndarray:
        """
        Generate zero labels for the given number of frames.

        Args:
        - num_frames (int): The number of frames to generate labels for.

        Returns:
        - labels (np.ndarray): The generated zero labels.
        """
        unique_labels = self.unique_labels
        labels = np.zeros((num_frames, len(unique_labels)), dtype=np.int32)
        return labels

    def _generate_labels_per_segment(self, feature_file: str, annotations_path: str, transcript_dir: str, label_dir: str) -> Tuple[str, float, float, int]:
        """
        Generate labels for a single segment of the audio feature.

        Args:
        - feature_file (str): The path to the audio feature file.
        - annotations_path (str): The path to the annotations file.
        - transcript_dir (str): The directory containing the transcript files.
        - label_dir (str): The directory where the labels will be saved.

        Returns:
        - feature_file (str): The path to the audio feature file.
        - start_time (float): The start time of the segment.
        - end_time (float): The end time of the segment.
        - label_count (int): The number of labels for the segment.
        """
        feature_file_name = os.path.basename(feature_file)
        parts = feature_file_name.rsplit('_', 1)
        audio_file_name = parts[0]
        segment_number = parts[1].replace('.npy', '')
        segment_index = int(segment_number) - 1

        # Find the start and end time for the segment
        transcript_file = None
        for root, _, files in os.walk(transcript_dir):
            for file in files:
                if audio_file_name in file:
                    transcript_file = os.path.join(root, file)

        if transcript_file is None:
            raise FileNotFoundError(
                f"Transcript file not found for {audio_file_name}")

        start_time = 0
        end_time = 0
        with open(transcript_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                if i == segment_index:
                    start_time = float(row[0])
                    end_time = float(row[1])
                    break

        feature = np.load(feature_file)
        num_frames = feature.shape[0]
        labels = self._generate_zero_labels(num_frames)

        # Add labels to the correct segment based on the audio file, start time, and end time
        annotations = pd.read_csv(annotations_path)
        for index, row in annotations.iterrows():
            annotation_file_name = os.path.basename(row['file']).split('.')[0]
            if annotation_file_name == audio_file_name:
                event_index = self.unique_labels.index(row['label'])

                event_start_frame = max(
                    int((row['start'] - start_time) / self.hop_length_seconds), 0)
                event_end_frame = min(
                    int((row['end'] - start_time) / self.hop_length_seconds), num_frames)

                if event_start_frame < num_frames and event_end_frame > 0:
                    labels[event_start_frame:event_end_frame, event_index] = 1

        label_file_name = feature_file_name.replace('.npy', '_label.npy')
        label_file_path = os.path.join(label_dir, label_file_name)
        np.save(label_file_path, labels)

        label_count = 0
        for i in range(labels.shape[1]):
            if np.sum(labels[:, i]) > 0:
                label_count += 1

        return feature_file, start_time, end_time, label_count

    def _generate_labels_for_single_list(self, list_features_files: List[str], annotations_path: str, transcript_dir: str, label_dir: str, index: int, file_per_process: int, feature_file_length: int) -> List[Tuple[str, float, float, int]]:
        """
        Generate labels for a list of audio feature files.

        Args:
        - list_features_files (List[str]): The list of audio feature files.
        - annotations_path (str): The path to the annotations file.
        - transcript_dir (str): The directory containing the transcript files.
        - label_dir (str): The directory where the labels will be saved.
        - index (int): The index of the process.
        - file_per_process (int): The number of files to process per process.
        - feature_file_length (int): The total number of feature files.

        Returns:
        - results (List[Tuple[str, float, float, int]]): The results of the label generation:
            - feature_file (str): The path to the audio feature file.
            - start_time (float): The start time of the segment.
            - end_time (float): The end time of the segment.
            - label_count (int): The number of labels for the segment.
        """
        process_index = index
        results = []

        for sub_list_index, feature_file in enumerate(list_features_files):
            index = process_index * file_per_process + sub_list_index
            label_file_name = os.path.basename(
                feature_file).replace('.npy', '_label.npy')
            label_file_path = os.path.join(label_dir, label_file_name)

            print(
                f"Generating labels {index + 1}/{feature_file_length}: {feature_file}")

            if not os.path.exists(label_file_path):
                result = self._generate_labels_per_segment(
                    feature_file, annotations_path, transcript_dir, label_dir)
                results.append(result)
                print(
                    f"Generating labels {index + 1}/{feature_file_length}: {feature_file} completed")

            else:
                print(
                    f"Generating labels {index + 1}/{feature_file_length}: {feature_file} already exists")

        return results

    def generate_all_labels_multiprocessing(self, wav_list: str, annotations_path: str, transcript_dir: str, feature_dir: str, label_dir: str, label_info_dir: str, n_process: int) -> None:
        """
        Generate labels for all the audio features using multiprocessing. It also generates a labels information file and saves it to the specified directory.

        Args:
        - wav_list (str): The list of audio files to extract features from.
        - annotations_path (str): The path to the annotations file.
        - transcript_dir (str): The directory containing the transcript files generated by WhisperX.
        - feature_dir (str): The directory containing the audio features (.npy files) generated by the feature extraction script.
        - label_dir (str): The directory where the labels will be saved.
        - label_info_dir (str): The directory where the labels information will be saved.
        - n_process (int): The number of processes to use for generating the labels.

        Raises:
        - FileNotFoundError: If any of the paths do not exist.
        - ValueError: If the required columns are not found in the annotations file.
        """
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        paths = [annotations_path, transcript_dir, feature_dir]

        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f'{path} not found.')

        annotations = pd.read_csv(annotations_path)

        required_columns = ['file', 'label', 'start', 'end']
        columns = annotations.columns.tolist()

        for column in required_columns:
            if column not in columns:
                raise ValueError(
                    f'{column} not found in the annotations file.')

        self.unique_labels = annotations['label'].unique().tolist()

        with open(wav_list, 'r') as f:
            wav_files = f.readlines()

        wav_list = [wav.strip() for wav in wav_files]

        feature_files = []
        for root, _, files in os.walk(feature_dir):
            for file in files:
                if file.endswith('.npy'):
                    feature_files.append(os.path.join(root, file))

        file_per_process = (len(feature_files) + n_process - 1) // n_process

        with multiprocessing.Pool(n_process) as pool:
            results = []
            for process_index in range(n_process):
                start_index = process_index * file_per_process
                end_index = min((process_index + 1) *
                                file_per_process, len(feature_files))

                if start_index < len(feature_files) and start_index < end_index:
                    feature_files_sub_list = feature_files[start_index:end_index]
                    result = pool.apply_async(self._generate_labels_for_single_list, args=(
                        feature_files_sub_list, annotations_path, transcript_dir, label_dir, process_index, file_per_process, len(feature_files)))
                    results.append(result)

            pool.close()
            pool.join()

            all_results = [item for sublist in [res.get() for res in results]
                           for item in sublist]

        self.labels_info_df = pd.DataFrame(
            all_results, columns=['feature_file', 'start_time', 'end_time', 'label_count'])

        labels_info_path = os.path.join(label_info_dir, 'labels_info.csv')
        self.labels_info_df.to_csv(labels_info_path, index=False)
        print(self.labels_info_df.head())
        print(f"Labels information saved to {labels_info_path}")

    def generate_all_labels_single_process(self, wav_list: str, annotations_path: str, transcript_dir: str, feature_dir: str, label_dir: str, label_info_dir: str) -> None:
        """
        Generate labels for all the audio features using a single process. It also generates a labels information file and saves it to the specified directory.

        Args:
        - wav_list (str): The list of audio files to extract features from.
        - annotations_path (str): The path to the annotations file.
        - transcript_dir (str): The directory containing the transcript files generated by WhisperX.
        - feature_dir (str): The directory containing the audio features (.npy files) generated by the feature extraction script.
        - label_dir (str): The directory where the labels will be saved.
        - label_info_dir (str): The directory where the labels information will be saved.

        Raises:
        - FileNotFoundError: If any of the paths do not exist.
        - ValueError: If the required columns are not found in the annotations file.
        """
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        paths = [annotations_path, transcript_dir, feature_dir]

        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f'{path} not found.')

        annotations = pd.read_csv(annotations_path)

        required_columns = ['file', 'label', 'start', 'end']
        columns = annotations.columns.tolist()

        for column in required_columns:
            if column not in columns:
                raise ValueError(
                    f'{column} not found in the annotations file.')

        self.unique_labels = annotations['label'].unique().tolist()

        with open(wav_list, 'r') as f:
            wav_files = f.readlines()

        wav_list = [wav.strip() for wav in wav_files]

        feature_files = []
        for root, _, files in os.walk(feature_dir):
            for file in files:
                if file.endswith('.npy'):
                    feature_files.append(os.path.join(root, file))

        all_results = []
        for index, feature_file in enumerate(feature_files):
            print(
                f"Processing {index + 1}/{len(feature_files)}: {feature_file}")
            result = self._generate_labels_per_segment(
                feature_file, annotations_path, transcript_dir, label_dir)
            all_results.append(result)

        self.labels_info_df = pd.DataFrame(
            all_results, columns=['feature_file', 'start_time', 'end_time', 'label_count'])

        labels_info_path = os.path.join(label_info_dir, 'labels_info.csv')
        self.labels_info_df.to_csv(labels_info_path, index=False)
        print(self.labels_info_df.head())
        print(f"Labels information saved to {labels_info_path}")


def generate_label(wav_list: str, annotation_path: str, transcript_dir: str, feature_dir: str, label_dir: str, label_info_dir: str, feature_config: str, n_process: int) -> None:
    """
    Generate labels for the audio features.

    Args:
    - wav_list (str): List of audio files to extract features from.
    - annotation_path (str): Path to the csv file containing the annotations.
    - transcript_dir (str): Directory containing the transcript files generated by WhisperX.
    - feature_dir (str): Directory containing the audio features (.npy files) generated by the feature extraction script.
    - label_dir (str): Directory where the labels will be saved.
    - label_info_dir (str): Directory where the labels information will be saved.
    - feature_config (str): Path to the feature configuration file.
    - n_process (int): Number of processes to use for feature extraction.
    """
    label_encoder = LabelEncoder(feature_config)
    label_encoder.compile_list_of_feature_files(feature_dir)
    label_encoder.compile_list_of_transcript_files(transcript_dir)
    label_encoder.sort_dataset_by_segment(annotation_path, label_info_dir)
    # label_encoder._sort_dataset_by_segment(
    #     annotation_path, transcript_dir, feature_dir)
    # label_encoder.generate_all_labels_multiprocessing(
    #     wav_list, annotation_path, transcript_dir, feature_dir, label_dir, label_info_dir, n_process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate labels for the training data.')
    parser.add_argument('--wav_list', type=str, dest='wav_list',
                        help='List of audio files to extract features from.', required=True)
    parser.add_argument('--annotation_path', type=str,
                        help='Path to the csv file containing the annotations', required=True)
    parser.add_argument('--transcript_dir', type=str,
                        help='Directory containing the transcript files generated by WhisperX', required=True)
    parser.add_argument('--feature_dir', type=str,
                        help='Directory containing the audio features (.npy files) generated by the feature extraction script', required=True)
    parser.add_argument('--label_dir', type=str,
                        help='Directory where the labels will be saved', required=True)
    parser.add_argument('--label_info_dir', type=str,
                        help='Directory where the labels information will be saved', required=True)
    parser.add_argument('--feature_config', type=str,
                        help='Path to the feature configuration file', required=True)
    parser.add_argument('--n_process', type=int, dest='n_process',
                        help='Number of processes to use for feature extraction.', default=4)

    arguments = parser.parse_args()

    wav_list = arguments.wav_list
    annotation_path = arguments.annotation_path
    transcript_dir = arguments.transcript_dir
    feature_dir = arguments.feature_dir
    label_dir = arguments.label_dir
    label_info_dir = arguments.label_info_dir
    feature_config = arguments.feature_config
    n_process = arguments.n_process

    paths = [wav_list, annotation_path, transcript_dir,
             feature_dir, label_dir, label_info_dir, feature_config]

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path "{path}" not found.')

    generate_label(wav_list, annotation_path, transcript_dir,
                   feature_dir, label_dir, label_info_dir, feature_config, n_process)
