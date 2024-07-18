import os
import argparse
import numpy as np
import pandas as pd
import configparser
import multiprocessing
import csv
from typing import List, Tuple


class LabelEncoder:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.sr = None
        self.unique_labels = []
        self.hop_length_seconds = None
        self._init_config()

    def _init_config(self) -> None:
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

    def _generate_zero_labels(self, num_frames: int) -> np.ndarray:
        unique_labels = self.unique_labels
        labels = np.zeros(
            (num_frames, len(unique_labels)), dtype=np.int32)
        return labels

    def _generate_labels_per_segment(self, feature_file: str, annotations_path: str, transcript_dir: str, label_dir: str) -> Tuple[str, float, float, int]:
        feature_file_name = os.path.basename(feature_file)
        parts = feature_file_name.rsplit('_', 1)
        audio_file_name = parts[0]
        segment_number = parts[1].replace('.npy', '')
        segment_index = int(segment_number) - 1

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
    label_encoder = LabelEncoder(feature_config)
    label_encoder.generate_all_labels_multiprocessing(
        wav_list, annotation_path, transcript_dir, feature_dir, label_dir, label_info_dir, n_process)


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
