import os
import configparser
from typing import List
import argparse
import multiprocessing
import librosa
import scipy
import numpy as np
import csv


class FeatureExtractor():
    """
    Extract features from audio files and save them to disk.

    Args:
    - config_path (str): Path to the feature extraction configuration file.

    Attributes:
    - config_path (str): Path to the feature extraction configuration file.
    - n_fft (int): Number of FFT points.
    - win_length (float): Length of the window in seconds.
    - hop_length (float): Hop length in seconds.
    - sr (int): Sampling rate.
    - n_mels (int): Number of Mel bands.
    - f_min (int): Minimum frequency.
    - f_max (int): Maximum frequency.
    - frame_length (int): Length of the frame.
    - win_length_seconds (int): Length of the window in samples.
    - hop_length_seconds (int): Hop length in samples.
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.n_fft = None
        self.win_length = None
        self.hop_length = None
        self.sr = None
        self.n_mels = None
        self.f_min = None
        self.f_max = None
        self.frame_length = None
        self.win_length_seconds = None
        self.hop_length_seconds = None

        self._init_config()

    def _init_config(self) -> None:
        """
        Initialize the feature extraction configuration.
        """
        config_path = self.config_path

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config = configparser.ConfigParser()
        config.read(config_path)

        if 'feature' not in config:
            raise ValueError(f"Config file does not contain 'feature' section")

        feature_config = config['feature']

        self.n_fft = int(feature_config.get('n_fft', 2048))
        self.win_length = float(feature_config.get('win_length', 0.04))
        self.hop_length = float(feature_config.get('hop_length', 0.02))
        self.sr = int(feature_config.get('sr', 16000))
        self.n_mels = int(feature_config.get('n_mels', 64))
        self.f_min = int(feature_config.get('f_min', 0))
        self.f_max = int(feature_config.get('f_max', 22050))

        if self.f_min < 0 or self.f_min > self.f_max or self.f_max > self.sr / 2:
            raise ValueError(
                f"Invalid frequency range: {self.f_min} to {self.f_max}")

        self.win_length_seconds = int(self.win_length * self.sr)
        self.hop_length_seconds = int(self.hop_length * self.sr)

    def _get_feature(self, input_file: str, output_file: str, transcript_file: str) -> List[np.ndarray]:
        """
        Extract features from an audio file and save them to disk.

        Args:
        - input_file (str): The path to the input audio file.
        - output_file (str): The path to save the extracted features.
        - transcript_file (str): The path to the transcript file.
        """
        sampling_rate = self.sr
        n_fft = self.n_fft
        n_mels = self.n_mels
        f_min = self.f_min
        f_max = self.f_max
        win_length_seconds = self.win_length_seconds
        hop_length_seconds = self.hop_length_seconds

        if win_length_seconds > n_fft:
            raise ValueError(
                f"Window length {win_length_seconds} is greater than n_fft {n_fft}\nIncrease n_fft or decrease win_length")

        segments = []
        with open(transcript_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                start_time = float(row[0])
                end_time = float(row[1])
                segments.append((start_time, end_time))

        y, sr = librosa.load(input_file, sr=sampling_rate)

        features_list = []

        for start_time, end_time in segments:
            segment = y[int(start_time * sr):int(end_time * sr)]

            if len(segment) < n_fft:
                segment = np.pad(
                    segment, (0, n_fft - len(segment)), mode='constant')

            window = scipy.signal.windows.hann(win_length_seconds, sym=False)

            mel_basis = librosa.filters.mel(
                sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max)
            epsilon = np.spacing(1)
            spectrogram = librosa.stft(
                segment + epsilon, n_fft=n_fft, hop_length=hop_length_seconds,
                win_length=win_length_seconds, window=window)
            spectrogram = np.abs(spectrogram)
            mel_spectrogram = np.dot(mel_basis, spectrogram)
            log_mel_spectrogram = np.log(mel_spectrogram + epsilon)

            feature = np.transpose(log_mel_spectrogram)
            features_list.append(feature)

        for i, feature in enumerate(features_list):
            file_name = f"{output_file.split('.')[0]}_{i+1:04d}.npy"
            np.save(file_name, feature)
            print(f"Saved feature to {file_name}")

        return features_list

    def _get_feature_for_single_list(self, list_audio_files: List[str], wav_dir: str, transcript_dir: str, feature_dir: str, index: int, file_per_process: int, wav_list_length: int) -> None:
        """
        Extract features from a list of audio files and save them to disk.

        Args:
        - list_audio_files (List[str]): List of audio files to extract features from.
        - wav_dir (str): Directory containing the (.wav) audio files.
        - transcript_dir (str): Directory containing the (.csv) transcript files.
        - feature_dir (str): Directory to save the extracted features.
        - index (int): The index of the process.
        - file_per_process (int): Number of files to process per process.
        - wav_list_length (int): Total number of audio files to process.
        """
        for sub_list_index, file in enumerate(list_audio_files):
            index = index * file_per_process + sub_list_index
            input_file = os.path.join(wav_dir, file)
            output_file = os.path.join(feature_dir, file)
            output_file_name_prefix = os.path.basename(
                output_file).split('.')[0]
            print(
                f"Extracting features {index + 1}/{wav_list_length}: {input_file}")

            transcript_file_name = file.replace('.wav', '.csv')
            transcript_file = self._find_transcript_file(
                transcript_dir, transcript_file_name)
            if not transcript_file:
                print(f"Transcript file not found for {input_file}")
                continue

            if not file_exists_with_prefix(feature_dir, output_file_name_prefix):
                self._get_feature(input_file, output_file, transcript_file)
                print(
                    f"Extracting features {index + 1}/{wav_list_length}: {input_file} completed")
            else:
                print(
                    f"Extracting features {index + 1}/{wav_list_length}: {input_file} already exists")

    def generate_feature_multiprocessing(self, list_audio_files: str, wav_dir: str, transcript_dir: str, feature_dir: str, n_process: int) -> None:
        """
        Extract features from a list of audio files and save them to disk using multiprocessing.

        Args:
        - list_audio_files (str): Path to the list of audio files to extract features from.
        - wav_dir (str): Directory containing the (.wav) audio files.
        - transcript_dir (str): Directory containing the (.csv) transcript files.
        - feature_dir (str): Directory to save the extracted features.
        - n_process (int): Number of processes to use for feature extraction.
        """
        with open(list_audio_files, 'r') as f:
            wav_list = f.readlines()

        wav_list = [wav.strip() for wav in wav_list]

        file_per_process = (len(wav_list) + n_process - 1) // n_process

        for process_index in range(n_process):
            start_index = process_index * file_per_process
            end_index = min((process_index + 1) *
                            file_per_process, len(wav_list))

            if start_index < len(wav_list) and start_index < end_index:
                sub_list = wav_list[start_index:end_index]
                process = multiprocessing.Process(target=self._get_feature_for_single_list, args=(
                    sub_list, wav_dir, transcript_dir, feature_dir, process_index, file_per_process, len(wav_list)))

                process.start()
                print(
                    f"Process {process_index + 1}/{min(n_process, len(wav_list))} started")

    def _find_transcript_file(self, transcript_dir: str, transcript_file_name: str) -> str:
        """
        Find the transcript file in the transcript directory.

        Args:
        - transcript_dir (str): Directory containing the transcript files.
        - transcript_file_name (str): Name of the transcript file.

        Returns:
        - str: The path to the transcript file.
        """
        for root, dirs, files in os.walk(transcript_dir):
            if transcript_file_name in files:
                return os.path.join(root, transcript_file_name)
        return None


def file_exists_with_prefix(directory: str, output_file_prefix: str) -> bool:
    """
    Check if a file with the given prefix exists in the directory.

    Args:
    - directory (str): The directory to search for the file.
    - output_file_prefix (str): The prefix of the file to search for.

    Returns:
    - bool: True if the file exists, False otherwise.
    """
    for filename in os.listdir(directory):
        if filename.startswith(output_file_prefix):
            return True
    return False


def extract_feature(wav_list: List[str], wav_dir: str, transcript_dir: str, feature_dir: str, feature_config_path: str, n_process: int) -> None:
    """
    Extract features from a list of audio files and save them to disk.

    Args:
    - wav_list (List[str]): List of audio files to extract features from.
    - wav_dir (str): Directory containing the (.wav) audio files.
    - transcript_dir (str): Directory containing the (.csv) transcript files.
    - feature_dir (str): Directory to save the extracted features.
    - feature_config_path (str): Path to the feature extraction configuration file.
    - n_process (int): Number of processes to use for feature extraction.
    """
    feature_extractor = FeatureExtractor(feature_config_path)
    feature_extractor.generate_feature_multiprocessing(
        wav_list, wav_dir, transcript_dir, feature_dir, n_process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract features from audio files and save them to disk.')
    parser.add_argument('--wav_list', type=str, dest='wav_list',
                        help='List of audio files to extract features from.', required=True)
    parser.add_argument('--wav_dir', type=str, dest='wav_dir',
                        help='Directory containing the (.wav) audio files.', required=True)
    parser.add_argument('--transcript_dir', type=str, dest='transcript_dir',
                        help='Directory containing the (.csv) transcript files.', required=True)
    parser.add_argument('--feature_dir', type=str, dest='feature_dir',
                        help='Directory to save the extracted features.', required=True)
    parser.add_argument('--feature_config', type=str, dest='feature_config',
                        help='Path to the feature extraction configuration file.', required=True)
    parser.add_argument('--n_process', type=int, dest='n_process',
                        help='Number of processes to use for feature extraction.', default=4)

    arguments = parser.parse_args()

    wav_list = arguments.wav_list
    wav_dir = arguments.wav_dir
    transcript_dir = arguments.transcript_dir
    feature_dir = arguments.feature_dir
    feature_config = arguments.feature_config
    n_process = arguments.n_process

    paths = [wav_list, wav_dir, transcript_dir, feature_dir, feature_config]

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path "{path}" not found.')

    extract_feature(wav_list, wav_dir, transcript_dir,
                    feature_dir, feature_config, n_process)
