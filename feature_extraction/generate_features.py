"""
This script is used to extract features from audio files and save them to disk.

Usage:
    python generate_features.py --wav_list <wav_list> --wav_dir <wav_dir> --feature_dir <feature_dir> --feature_config <feature_config> --n_process <n_process>
    
    - wav_list (str): List of audio files to extract features from.
    - wav_dir (str): Directory containing the audio files.
    - feature_dir (str): Directory to save the extracted features.
    - feature_config (str): Path to the feature extraction configuration file.
    - n_process (int): Number of processes to use for feature extraction.
    
Example:
    python generate_features.py --wav_list data/metadata/wav_list.lst --wav_dir data/audio --feature_dir data/feature --feature_config feature_extraction/feature.cfg --n_process 4

References:
    - (https://github.com/Kikyo-16/Sound_event_detection/blob/master/feature_extraction/gen_feature.py)
"""


import os
import configparser
from typing import List
import argparse
import multiprocessing
import librosa
import scipy
import numpy as np


class FeatureExtractor():
    def __init__(self, config_path: str) -> None:
        """
        FeatureExtractor Class
        ======================

        The FeatureExtractor class is used to extract features from audio files and save them to disk.

        Public Methods
        --------------
        - extract_feature(wav_list: List[str], wav_dir: str, feature_dir: str, feature_config_path: str, n_process: int) -> None

        Public Properties
        -----------------
        - n_fft: Number of points in the Fast Fourier Transform.
        - win_length: Window length in seconds.
        - hop_length: Hop length in seconds.
        - sr: Sampling rate of the audio files.
        - n_mels: Number of Mel bands to generate.
        - f_min: Minimum frequency.
        - f_max: Maximum frequency.
        - frame_length: Length of the frame.
        - win_length_seconds: Window length in seconds.
        - hop_length_seconds: Hop length in seconds.

        Dependencies
        ------------
        - [os](https://docs.python.org/3/library/os.html)
        - [configparser](https://docs.python.org/3/library/configparser.html)
        - [numpy](https://numpy.org/)
        - [librosa](https://librosa.org/doc/main/index.html)
        - [scipy](https://docs.scipy.org/doc/scipy/reference/index.html)
        - [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
        - [typing](https://docs.python.org/3/library/typing.html)

        References:
        -----------
        - (https://github.com/Kikyo-16/Sound_event_detection/blob/master/feature_extraction/gen_feature.py)

        """
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

        # Initialize the feature extraction configuration
        self._init_extractor_config()

    def _init_extractor_config(self) -> None:
        """
        Initialize the feature extraction configuration file.

        Parameters:
            None

        Return:
            None
        """
        # Load the configuration file
        config_path = self.config_path

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        config = configparser.ConfigParser()
        config.read(config_path)

        if 'feature' not in config:
            raise ValueError(f"Config file does not contain 'feature' section")

        feature_config = config['feature']

        # Set the configuration parameters
        self.n_fft = int(feature_config.get('n_fft', 2048))
        self.win_length = float(feature_config.get('win_length', 0.04))
        self.hop_length = float(feature_config.get('hop_length', 0.02))
        self.sr = int(feature_config.get('sr', 44100))
        self.n_mels = int(feature_config.get('n_mels', 64))
        self.f_min = int(feature_config.get('f_min', 0))
        self.f_max = int(feature_config.get('f_max', 22050))

        if self.f_min < 0 or self.f_min > self.f_max or self.f_max > self.sr / 2:
            raise ValueError(
                f"Invalid frequency range: {self.f_min} to {self.f_max}")

        self.output_frames = int(feature_config.get('LEN', 400))

        # Calculate the frame length, window length, and hop length in seconds
        self.win_length_seconds = int(self.win_length * self.sr)
        self.hop_length_seconds = int(self.hop_length * self.sr)

    def _get_feature(self, input_file: str, output_file: str) -> np.ndarray:
        """
        Extract features from an audio file and save them to disk.

        Parameters:
            input_file (str): Path to the input audio file.
            output_file (str): Path to save the extracted features.

        Return:
            feature_output (np.ndarray): Extracted features from the audio file.
        """
        sampling_rate = self.sr
        n_fft = self.n_fft  # number of points in the Fast Fourier Transform
        n_mels = self.n_mels  # number of Mel bands to generate
        f_min = self.f_min  # minimum frequency
        f_max = self.f_max  # maximum frequency
        win_length_seconds = self.win_length_seconds  # window length in seconds
        hop_length_seconds = self.hop_length_seconds  # hop length in seconds
        output_length = self.output_frames  # number of frames to output

        # Load the audio file
        y, sr = librosa.load(input_file, sr=sampling_rate)

        # Apply Hanning window
        window = scipy.signal.hann(win_length_seconds, sym=False)

        # Compute the Mel spectrogram
        mel_basis = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max)
        epsilon = np.spacing(1)
        spectogram = librosa.stft(
            y + epsilon, n_fft=n_fft, hop_length=hop_length_seconds, win_length=win_length_seconds, window=window)
        spectogram = np.abs(spectogram)
        mel_spectogram = np.dot(mel_basis, spectogram)
        log_mel_spectogram = np.log(mel_spectogram + epsilon)

        feature = np.transpose(log_mel_spectogram)

        # Use 10 seconds of audio as input
        frame_length = int(sampling_rate * 10 / hop_length_seconds + 1)
        feature_temp = np.zeros((frame_length, feature.shape[1]))

        if feature.shape[0] < frame_length:
            feature_temp[:feature.shape[0]] = feature

        else:
            feature_temp = feature[:frame_length]

        feature_output = np.zeros((output_length, feature.shape[1]))

        # When the frame length not matches the output length
        if not frame_length == output_length:

            # If the frame length is less than the output length, pad and center the feature
            if frame_length < output_length:
                start_index = (output_length - frame_length) // 2
                end_index = output_length - start_index
                feature_output[start_index:end_index] = feature_temp

            # If the frame length is greater than the output length, crop the feature
            else:
                start_index = (frame_length - output_length) // 2
                end_index = frame_length - start_index
                feature_output = feature_temp[start_index:end_index]

        else:
            feature_output = feature_temp

        # Save the extracted features to disk
        np.save(output_file.split('.')[0], feature_output)
        return feature_output

    def _get_feature_for_single_list(self, list: List[str], wav_dir: str, feature_dir: str, index: int) -> None:
        """
        Extract features for a list of audio files and save them to disk.

        Parameters:
            list (list): List of audio files to extract features from.
            wav_dir (str): Directory containing the audio files.
            feature_dir (str): Directory to save the extracted features.
            index (int): Index of the list of audio files.

        Return:
            None
        """
        for file in list:
            input_file = os.path.join(wav_dir, file)
            output_file = os.path.join(feature_dir, file)
            print(f"Extracting features {index}/{len(list)}: {input_file}")

            if not os.path.exists(output_file):
                self._get_feature(input_file, output_file)
                print(
                    f"Extracting features {index}/{len(list)}: {input_file} completed")
            else:
                print(
                    f"Extracting features {index}/{len(list)}: {input_file} already exists")

    def _get_feature_multiprocessing(self, list: List[str], wav_dir: str, feature_dir: str, n_process: int) -> None:
        """
        Extract features for a list of audio files using multiprocessing.

        Parameters:
            list (list): List of audio files to extract features from.
            wav_dir (str): Directory containing the audio files.
            feature_dir (str): Directory to save the extracted features.
            n_process (int): Number of processes to use for feature extraction.

        Return:
            None
        """
        with open(list, 'r') as f:
            wav_list = f.readlines()

        wav_list = [wav.strip() for wav in wav_list]

        file_per_process = (len(list) + n_process - 1) // n_process

        for process_index in range(n_process):
            start_index = process_index * file_per_process
            end_index = min((process_index + 1) * file_per_process, len(list))

            # Check if the start index is within the list
            if start_index < len(list) and start_index < end_index:
                sub_list = wav_list[start_index:end_index]
                process = multiprocessing.Process(target=self._get_feature_for_single_list, args=(
                    sub_list, wav_dir, feature_dir, process_index + 1))

                # Start the process
                process.start()
                print(f"Process {process_index + 1}/{n_process} started")


def extract_feature(wav_list: List[str], wav_dir: str, feature_dir: str, feature_config_path: str, n_process: int) -> None:
    """
    Generate features for a list of audio files and save them to disk.

    Parameters:
        wav_list (list): List of audio files to extract features from.
        wav_dir (str): Directory containing the audio files.
        feature_dir (str): Directory to save the extracted features.
        feature_config_path (str): Path to the feature extraction configuration file.
        n_process (int): Number of processes to use for feature extraction.

    Return:
        None
    """
    feature_extractor = FeatureExtractor(feature_config_path)
    feature_extractor._get_feature_multiprocessing(
        wav_list, wav_dir, feature_dir,  n_process)


@property
def get_n_fft(self) -> int:
    """
    Get the number of points in the Fast Fourier Transform.

    Parameters:
        None

    Returns:
        n_fft (int): Number of points in the Fast Fourier Transform.
    """
    return self.n_fft


@property
def get_win_length(self) -> float:
    """
    Get the window length in seconds.

    Parameters:
        None

    Returns:
        win_length (float): Window length in seconds.
    """
    return self.win_length


@property
def get_hop_length(self) -> float:
    """
    Get the hop length in seconds.

    Parameters:
        None

    Returns:
        hop_length (float): Hop length in seconds.
    """
    return self.hop_length


@property
def get_sr(self) -> int:
    """
    Get the sampling rate of the audio files.

    Parameters:
        None

    Returns:
        sr (int): Sampling rate of the audio files.
    """
    return self.sr


@property
def get_n_mels(self) -> int:
    """
    Get the number of Mel bands to generate.

    Parameters:
        None

    Returns:
        n_mels (int): Number of Mel bands to generate.
    """
    return self.n_mels


@property
def get_f_min(self) -> int:
    """
    Get the minimum frequency.

    Parameters:
        None

    Returns:
        f_min (int): Minimum frequency.
    """
    return self.f_min


@property
def get_f_max(self) -> int:
    """
    Get the maximum frequency.

    Parameters:
        None

    Returns:
        f_max (int): Maximum frequency.
    """
    return self.f_max


@property
def get_frame_length(self) -> int:
    """
    Get the length of the frame.

    Parameters:
        None

    Returns:    
        frame_length (int): Length of the frame.       
    """
    return self.frame_length


@property
def get_win_length_seconds(self) -> int:
    """
    Get the window length in seconds.

    Parameters:
        None

    Returns:
        win_length_seconds (int): Window length in seconds.
    """
    return self.win_length_seconds


@property
def get_hop_length_seconds(self) -> int:
    """
    Get the hop length in seconds.

    Parameters:
        None

    Returns:
        hop_length_seconds (int): Hop length in seconds.
    """
    return self.hop_length_seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract features from audio files and save them to disk.')
    parser.add_argument('--wav_list', type=str, dest='wav_list',
                        help='List of audio files to extract features from.', required=True)
    parser.add_argument('--wav_dir', type=str, dest='wav_dir',
                        help='Directory containing the audio files.', required=True)
    parser.add_argument('--feature_dir', type=str, dest='feature_dir',
                        help='Directory to save the extracted features.', required=True)
    parser.add_argument('--feature_config', type=str, dest='feature_config',
                        help='Path to the feature extraction configuration file.', required=True)
    parser.add_argument('--n_process', type=int, dest='n_process',
                        help='Number of processes to use for feature extraction.', default=4)

    arguments = parser.parse_args()

    wav_list = arguments.wav_list
    wav_dir = arguments.wav_dir
    feature_dir = arguments.feature_dir
    feature_config = arguments.feature_config
    n_process = arguments.n_process

    paths = [wav_list, wav_dir, feature_dir, feature_config]

    for path in paths:
        assert os.path.exists(path), f"Path not found: {path}"

    extract_feature(wav_list, wav_dir, feature_dir, feature_config, n_process)
