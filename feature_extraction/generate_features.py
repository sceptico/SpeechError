
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
        self.init_extractor_config()

    def init_extractor_config(self) -> None:
        """
        Initialize the feature extraction configuration file.

        Parameters:
            None

        Return:
            None
        """
        # Load the configuration file
        config_path = self.config_path
        assert os.path.exists(
            config_path), f"Config file not found at {config_path}"
        config = configparser.ConfigParser()
        config.read(config_path)
        assert 'feature' in config, f"Config file does not contain 'feature' section"
        feature_config = config['feature']

        # Set the configuration parameters
        self.n_fft = int(feature_config.get('n_fft', 2048))
        self.win_length = float(feature_config.get('win_length', 0.04))
        self.hop_length = float(feature_config.get('hop_length', 0.02))
        self.sr = int(feature_config.get('sr', 16000))
        self.n_mels = int(feature_config.get('n_mels', 64))
        self.f_min = int(feature_config.get('f_min', 0))
        self.f_max = int(feature_config.get('f_max', 22050))
        assert self.max > self.min, f"Invalid frequency range: {self.min} to {self.max}"
        self.output_frames = int(feature_config.get('LEN', 400))

        # Calculate the frame length, window length, and hop length in seconds
        self.win_length_seconds = self.win_length * self.sr
        self.hop_length_seconds = self.hop_length * self.sr

    def get_feature(self, input_file: str, output_file: str) -> np.ndarray:
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
        spectogram = np.abs(mel_spectogram)
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
        np.save(output_file, feature_output)
        return feature_output

    def get_feature_for_single_list(self, list: List[str], wav_dir: str, feature_dir: str, index: int) -> None:
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
                self.get_feature(input_file, output_file)
                print(
                    f"Extracting features {index}/{len(list)}: {input_file} completed")
            else:
                print(
                    f"Extracting features {index}/{len(list)}: {input_file} already exists")

    def get_feature_multiprocessing(self, list: List[str], wav_dir: str, feature_dir: str, n_process: int) -> None:
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
                process = multiprocessing.Process(target=self.get_feature_for_single_list, args=(
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
    feature_extractor.get_feature_multiprocessing(
        wav_list, wav_dir, feature_dir,  n_process)


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
