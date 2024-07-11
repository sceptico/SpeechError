"""
This script is used to convert .mp3 files to .wav files.

Usage:
    python convert_mp3_to_wav.py --audio_dir <audio_dir> --output_dir <output_dir> --sample_rate <sample_rate>
    
    - audio_dir (str): The directory containing the .mp3 files.
    - output_dir (str): The directory to save the converted .wav files.
    - sample_rate (int): The sample rate of the output .wav files.
    
Example:
    python convert_mp3_to_wav.py --audio_dir /data/audio --output_dir /data/audio_wav --sample_rate 16000
"""

import os
import librosa
import soundfile as sf
import argparse


def convert_mp3_to_wav(audio_dir: str, output_dir: str, sample_rate: int) -> None:
    """
    Convert .mp3 files to .wav files.

    Args:
    - audio_dir (str): The directory containing the .mp3 files.
    - output_dir (str): The directory to save the converted .wav files.
    - sample_rate (int): The sample rate of the output .wav files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.mp3'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(
                    output_dir, file.replace('.mp3', '.wav'))

                # Check if .wav file already exists
                if os.path.exists(output_path):
                    print(f'{output_path} already exists, skipping...')
                    continue

                # Load the .mp3 file
                y, sr = librosa.load(input_path, sr=sample_rate)

                # Save as .wav file
                sf.write(output_path, y, sample_rate)
                print(f'Converted {input_path} to {output_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert .mp3 files to .wav files.')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='The directory containing the .mp3 files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The directory to save the converted .wav files.')
    parser.add_argument('--sample_rate', type=int, required=True,
                        help='The sample rate of the output .wav files.')

    args = parser.parse_args()

    convert_mp3_to_wav(args.audio_dir, args.output_dir, args.sample_rate)
